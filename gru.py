import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# ==== 1) Column setup ====
TARGET = 'Is Account Takeover'  # exact name from your dataframe
DROP_COLS = ['index', 'Login Timestamp', 'User ID']  # drop as predictors (keep TS + UID for sorting/grouping)

# Categorical features -> will use embeddings
CAT_COLS = [
    'Country',
    'Region',
    'City',
    'User Agent String',
    'Browser Name and Version',
    'OS Name and Version',
    'Device Type',
    'Success Group',
]

# Binary/features we’ll treat as numeric floats (0/1)
BIN_COLS = [
    'Login Successful',
    'Is Attack IP',
    'Is First Login',
    'Has IP Changed',
    'Country Changed',
    'Browser Changed'
]

# Numeric features
NUM_COLS = [
    'Time Since Last Login',
    'Login Hour',
    'Login Day of Week',
    'Failed Logins Since Last Success'
]

def prepare_dataframe(df: pd.DataFrame):
    # Make sure timestamp is datetime (used only for sorting)
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])

    # Ensure binary columns are 0/1 floats
    for c in BIN_COLS:
        df[c] = df[c].astype(float)

    # Factorize categoricals with PAD=0, UNK=1, real classes start at 2
    cats_info = {}  # col -> {"vocab_size": int}
    for c in CAT_COLS:
        codes, uniques = pd.factorize(df[c].astype('string'), sort=False)
        # pandas factorize gives -1 for NaN; shift by +2 so: -1 -> 1 (UNK), 0 -> 2, ...
        df[c + '__code'] = (codes + 2).astype('int64')
        cats_info[c] = {"vocab_size": len(uniques) + 2}  # +2 for PAD(0) and UNK(1)

    # Standardize the true numeric columns (fit on train in real projects!)
    scaler = StandardScaler()
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])

    return df, cats_info, scaler

def build_windows(df: pd.DataFrame, window_size=20):
    # Sort within each user by time
    df = df.sort_values(['User ID', 'Login Timestamp'])

    cat_code_cols = [c + '__code' for c in CAT_COLS]
    num_all_cols = BIN_COLS + NUM_COLS

    X_cats, X_nums, y, end_ts = [], [], [], []

    for uid, g in df.groupby('User ID', sort=False):
        g = g.sort_values('Login Timestamp')

        cat_mat_full = g[cat_code_cols].to_numpy(dtype=np.int64)            # (n, C)
        num_mat_full = g[num_all_cols].to_numpy(dtype=np.float32)           # (n, D)
        labels       = g[TARGET].astype('int64').to_numpy()                 # (n,)
        ts           = g['Login Timestamp'].astype('int64').to_numpy()      # ns since epoch -> int64

        n = len(g)
        for end in range(n):
            start = max(0, end - window_size + 1)
            L = end - start + 1

            # Left-pad to fixed window_size
            cats_win = np.zeros((window_size, len(CAT_COLS)), dtype=np.int64)      # PAD=0
            nums_win = np.zeros((window_size, len(num_all_cols)), dtype=np.float32)

            cats_win[-L:] = cat_mat_full[start:end+1]
            nums_win[-L:] = num_mat_full[start:end+1]

            X_cats.append(cats_win)
            X_nums.append(nums_win)
            y.append(labels[end])             # label for current timestep
            end_ts.append(ts[end])            # window ends at this timestamp

    X_cats = np.stack(X_cats)                      # (N, T, C)
    X_nums = np.stack(X_nums)                      # (N, T, D)
    y = np.asarray(y, dtype=np.float32)            # BCE expects float
    end_ts = np.asarray(end_ts, dtype=np.int64)    # for chronological split
    return X_cats, X_nums, y, end_ts

# ==== 4) PyTorch Dataset ====
class SeqDataset(Dataset):
    def __init__(self, X_cats, X_nums, y):
        self.X_cats = torch.from_numpy(X_cats)  # long
        self.X_nums = torch.from_numpy(X_nums)  # float
        self.y = torch.from_numpy(y)            # float

    def __len__(self):
        return self.X_cats.shape[0]

    def __getitem__(self, idx):
        return {
            "cats": self.X_cats[idx].long(),   # (T, C)
            "nums": self.X_nums[idx].float(),  # (T, D)
            "y": self.y[idx].float(),          # scalar
        }

# ==== 5) GRU model with embeddings for categoricals ====
class GRUTakeoverModel(nn.Module):
    def __init__(self, cats_info, num_dim, hidden_size=128, num_layers=1, dropout=0.1, window_size=20):
        super().__init__()
        self.window_size = window_size

        # Build one embedding per categorical column
        self.cat_cols = list(cats_info.keys())
        self.emb_layers = nn.ModuleList()
        emb_dims = []
        for c in self.cat_cols:
            vocab = cats_info[c]["vocab_size"]
            # A simple heuristic for embedding dim; adjust as you like
            emb_dim = min(32, max(4, vocab // 4))
            self.emb_layers.append(nn.Embedding(num_embeddings=vocab, embedding_dim=emb_dim, padding_idx=0))
            emb_dims.append(emb_dim)

        self.total_in_dim = sum(emb_dims) + num_dim
        self.gru = nn.GRU(
            input_size=self.total_in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)  # binary logit
        )

    def forward(self, cats, nums):
        # cats: (B, T, C) longs; nums: (B, T, D) floats
        embs = []
        # apply embedding for each categorical column at each timestep
        for i, emb in enumerate(self.emb_layers):
            embs.append(emb(cats[:, :, i]))  # (B, T, emb_dim)
        cat_emb = torch.cat(embs, dim=-1)      # (B, T, sum_emb_dim)

        x = torch.cat([cat_emb, nums], dim=-1) # (B, T, total_in_dim)
        out, _ = self.gru(x)                   # out: (B, T, H)
        last = out[:, -1, :]                   # use last timestep hidden
        logit = self.head(last).squeeze(-1)    # (B,)
        return logit

# ==== Training and Evaluation Utilities ====
def stratified_chronological_split(X_cats, X_nums, y, end_ts, train_pos_ratio=0.1, val_pos_ratio=0.1, test_pos_ratio=0.1):
    """
    Chronologically splits the data, but adjusts boundaries so each split has approximately the requested positive class ratio.
    Prints the actual percentage of total data in each split.
    """
    order = np.argsort(end_ts)
    X_cats = X_cats[order]
    X_nums = X_nums[order]
    y      = y[order]
    N = len(y)
    pos_idx = np.where(y == 1)[0]
    n_pos = len(pos_idx)
    # Calculate desired number of positives per split
    train_pos = int(n_pos * train_pos_ratio)
    val_pos   = int(n_pos * val_pos_ratio)
    test_pos  = n_pos - train_pos - val_pos
    # Find split indices chronologically
    pos_counter = 0
    train_end = 0
    val_end = 0
    for i in range(N):
        if y[i] == 1:
            pos_counter += 1
        if pos_counter == train_pos:
            train_end = i + 1
            break
    pos_counter2 = 0
    for i in range(train_end, N):
        if y[i] == 1:
            pos_counter2 += 1
        if pos_counter2 == val_pos:
            val_end = i + 1
            break
    if train_end == 0:
        train_end = int(N * 0.5)
    if val_end == 0:
        val_end = train_end + int(N * 0.15)
    idxs = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, N)
    }
    # Print actual split sizes and positive ratios
    def pct(n):
        return 100.0 * n / N
    train_pos_actual = y[idxs['train'][0]:idxs['train'][1]].sum()
    val_pos_actual = y[idxs['val'][0]:idxs['val'][1]].sum()
    test_pos_actual = y[idxs['test'][0]:idxs['test'][1]].sum()
    print(f"Train: {idxs['train'][1]-idxs['train'][0]} samples ({pct(idxs['train'][1]-idxs['train'][0]):.2f}%), Positives: {train_pos_actual} ({pct(train_pos_actual):.2f}%)")
    print(f"Val:   {idxs['val'][1]-idxs['val'][0]} samples ({pct(idxs['val'][1]-idxs['val'][0]):.2f}%), Positives: {val_pos_actual} ({pct(val_pos_actual):.2f}%)")
    print(f"Test:  {idxs['test'][1]-idxs['test'][0]} samples ({pct(idxs['test'][1]-idxs['test'][0]):.2f}%), Positives: {test_pos_actual} ({pct(test_pos_actual):.2f}%)")
    return X_cats, X_nums, y, idxs

def chronological_split(X_cats, X_nums, y, end_ts, train_frac=0.5, val_frac=0.15):
    order = np.argsort(end_ts)
    X_cats = X_cats[order]
    X_nums = X_nums[order]
    y      = y[order]
    N = len(y)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    n_test  = N - n_train - n_val
    idxs = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, N)
    }
    return X_cats, X_nums, y, idxs

def create_datasets(X_cats, X_nums, y, idxs):
    train_ds = SeqDataset(X_cats[idxs['train'][0]:idxs['train'][1]], X_nums[idxs['train'][0]:idxs['train'][1]], y[idxs['train'][0]:idxs['train'][1]])
    valid_ds = SeqDataset(X_cats[idxs['val'][0]:idxs['val'][1]], X_nums[idxs['val'][0]:idxs['val'][1]], y[idxs['val'][0]:idxs['val'][1]])
    test_ds  = SeqDataset(X_cats[idxs['test'][0]:idxs['test'][1]], X_nums[idxs['test'][0]:idxs['test'][1]], y[idxs['test'][0]:idxs['test'][1]])
    return train_ds, valid_ds, test_ds

def create_loaders(train_ds, valid_ds, test_ds, train_bs=128, val_bs=256, test_bs=256):
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True,  drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=val_bs, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=test_bs, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(cats_info, num_dim, window_size, device, hidden_size=128, num_layers=1, dropout=0.1):
    model = GRUTakeoverModel(
        cats_info=cats_info, num_dim=num_dim,
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, window_size=window_size
    ).to(device)
    return model

def get_pos_weight(y_train, device):
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    if pos == 0:
        print("Warning: no positive samples in training; setting pos_weight=1.0")
        return torch.tensor(1.0, device=device)
    else:
        return torch.tensor(neg / max(pos, 1.0), device=device)

def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            cats   = batch['cats'].to(device)
            nums   = batch['nums'].to(device)
            target = batch['y'].to(device)
            logits = model(cats, nums)
            loss   = criterion(logits, target)
            bs     = target.shape[0]
            total += loss.item() * bs
            count += bs
    return total / max(count, 1)

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=5):
    best_val = float('inf')
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        for batch in train_loader:
            cats   = batch['cats'].to(device)
            nums   = batch['nums'].to(device)
            target = batch['y'].to(device)

            optimizer.zero_grad()
            logits = model(cats, nums)
            loss   = criterion(logits, target)
            loss.backward()
            optimizer.step()

            bs = target.shape[0]
            train_total  += loss.item() * bs
            train_count  += bs

        train_loss = train_total / max(train_count, 1)
        val_loss   = evaluate_loss(model, valid_loader, criterion, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val

def collect_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for b in loader:
            p = torch.sigmoid(model(b['cats'].to(device), b['nums'].to(device))).cpu().numpy()
            probs.append(p)
            ys.append(b['y'].numpy())
    return np.concatenate(probs), np.concatenate(ys)

def print_metrics(val_y, val_p, test_y, test_p):
    print("Val AUROC :", roc_auc_score(val_y, val_p))
    print("Val AUPRC :", average_precision_score(val_y, val_p))
    print("Val logloss (unweighted):", log_loss(val_y, val_p))
    print("Test AUROC:", roc_auc_score(test_y, test_p))
    print("Test AUPRC:", average_precision_score(test_y, test_p))
    print("Test logloss (unweighted):", log_loss(test_y, test_p))

def print_positive_counts(y, idxs):
    print(y[idxs['train'][0]:idxs['train'][1]].sum())
    print(y[idxs['val'][0]:idxs['val'][1]].sum())
    print(y[idxs['test'][0]:idxs['test'][1]].sum())