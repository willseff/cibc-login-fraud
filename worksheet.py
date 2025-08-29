
# %%
# read this file C:\Users\LiW\.cache\kagglehub\datasets\dasgroup\rba-dataset\versions\1

import polars as pl

df = pl.read_csv("rba-dataset.csv")

#%% 


positive_count = (df["Is Account Takeover"] == True).sum()
total_count = df.height
percentage = (positive_count / total_count) * 100

print(f"Percentage of positive 'Is Account Takeover': {percentage:.2f}%")

# %%


#%% 
df = df.drop("Round-Trip Time [ms]")


# %%

import numpy as np 

# Find User IDs with at least one positive case
positive_user_ids = df.filter(pl.col("Is Account Takeover") == 1)["User ID"].unique().to_numpy()

# Find User IDs with only negative cases
all_user_ids = df["User ID"].unique().to_numpy()
negative_user_ids = np.setdiff1d(all_user_ids, positive_user_ids)

# Choose how many negative User IDs to keep (e.g., 10x the number of positive User IDs)
n_neg_keep = len(positive_user_ids) * 100
np.random.seed(42)
neg_keep_ids = np.random.choice(negative_user_ids, n_neg_keep, replace=False)

# Combine User IDs to keep
keep_ids = np.concatenate([positive_user_ids, neg_keep_ids])

# Filter DataFrame to keep only selected User IDs
downsampled_df = df.filter(pl.col("User ID").is_in(keep_ids))

print(downsampled_df)

# %% 
# see the balance of the downsampled dataset
downsampled_positive_count = (downsampled_df["Is Account Takeover"] == True).sum()
downsampled_total_count = downsampled_df.height
downsampled_percentage = (downsampled_positive_count / downsampled_total_count) * 100

print(f"Percentage of positive 'Is Account Takeover' in downsampled dataset: {downsampled_percentage:.2f}%")

# %%
pl.Config.set_tbl_cols(20)
# see the percentage of each column that is null in downsampled dataset
null_percentages = downsampled_df.null_count() / downsampled_df.height * 100
print("Percentage of null values in each column of downsampled dataset:")
print(null_percentages)

# %%

# names of the columns
print("Column names in downsampled dataset:")
print(downsampled_df.columns)

#%%
final_logins_df = (
    downsampled_df
    .with_columns([
        pl.col("Login Timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f")
    ])
    .sort(["User ID", "Login Timestamp"])
    .with_columns([
        (pl.col("Login Timestamp") - pl.col("Login Timestamp").shift(1)).over("User ID").alias("Time Since Last Login")
    ])
)

# fill na with 0
final_logins_df = final_logins_df.with_columns([
    pl.col("Time Since Last Login").fill_null(pl.duration(milliseconds=0))
])

final_logins_df = final_logins_df.with_columns([
    pl.col("Time Since Last Login").dt.total_seconds().alias("Time Since Last Login")
])

# %%
final_logins_df = final_logins_df.with_columns([
    (pl.col("Time Since Last Login").is_null()).alias("Is First Login")
])



# %% 
for col in ["Region", "City", "Device Type"]:
    final_logins_df = final_logins_df.with_columns([
        pl.col(col).fill_null("null")
    ])


# %% okay now make login hour
final_logins_df = final_logins_df.with_columns([
    pl.col("Login Timestamp").dt.hour().alias("Login Hour")
])

# %%
# make login day of week
final_logins_df = final_logins_df.with_columns([
    pl.col("Login Timestamp").dt.weekday().alias("Login Day of Week")
])

# %%

# make flag for if the IP has changed
final_logins_df = final_logins_df.with_columns([
    (pl.col("IP Address") != pl.col("IP Address").shift(1)).over("User ID").alias("Has IP Changed")
])

# fill na with false
final_logins_df = final_logins_df.with_columns([
    pl.col("Has IP Changed").fill_null(False)
])

# %%
# make flag if the country has changed
final_logins_df = final_logins_df.with_columns([
    (pl.col("Country") != pl.col("Country").shift(1).over("User ID")).alias("Country Changed")
])

# fill na with false
final_logins_df = final_logins_df.with_columns([
    pl.col("Country Changed").fill_null(False)
])


# %%
# Ensure Login Successful is boolean
final_logins_df = final_logins_df.with_columns([
    pl.col("Login Successful").cast(pl.Boolean)
])

# Cumulative sum of successful logins per user (acts as a group id)
final_logins_df = final_logins_df.with_columns([
    pl.col("Login Successful").cum_sum().over("User ID").alias("Success Group")
])

# For each group, count failed logins (Login Successful == False)
final_logins_df = final_logins_df.with_columns([
    ((pl.col("Login Successful") == False).cast(pl.Int32)).sum().over(["User ID", "Success Group"]).alias("Failed Logins Since Last Success")
])

# fill na with 0
final_logins_df = final_logins_df.with_columns([
    pl.col("Failed Logins Since Last Success").fill_null(0)
])

# %% 
final_logins_df = final_logins_df.with_columns([
    (pl.col("Browser Name and Version") != pl.col("Browser Name and Version").shift(1).over("User ID")).alias("Browser Changed")
])

# fill na with 0
final_logins_df = final_logins_df.with_columns([
    pl.col("Browser Changed").fill_null(False)
])

#%%
final_logins_df

# %%

# drop Round-Trip Time [ms], IP Address, ASN
final_logins_df = final_logins_df.drop(["IP Address", "ASN"])

# %%
# drop Browser Name and Version
final_logins_df.columns

# %%
df.dtypes



# %%
# ==== 0) Imports ====
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

df = final_logins_df.to_pandas()
df_prep, cats_info, scaler = prepare_dataframe(df)

# %%
# get number of na in each col
df_prep

# %%

def build_windows(df: pd.DataFrame, window_size=20):
    # Sort within each user by time
    df = df.sort_values(['User ID', 'Login Timestamp'])

    cat_code_cols = [c + '__code' for c in CAT_COLS]
    num_all_cols = BIN_COLS + NUM_COLS

    X_cats, X_nums, y = [], [], []

    for uid, g in df.groupby('User ID', sort=False):
        g = g.sort_values('Login Timestamp')

        cat_mat_full = g[cat_code_cols].to_numpy(dtype=np.int64)            # (n, C)
        num_mat_full = g[num_all_cols].to_numpy(dtype=np.float32)           # (n, D)
        labels       = g[TARGET].astype('int64').to_numpy()                 # (n,)

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
            y.append(labels[end])  # predict label for the current (last) step

    X_cats = np.stack(X_cats)  # (N, T, C)
    X_nums = np.stack(X_nums)  # (N, T, D)
    y = np.asarray(y, dtype=np.float32)  # BCE loss expects float
    return X_cats, X_nums, y

WINDOW = 20
X_cats, X_nums, y = build_windows(df_prep, window_size=WINDOW)


# %%  gru model 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


# %%
# 6c) Train/val split (time-based split recommended in practice; here simple split)
n = len(y)
idx = int(n * 0.8)
train_ds = SeqDataset(X_cats[:idx], X_nums[:idx], y[:idx])
valid_ds = SeqDataset(X_cats[idx:], X_nums[idx:], y[idx:])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False, drop_last=False)

# 6d) Model / loss / optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_dim = X_nums.shape[-1]
model = GRUTakeoverModel(cats_info=cats_info, num_dim=num_dim, hidden_size=128, num_layers=1, dropout=0.1, window_size=WINDOW).to(device)

criterion = nn.BCEWithLogitsLoss()  # you can set pos_weight for class imbalance
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 6e) One quick training epoch (demo)
model.train()
for batch in train_loader:
    cats = batch['cats'].to(device)  # (B,T,C)
    nums = batch['nums'].to(device)  # (B,T,D)
    target = batch['y'].to(device)   # (B,)

    optimizer.zero_grad()
    logits = model(cats, nums)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()


# %%

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

from copy import deepcopy

# ==== Build windows ====
WINDOW = 20
X_cats, X_nums, y, end_ts = build_windows(df_prep, window_size=WINDOW)

# ==== Chronological split: 70/15/15 (train/val/test) ====
order = np.argsort(end_ts)
X_cats = X_cats[order]
X_nums = X_nums[order]
y      = y[order]

N = len(y)
n_train = int(N * 0.40)
n_val   = int(N * 0.15)
n_test  = N - n_train - n_val

train_ds = SeqDataset(X_cats[:n_train],                    X_nums[:n_train],                    y[:n_train])
valid_ds = SeqDataset(X_cats[n_train:n_train+n_val],       X_nums[n_train:n_train+n_val],       y[n_train:n_train+n_val])
test_ds  = SeqDataset(X_cats[n_train+n_val:],              X_nums[n_train+n_val:],              y[n_train+n_val:])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False)

# ==== Model / loss / optim ====
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_dim = X_nums.shape[-1]
model   = GRUTakeoverModel(
    cats_info=cats_info, num_dim=num_dim,
    hidden_size=128, num_layers=1, dropout=0.1, window_size=WINDOW
).to(device)

# Class imbalance: pos_weight = (#neg / #pos) on TRAIN ONLY
pos = float(y[:n_train].sum())
neg = float(n_train - pos)
if pos == 0:
    print("Warning: no positive samples in training; setting pos_weight=1.0")
    pos_weight = torch.tensor(1.0, device=device)
else:
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ==== Helpers ====
def evaluate_loss(model, loader):
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

# ==== Training loop (multi-epoch with best checkpoint on val loss) ====
EPOCHS = 5
best_val = float('inf')
best_state = None
history = []

for epoch in range(1, EPOCHS + 1):
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
    val_loss   = evaluate_loss(model, valid_loader)

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        best_state = deepcopy(model.state_dict())

# ==== Load best model and evaluate on TEST ====
if best_state is not None:
    model.load_state_dict(best_state)

test_loss = evaluate_loss(model, test_loader)
print(f"\nBest Val Loss: {best_val:.4f}")
print(f"Test  Loss   : {test_loss:.4f}")

# Optional: history as a quick table
try:
    import pandas as pd
    hist_df = pd.DataFrame(history)
    print("\nHistory (head):")
    print(hist_df.head())
except Exception:
    pass

#%%
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import numpy as np
import torch
model.eval()
def collect(loader):
    probs, ys = [], []
    with torch.no_grad():
        for b in loader:
            p = torch.sigmoid(model(b['cats'].to(device), b['nums'].to(device))).cpu().numpy()
            probs.append(p)
            ys.append(b['y'].numpy())
    return np.concatenate(probs), np.concatenate(ys)

val_p, val_y = collect(valid_loader)
test_p, test_y = collect(test_loader)

print("Val AUROC :", roc_auc_score(val_y, val_p))
print("Val AUPRC :", average_precision_score(val_y, val_p))
print("Val logloss (unweighted):", log_loss(val_y, val_p))

print("Test AUROC:", roc_auc_score(test_y, test_p))
print("Test AUPRC:", average_precision_score(test_y, test_p))
print("Test logloss (unweighted):", log_loss(test_y, test_p))

# %% 
# show number of positive cases in each set
print(y[:n_train].sum())
print(y[n_train:n_train+n_val].sum())
print(y[n_train+n_val:].sum())