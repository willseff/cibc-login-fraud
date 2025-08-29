from pipeline import read_data, downsample, engineer_features
from gru import *
import numpy as np

# %%
from pipeline import read_data, downsample, engineer_features

# Step 1: Read the data
file_path = "rba-dataset.csv"
df = read_data(file_path)

# Step 2: Downsample (e.g., keep 10x negatives for each positive)
downsampled_df = downsample(df, factor=10)

# Step 3: Feature engineering
df = engineer_features(downsampled_df)

# %%
import gru
import importlib

importlib.reload(gru)
df_prep, cats_info, scaler = gru.prepare_dataframe(df)

WINDOW = 20
X_cats, X_nums, y, end_ts = gru.build_windows(df_prep, window_size=WINDOW)

# %%
import numpy as np
importlib.reload(gru)
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from copy import deepcopy

# ==== Chronological split: 70/15/15 (train/val/test) ====
order = np.argsort(end_ts)
X_cats = X_cats[order]
X_nums = X_nums[order]
y      = y[order]

N = len(y)
n_train = int(N * 0.50)
n_val   = int(N * 0.15)
n_test  = N - n_train - n_val

train_ds = gru.SeqDataset(X_cats[:n_train],                    X_nums[:n_train],                    y[:n_train])
valid_ds = gru.SeqDataset(X_cats[n_train:n_train+n_val],       X_nums[n_train:n_train+n_val],       y[n_train:n_train+n_val])
test_ds  = gru.SeqDataset(X_cats[n_train+n_val:],              X_nums[n_train+n_val:],              y[n_train+n_val:])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False)

# ==== Model / loss / optim ====
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_dim = X_nums.shape[-1]
model   = gru.GRUTakeoverModel(
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


# %%

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

# %%

from pipeline import read_data, downsample, engineer_features
import gru
import importlib
importlib.reload(gru)

# Step 1: Read and preprocess data
file_path = "rba-dataset.csv"
df = read_data(file_path)
downsampled_df = downsample(df, factor=200)
df = engineer_features(downsampled_df)

# Step 2: Prepare for GRU model
df_prep, cats_info, scaler = gru.prepare_dataframe(df)
WINDOW = 20
X_cats, X_nums, y, end_ts = gru.build_windows(df_prep, window_size=WINDOW)

# Step 3: Chronological split
X_cats, X_nums, y, idxs = gru.stratified_chronological_split(X_cats, X_nums, y, end_ts, train_pos_ratio=0.6, val_pos_ratio=0.2, test_pos_ratio=0.2)

# Step 4: Create datasets and loaders
train_ds, valid_ds, test_ds = gru.create_datasets(X_cats, X_nums, y, idxs)
train_loader, valid_loader, test_loader = gru.create_loaders(train_ds, valid_ds, test_ds)

# Step 5: Model, optimizer, loss
device = gru.get_device()
num_dim = X_nums.shape[-1]
model = gru.create_model(cats_info, num_dim, WINDOW, device)
pos_weight = gru.get_pos_weight(y[idxs['train'][0]:idxs['train'][1]], device)
criterion = gru.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = gru.get_optimizer(model)

# Step 6: Train model
model, history, best_val = gru.train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=5)

# Step 7: Evaluate
test_loss = gru.evaluate_loss(model, test_loader, criterion, device)
print(f"Best Val Loss: {best_val:.4f}")
print(f"Test  Loss   : {test_loss:.4f}")

# Step 8: Metrics
val_p, val_y = gru.collect_probs(model, valid_loader, device)
test_p, test_y = gru.collect_probs(model, test_loader, device)
gru.print_metrics(val_y, val_p, test_y, test_p)

