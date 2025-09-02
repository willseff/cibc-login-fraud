# %%
from pipeline import read_data, downsample, engineer_features
import gru
import importlib
importlib.reload(gru)
from data_generation import generate_user_table

# Step 1: Read and preprocess data
file_path = "rba-dataset.csv"
df = read_data(file_path)
downsampled_df = downsample(df, factor=200)
df = engineer_features(downsampled_df)

# %%
importlib.reload(gru)
# Step 2: Prepare for GRU model
df_prep, cats_info, scaler = gru.prepare_dataframe(df)
WINDOW = 20
X_cats, X_nums, y, end_ts, user_ids = gru.build_windows(df_prep, window_size=WINDOW)

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

# %%
#Step 9: Generate user table
import data_generation
importlib.reload(data_generation)
from data_generation import generate_user_table

user_table = generate_user_table(downsampled_df.to_pandas())

# %%

# === Split the original train+val region into two halves, train on the first, predict on the second ===
import numpy as np
import pandas as pd

# 1) Slice out the original train+val span (chronological)
train_start, train_end = idxs['train']
val_start,   val_end   = idxs['val']
tv_start, tv_end = train_start, val_end   # inclusive train..val block from the first split

X_cats_trainval = X_cats[tv_start:tv_end]   
X_nums_trainval = X_nums[tv_start:tv_end]
y_trainval      = y[tv_start:tv_end]
end_ts_trainval = end_ts[tv_start:tv_end]

# 2) Within that block, split 50/50 by positives (chronologically) and set no "test" portion
X_cats_tv2, X_nums_tv2, y_tv2, idxs_tv2 = gru.stratified_chronological_split(
    X_cats_trainval, X_nums_trainval, y_trainval, end_ts_trainval,
    train_pos_ratio=0.5, val_pos_ratio=0.5, test_pos_ratio=0.0
)

# 3) Datasets & loaders for the inner split
train_ds2, val_ds2, _ = gru.create_datasets(X_cats_tv2, X_nums_tv2, y_tv2, idxs_tv2)
train_loader2, val_loader2, _ = gru.create_loaders(train_ds2, val_ds2, None)

# 4) Fresh model and weighted loss computed only from the inner-train slice (prevents leakage)
device   = gru.get_device()
num_dim  = X_nums.shape[-1]   # same numeric feature width as before
WINDOW   = X_cats.shape[1]    # or keep your constant, e.g., WINDOW = 20 used earlier
model2   = gru.create_model(cats_info, num_dim, WINDOW, device)

inner_train_y = y_tv2[idxs_tv2['train'][0]:idxs_tv2['train'][1]]
pos_weight    = gru.get_pos_weight(inner_train_y, device)

criterion2 = gru.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer2 = gru.get_optimizer(model2)

# 5) Train and then score the inner-validation half
model2, hist2, best_val2 = gru.train_model(model2, train_loader2, val_loader2, criterion2, optimizer2, device, epochs=5)
val_p2, val_y2 = gru.collect_probs(model2, val_loader2, device)

# 6) Map inner-validation rows back to absolute indices and to User IDs for merging later
inner_val_start, inner_val_end = idxs_tv2['val']
# indices relative to the train+val block:
rel_idx_tv2 = np.arange(0, len(y_trainval))
rel_val_idx = rel_idx_tv2[inner_val_start:inner_val_end]
# absolute indices w.r.t. the full (X_cats, X_nums, y)
abs_val_idx = (tv_start + rel_val_idx)

# grab the corresponding User IDs from the array returned by build_windows
val_user_ids = user_ids[abs_val_idx]

# 7) Final DataFrame of OOF-style predictions for this val half
oof_val = pd.DataFrame({
    "abs_index": abs_val_idx,
    "User ID":   val_user_ids,
    "y_true":    val_y2.astype(int),
    "p_hat":     val_p2
}).reset_index(drop=True)

print(oof_val.head())
print(f"\nInner-split validation size: {len(oof_val)}")

# %%
# merge with user_table
oof_val = oof_val.merge(user_table, on="User ID", how="left")

oof_val

# %%
# drop abs_index, y_true, Any_Account_Takeover
oof_val = oof_val.drop(columns=["abs_index", "Is Account Takeover", "Any Account Takeover"], errors="ignore")


# %% oof_val
#oof_val = oof_val.drop(columns=["User ID"], errors="ignore")



# %%  
# train logistic regression model using only p_hat and Age from oof_val
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Prepare features and target
X = oof_val[["p_hat", "Age"]].values
y = oof_val["y_true"].values if "y_true" in oof_val.columns else None

# Remove rows with missing values
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
if y is not None:
    y = y[mask]

# Train logistic regression
if y is not None:
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X, y)
    preds = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    acc = accuracy_score(y, preds > 0.5)
    print(f"Logistic Regression AUC: {auc:.4f}")
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    # Optionally, print coefficients
    print(f"Coefficients: {lr.coef_}")
    from sklearn.metrics import average_precision_score
    auprc = average_precision_score(y, preds)
    print(f"Logistic Regression AUPRC: {auprc:.4f}")
else:
    print("y_true column not found in oof_val. Cannot train logistic regression.")

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score



# --- Use the SAME arrays/datasets that produced test_p/test_y ---
# We already have: test_p, test_y from collect_probs(...)

# Get the test user IDs from the ORIGINAL arrays & idxs used to build test_loader
test_start, test_end = idxs['test']
test_user_ids = user_ids[test_start:test_end]  # from the original build_windows

test_df = pd.DataFrame({
    "User ID": test_user_ids,
    "y_true":  test_y.astype(int),   # <-- use the labels collected with test_p
    "p_hat":   test_p
})

# Merge static features
test_df = test_df.merge(user_table, on="User ID", how="left")

# Coerce numeric Age and drop missing rows (same preprocessing as training)
test_df["Age"] = pd.to_numeric(test_df["Age"], errors="coerce")
test_df = test_df.dropna(subset=["p_hat", "Age", "y_true"])

X_test = test_df[["p_hat", "Age"]].to_numpy()
y_test = test_df["y_true"].to_numpy()

# Reuse the trained logistic regression 'lr'
test_preds = lr.predict_proba(X_test)[:, 1]

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
print(f"Test AUPRC: {average_precision_score(y_test, test_preds):.4f}")
print(f"Test AUC:   {roc_auc_score(y_test, test_preds):.4f}")
print(f"Test Acc:   {accuracy_score(y_test, test_preds > 0.5):.4f}")

# %%
# compute AUPRC with just the p_hat and the y_true


auprc = average_precision_score(test_y, test_p)
print(f"AUPRC: {auprc:.4f}")
# %%

print("LR coef on [p_hat, Age]:", lr.coef_)
