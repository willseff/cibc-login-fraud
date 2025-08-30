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
from data_generation import generate_user_table

user_table = generate_user_table(downsampled_df.to_pandas())

# %%
import numpy as np

train_start, train_end = idxs['train']
val_start, val_end = idxs['val']
test_start, test_end = idxs['test']

# Merge train+val
X_cats_trainval = X_cats[train_start:val_end]
X_nums_trainval = X_nums[train_start:val_end]
y_trainval      = y[train_start:val_end]

X_cats_test, X_nums_test, y_test = X_cats[test_start:test_end], X_nums[test_start:test_end], y[test_start:test_end]


# %%
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)  # no shuffle, keeps order
oof_preds = np.zeros(len(y_trainval))
device = gru.get_device()
num_dim = X_nums.shape[-1]

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cats_trainval)):
    print(f"=== Fold {fold+1} ===")
    # Build datasets
    train_ds = gru.SeqDataset(X_cats_trainval[tr_idx], X_nums_trainval[tr_idx], y_trainval[tr_idx])
    val_ds   = gru.SeqDataset(X_cats_trainval[val_idx], X_nums_trainval[val_idx], y_trainval[val_idx])
    train_loader, val_loader, _ = gru.create_loaders(train_ds, val_ds, val_ds)

    # Fresh model
    oof_model = gru.create_model(cats_info, num_dim, WINDOW, device)
    pos_weight = gru.get_pos_weight(y_trainval[tr_idx], device)
    criterion = gru.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = gru.get_optimizer(oof_model)

    # Train
    oof_model, history, best_val = gru.train_model(oof_model, train_loader, val_loader, criterion, optimizer, device, epochs=5)

    # Collect predictions on this fold’s validation set
    val_p, _ = gru.collect_probs(oof_model, val_loader, device)
    oof_preds[val_idx] = val_p

# %%
import pandas as pd

# combine the oof_preds with the User IDs
user_ids = np.concatenate([df['User ID'].values[train_start:val_end]])
oof_df = pd.DataFrame({'User ID': user_ids, 'prediction': oof_preds})

# %%
# merge with the user_table
oof_df = oof_df.merge(user_table, on='User ID', how='left')

# drop the Any Account Takeover column
oof_df = oof_df.drop(columns=['Any Account Takeover', 'Is Account Takeover'])

# add y_trainval as the target
oof_df['target'] = y_trainval

# %%
oof_df.target.sum()

# %%  show rows where Is Account Takeover is True
np.sum(y_trainval)
# %%

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Features and target
X = oof_df.drop(columns=["target"])  
y = oof_df["target"].astype(int)  # convert True/False to 1/0

# Specify categorical columns (CatBoost handles them natively)
cat_features = ["Job Category", 'User ID']

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Define Pools (CatBoost requires this for categorical features)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# Initialize CatBoost model
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    od_type=None,
    eval_metric="PRAUC",
    loss_function="Logloss",
    verbose=100,
    random_seed=50,
    auto_class_weights="Balanced"  # adjust for imbalance (1:negative, 10:positive as example)
)

# Train model
cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=500)

# Evaluate
y_pred_proba = cat_model.predict_proba(X_val)[:, 1]
y_pred = cat_model.predict(X_val)

print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))
print("PR AUC:", average_precision_score(y_val, y_pred_proba))
print(classification_report(y_val, y_pred))

# %%

oof_df

# %%
# === Prepare test set features ===
import numpy as np

# Collect GRU probabilities for test set
test_p, test_y = gru.collect_probs(model, test_loader, device)

# Combine with User IDs
test_user_ids = df['User ID'].values[test_start:test_end]
test_df = pd.DataFrame({'User ID': test_user_ids, 'prediction': test_p})

# Merge with static user_table
test_df = test_df.merge(user_table, on='User ID', how='left')

# Add target
test_df['target'] = test_y

# === Run CatBoost on test set ===
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"].astype(int)

test_pool = Pool(X_test, y_test, cat_features=["Job Category", "User ID"])

# Predict
test_pred_proba = cat_model.predict_proba(test_pool)[:, 1]
test_pred = cat_model.predict(test_pool)

# Metrics
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print("Test ROC AUC:", roc_auc_score(y_test, test_pred_proba))
print("Test PR AUC:", average_precision_score(y_test, test_pred_proba))
print(classification_report(y_test, test_pred))


# %% variable importance 
import pandas as pd

# If you trained with Pool(X, y, cat_features=...), reuse that same Pool
# and pass the *exact* feature order used in training.
feature_names = list(X_train.columns)  # or X_trv.columns if you used train+val

imp = cat_model.get_feature_importance(train_pool, type="PredictionValuesChange")
imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

print(imp_df.head(20))
