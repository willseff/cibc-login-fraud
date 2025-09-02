
"""
Export artifacts for a lightweight Streamlit dashboard.

Run this AFTER you've trained your GRU the way you like (on your big machine / offline).
It saves:
- artifacts/model_state.pt
- artifacts/model_meta.json  (window size, cats_info, etc.)
- artifacts/scaler.pkl
- artifacts/eval_arrays.npz  (val_y, val_p, test_y, test_p)
- artifacts/test_index.parquet (user_id, end_ts, test_p, test_y)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import torch

import gru
from pipeline import read_data, downsample, engineer_features

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def to_jsonable(obj):
    # cats_info might include numpy types; convert safely
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

def train_and_collect(window_size=20, epochs=5, downsample_factor=200,
                      train_pos_ratio=0.6, val_pos_ratio=0.2, test_pos_ratio=0.2,
                      csv_path="rba-dataset.csv"):
    # Prepare data using your pipeline
    df_pl = read_data(csv_path)
    df_pl = downsample(df_pl, factor=int(downsample_factor))
    df_pd = engineer_features(df_pl)

    # Sequences
    df_prep, cats_info, scaler = gru.prepare_dataframe(df_pd)
    X_cats, X_nums, y, end_ts, user_ids = gru.build_windows(df_prep, window_size=window_size)

    # Chronological stratified split
    X_cats, X_nums, y, idxs = gru.stratified_chronological_split(
        X_cats, X_nums, y, end_ts,
        train_pos_ratio=train_pos_ratio,
        val_pos_ratio=val_pos_ratio,
        test_pos_ratio=test_pos_ratio
    )

    # Datasets & loaders
    train_ds, valid_ds, test_ds = gru.create_datasets(X_cats, X_nums, y, idxs)
    train_loader, valid_loader, test_loader = gru.create_loaders(train_ds, valid_ds, test_ds)

    # Model
    device = gru.get_device()
    num_dim = X_nums.shape[-1]
    model = gru.create_model(cats_info, num_dim, window_size, device)

    pos_weight = gru.get_pos_weight(y[idxs['train'][0]:idxs['train'][1]], device)
    criterion = gru.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = gru.get_optimizer(model)

    # Train
    model, history, best_val = gru.train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=int(epochs))

    # Collect probs
    val_p, val_y = gru.collect_probs(model, valid_loader, device)
    test_p, test_y = gru.collect_probs(model, test_loader, device)

    # Build a test index table for per-user exploration
    t0, t1 = idxs['test']
    test_index = pd.DataFrame({
        "user_id": np.array(user_ids[t0:t1]),
        "end_ts": np.array(end_ts[t0:t1]),
        "test_p": test_p,
        "test_y": test_y.astype(int),
    })

    return {
        "model": model,
        "cats_info": cats_info,
        "scaler": scaler,
        "window_size": window_size,
        "val_p": val_p, "val_y": val_y,
        "test_p": test_p, "test_y": test_y,
        "test_index": test_index,
    }

def export_artifacts(state):
    # model
    torch.save(state["model"].state_dict(), os.path.join(ART_DIR, "model_state.pt"))

    # meta
    meta = {
        "window_size": state["window_size"],
        "cats_info": to_jsonable(state["cats_info"]),
    }
    with open(os.path.join(ART_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # scaler
    joblib.dump(state["scaler"], os.path.join(ART_DIR, "scaler.pkl"))

    # arrays
    np.savez_compressed(
        os.path.join(ART_DIR, "eval_arrays.npz"),
        val_p=state["val_p"],
        val_y=state["val_y"],
        test_p=state["test_p"],
        test_y=state["test_y"],
    )

    # index
    state["test_index"].to_parquet(os.path.join(ART_DIR, "test_index.parquet"), index=False)

if __name__ == "__main__":
    # Adjust params as you like; run offline on your beefy box
    s = train_and_collect(window_size=20, epochs=5, downsample_factor=200)
    export_artifacts(s)
    print("Artifacts written to ./artifacts")
