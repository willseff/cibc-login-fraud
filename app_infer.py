
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, classification_report, average_precision_score, roc_auc_score

st.set_page_config(page_title="Login Fraud – Inference Dashboard", layout="wide")
st.title("Login Fraud Detector – Inference Dashboard")
st.caption("Loads precomputed artifacts; no heavy training or raw data required.")

st.sidebar.header("Artifacts")
artifacts_zip = st.file_uploader("Optionally upload a zipped artifacts folder (.zip)", type=["zip"])
art_dir = st.text_input("Or path to local artifacts folder", value="artifacts")

import os, zipfile, tempfile, shutil

def ensure_artifacts():
    if artifacts_zip is not None:
        tmp = tempfile.mkdtemp()
        zpath = os.path.join(tmp, "artifacts.zip")
        with open(zpath, "wb") as f:
            f.write(artifacts_zip.read())
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(tmp)
        # find a folder that contains eval_arrays.npz
        for root, dirs, files in os.walk(tmp):
            if "eval_arrays.npz" in files:
                return root
        st.error("Zip did not contain eval_arrays.npz.")
        return None
    else:
        return art_dir

art_path = ensure_artifacts()
if not art_path or not os.path.exists(os.path.join(art_path, "eval_arrays.npz")):
    st.warning("Provide artifacts (eval_arrays.npz, test_index.parquet, model_meta.json). See exporter script.")
    st.stop()

# Load lightweight arrays
arrays = np.load(os.path.join(art_path, "eval_arrays.npz"))
val_p = arrays["val_p"]; val_y = arrays["val_y"]
test_p = arrays["test_p"]; test_y = arrays["test_y"]

# Optional: user-level table
test_index_path = os.path.join(art_path, "test_index.parquet")
test_index = None
if os.path.exists(test_index_path):
    test_index = pd.read_parquet(test_index_path)

st.subheader("Overall Metrics")
left, right = st.columns(2)
with left:
    st.metric("Val AUPRC", f"{average_precision_score(val_y, val_p):.4f}")
    st.metric("Val AUROC", f"{roc_auc_score(val_y, val_p):.4f}")
with right:
    st.metric("Test AUPRC", f"{average_precision_score(test_y, test_p):.4f}")
    st.metric("Test AUROC", f"{roc_auc_score(test_y, test_p):.4f}")

st.subheader("Curves (Validation)")
prec, rec, _ = precision_recall_curve(val_y, val_p)
fig1 = plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (Val)")
st.pyplot(fig1, clear_figure=True)

fpr, tpr, _ = roc_curve(val_y, val_p)
fig2 = plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Val)")
st.pyplot(fig2, clear_figure=True)

st.subheader("Threshold Analysis (Test)")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
yhat = (test_p >= threshold).astype(int)
cm = confusion_matrix(test_y, yhat, labels=[0,1])

fig3 = plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.xticks([0,1], ["Pred 0", "Pred 1"]); plt.yticks([0,1], ["True 0", "True 1"])
plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
st.pyplot(fig3, clear_figure=True)

report = classification_report(test_y, yhat, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().style.format(precision=3))

st.markdown("---")
st.subheader("Risk Explorer (Top Accounts)")
if test_index is not None:
    # Show top-N by predicted probability
    N = st.number_input("Top N", min_value=5, max_value=1000, value=50, step=5)
    top = test_index.sort_values("test_p", ascending=False).head(int(N))
    st.dataframe(top.reset_index(drop=True))

    # Per-user view
    uniq_users = top["user_id"].astype(str).unique().tolist()
    sel_user = st.selectbox("Pick a user id", uniq_users)
    # If multiple windows per user exist in test_index, show time series
    user_rows = test_index[test_index["user_id"].astype(str) == sel_user].sort_values("end_ts")
    if len(user_rows):
        fig4 = plt.figure()
        plt.plot(user_rows["end_ts"], user_rows["test_p"], marker="o")
        plt.xlabel("End timestamp"); plt.ylabel("Fraud probability")
        plt.title(f"Score over time – user {sel_user}")
        st.pyplot(fig4, clear_figure=True)
else:
    st.info("Upload test_index.parquet to unlock per-user exploration.")

# --- Sidebar controls already exist above ---

st.sidebar.header("Methodology")

st.sidebar.markdown("""
**Data**
- Simulated login dataset (`rba-dataset.csv`)
- Positive = fraudulent logins (rare)
- Negative = legitimate logins
""")

st.sidebar.markdown("""
**Feature Engineering**
- Windowed sequences (last N logins per user)
- Categorical: user ID, job category → embeddings
- Numerical: login time, velocity, device/browser
- Scaled with `scaler.pkl`
""")

st.sidebar.markdown("""
**Model**
- GRU (Gated Recurrent Unit) for sequential data
- Embeddings + numeric features → GRU → dense layer → sigmoid
- Weighted BCE loss for imbalance
""")

st.sidebar.markdown("""
**Training / Eval**
- Chronological stratified split
- Downsampling for efficiency
- Metrics: AUPRC (primary), AUROC (secondary)
- Threshold tuning for ops trade-offs
""")
