
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, classification_report, average_precision_score, roc_auc_score

st.set_page_config(page_title="Login Fraud – Inference Dashboard", layout="wide")

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Methodology"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Artifacts")
artifacts_zip = st.sidebar.file_uploader("Upload zipped artifacts (.zip)", type=["zip"])
art_dir = st.sidebar.text_input("Or local artifacts folder", value="artifacts")

# ---------------- Artifact Loading ----------------
import zipfile, tempfile

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

arrays = np.load(os.path.join(art_path, "eval_arrays.npz"))
val_p = arrays["val_p"]; val_y = arrays["val_y"]
test_p = arrays["test_p"]; test_y = arrays["test_y"]

test_index = None
test_index_path = os.path.join(art_path, "test_index.parquet")
if os.path.exists(test_index_path):
    test_index = pd.read_parquet(test_index_path)

# ---------------- Page: Dashboard ----------------
def render_dashboard():
    st.title("Login Fraud Detector – Inference Dashboard")
    st.caption("Loads precomputed artifacts; no heavy training or raw data required.")

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
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thresh_dash")
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
        N = st.number_input("Top N", min_value=5, max_value=1000, value=50, step=5, key="topn")
        top = test_index.sort_values("test_p", ascending=False).head(int(N))
        st.dataframe(top.reset_index(drop=True))

        uniq_users = top["user_id"].astype(str).unique().tolist()
        if len(uniq_users):
            sel_user = st.selectbox("Pick a user id", uniq_users, key="user_sel")
            user_rows = test_index[test_index["user_id"].astype(str) == sel_user].sort_values("end_ts")
            if len(user_rows):
                fig4 = plt.figure()
                plt.plot(user_rows["end_ts"], user_rows["test_p"], marker="o")
                plt.xlabel("End timestamp"); plt.ylabel("Fraud probability")
                plt.title(f"Score over time – user {sel_user}")
                st.pyplot(fig4, clear_figure=True)
    else:
        st.info("Upload test_index.parquet to unlock per-user exploration.")

# ---------------- Page: Methodology ----------------
def render_methodology():
    st.title("Methodology")
    st.caption("How the model and data pipeline were built.")

    st.markdown("### Data")
    st.markdown("""
    - **Source**: Login event dataset (`rba-dataset.csv`).
    - **Target**: Fraudulent vs. legitimate login windows.
    - **Class imbalance**: Highly skewed toward legitimate logins.
    """)

    st.markdown("### Feature Engineering")
    st.markdown("""
    - **Windowing**: Build fixed-length sequences per user (e.g., last *N* logins).
    - **Categorical features**: user_id, job category → learned embeddings.
    - **Numerical features**: time-based signals (hour, recency), velocity, device/browser signals → standardized via `scaler.pkl`.
    - **Labeling**: Window labeled positive if it ends in a fraudulent event.
    """)

    st.markdown("### Model")
    st.markdown("""
    - **Architecture**: Embeddings + scaled numerics → GRU → dense layer → sigmoid.
    - **Loss**: Weighted BCE to address class imbalance.
    - **Window size**: Typically 20 (configurable).
    """)

    st.markdown("### Training & Evaluation")
    st.markdown("""
    - **Split**: Chronological, stratified to preserve positives across train/val/test (prevents leakage).
    - **Efficiency**: Optional downsampling of negatives for faster training.
    - **Metrics**: **AUPRC** (primary) and AUROC (secondary).
    - **Threshold tuning**: Emphasis on precision/recall trade-offs for fraud ops.
    """)

    with st.expander("Next steps & deployment ideas"):
        st.markdown("""
        - Add IP reputation, device graph features, and login velocity aggregations.
        - Stack with a metadata model (e.g., CatBoost on `p_hat` + static features).
        - Serve via FastAPI; this Streamlit app can call the API.
        - Monitor drift and alert on threshold breaches.
        """)

    # Optional diagrams (placeholders)
    st.markdown("### Diagrams")
    st.info("Add `figures/` with PNGs for pipeline or architecture; then call `st.image(...)`.")

# ---------------- Router ----------------
if page == "Dashboard":
    render_dashboard()
else:
    render_methodology()
