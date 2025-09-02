
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

# ---------------- Artifact Loading ----------------

art_path = "artifacts"   # <--- change to your absolute or relative path

if not os.path.exists(os.path.join(art_path, "eval_arrays.npz")):
    st.error(f"Artifacts not found in {art_path}. Make sure eval_arrays.npz is there.")
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
    col1, col2 = st.columns(2)

    with col1:
        prec, rec, _ = precision_recall_curve(val_y, val_p)
        fig1, ax1 = plt.subplots(figsize=(4,3))  # smaller figure
        ax1.plot(rec, prec)
        ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("PR (Val)")
        st.pyplot(fig1)

    with col2:
        fpr, tpr, _ = roc_curve(val_y, val_p)
        fig2, ax2 = plt.subplots(figsize=(4,3))  # smaller figure
        ax2.plot(fpr, tpr)
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC (Val)")
        st.pyplot(fig2)

    st.subheader("Threshold Analysis (Test)")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thresh_dash")
    yhat = (test_p >= threshold).astype(int)
    cm = confusion_matrix(test_y, yhat, labels=[0,1])

    fig3, ax = plt.subplots(figsize=(2,2), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})", fontsize=10)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

    st.pyplot(fig3, use_container_width=False)

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
                fig4, ax4 = plt.subplots(figsize=(8,2))   # 👈 smaller height (2 inches)
                ax4.plot(user_rows["end_ts"], user_rows["test_p"], marker="o")
                ax4.set_xlabel("End timestamp")
                ax4.set_ylabel("Fraud probability")
                ax4.set_title(f"Score over time – user {sel_user}", fontsize=10)
                st.pyplot(fig4, clear_figure=True, use_container_width=True)
    else:
        st.info("Upload test_index.parquet to unlock per-user exploration.")

# ---------------- Page: Methodology ----------------
def render_methodology():
    st.title("Methodology")
    st.caption("How the model and data pipeline were built.")

    st.markdown("### Data")
    st.markdown("""
    - **Source**: [Kaggle – RBA Dataset](https://www.kaggle.com/datasets/dasgroup/rba-dataset/data)  
    - **Size**: Extremely imbalanced with only **141 fraudulent events** out of over **7 million login records**.  
    - **Target variable**: `Is Account Takeover` (binary indicator for fraudulent logins).  
    """)

    st.markdown("### Data Sampling")
    st.markdown("""
    Because the dataset was extremely imbalanced, the raw data was **downsampled**.  
    - All customers with at least one **positive case** (account takeover) were included.  
    - For each such customer, **200 other customers** without positives were randomly sampled.  
    This produced a more balanced working dataset for modeling while still preserving all rare fraudulent events.
    """)

    st.markdown("### Features Used")
    st.markdown("""
    Only features related to **IP address, device, and login times** were used.  
    Full feature list in the dataset:
    - `index`
    - `Login Timestamp`
    - `IP Address`
    - `Country`
    - `Region`
    - `City`
    - `Browser Name and Version`
    - `OS Name and Version`
    - `Device Type`
    - `Login Successful`
    - `Is Attack IP`
    - `Is Account Takeover` *(target)*
    """)

    st.markdown("### Engineered Features")
    st.markdown("""
    Several behavioral features were created to capture anomalies across login sessions:

    - **Login Hour & Login Day of Week**  
      Extracted from the timestamp to capture time-based usage patterns.  
      Example: a user typically logs in during the day, but sudden logins at 3 AM may be suspicious.

    - **Time Since Last Login**  
      Difference in seconds between the current and previous login for the same user.  
      Captures abnormal activity frequency (e.g., dozens of logins in a short span).

    - **Is First Login**  
      Boolean flag indicating whether this is the first recorded login for the user.  
      Ensures features like IP/Browser change are interpreted correctly.

    - **Has IP Changed**  
      Flag for whether the IP address differs from the previous login for the same user.  
      Frequent changes may indicate use of proxies or suspicious access.

    - **Country Changed**  
      Flag for whether the login country differs from the previous session.  
      Helps detect improbable travel or VPN-based takeovers.

    - **Success Group**  
      Running count of successful logins, used to segment activity windows between consecutive successes.

    - **Failed Logins Since Last Success**  
      Number of failed attempts since the most recent successful login.  
      High values may indicate brute-force or credential-stuffing attacks.

    - **Browser Changed**  
      Flag for whether the browser (name + version) differs from the prior login.  
      Captures unusual device or environment changes.
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
    - **Window size**: Last 20 logins.
    """)

    st.markdown("### Training & Evaluation")
    st.markdown("""
    - **Split**: Chronological, stratified to preserve positives across train/val/test (prevents leakage).
    - **Efficiency**: Optional downsampling of negatives for faster training.
    - **Metrics**: **AUPRC** (primary) and AUROC (secondary).
    - **Threshold tuning**: Emphasis on precision/recall trade-offs for fraud ops.
    """)

    st.markdown("###Next steps & deployment ideas:")
    st.markdown("""
    - Stack with a metadata model (e.g., CatBoost on `p_hat` + static features).
    - Serve via FastAPI; this Streamlit app can call the API.
    - Monitor drift and alert on threshold breaches.
    """)



# ---------------- Router ----------------
if page == "Dashboard":
    render_dashboard()
else:
    render_methodology()
