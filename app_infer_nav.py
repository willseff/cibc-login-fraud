
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, classification_report, average_precision_score, roc_auc_score
import matplotlib.dates as mdates

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
if "end_ts" in test_index.columns:
    test_index["end_ts"] = pd.to_datetime(test_index["end_ts"], unit="ms")

# ---------------- Page: Dashboard ----------------
def render_dashboard():
    st.title("Login Fraud Detector – Inference Dashboard")
    st.caption("Technical view: PR-first, compact ROC, live threshold trade-offs.")

    # ---------- Precompute ----------
    prevalence = float(np.mean(test_y))
    auprc_val = average_precision_score(val_y, val_p)
    auprc_test = average_precision_score(test_y, test_p)
    auroc_val = roc_auc_score(val_y, val_p)
    auroc_test = roc_auc_score(test_y, test_p)
    lift_vs_baseline = auprc_test / max(prevalence, 1e-12)

    # ---------- Top metrics (PR emphasized) ----------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUPRC (Validation)", f"{auprc_val:.4f}")
    m2.metric("AUPRC (Test)", f"{auprc_test:.4f}")
    m3.metric("Prevalence (Test)", f"{prevalence:.6f}")
    m4.metric("Lift vs Baseline (Test)", f"{lift_vs_baseline:.1f}×")
    # AUROC present but de-emphasized
    with m5:
        st.caption("AUROC (smaller relevance, imbalanced)")
        st.write(f"Val: **{auroc_val:.3f}**  \nTest: **{auroc_test:.3f}**")

    st.divider()

    # ---------- Curves (PR big, ROC small/optional) ----------
    st.subheader("Curves")

    c1, c2, c3 = st.columns(3)

    # --- PR (Validation) ---
    with c1:
        prec_v, rec_v, _ = precision_recall_curve(val_y, val_p)
        fig_v, ax_v = plt.subplots(figsize=(3.8, 3.0), dpi=130)
        ax_v.plot(rec_v, prec_v)
        ax_v.set_xlabel("Recall"); ax_v.set_ylabel("Precision")
        ax_v.set_title(f"PR (Val)\nAUPRC={auprc_val:.3f}")
        ax_v.grid(True, alpha=0.25)
        st.pyplot(fig_v, clear_figure=True, use_container_width=True)


    # --- PR (Test) ---
    with c2:
        prec_t, rec_t, _ = precision_recall_curve(test_y, test_p)
        fig_t, ax_t = plt.subplots(figsize=(3.8, 3.0), dpi=130)
        ax_t.plot(rec_t, prec_t)
        ax_t.set_xlabel("Recall"); ax_t.set_ylabel("Precision")
        ax_t.set_title(f"PR (Test)\nAUPRC={auprc_test:.3f}")
        ax_t.grid(True, alpha=0.25)
        st.pyplot(fig_t, clear_figure=True, use_container_width=True)

    # --- ROC (Validation, compact) ---
    with c3:
        with st.expander("ROC (Val) — less useful, imbalanced data", expanded=False):
            fpr, tpr, _ = roc_curve(val_y, val_p)
            fig_r, ax_r = plt.subplots(figsize=(3.8, 3.0), dpi=130)
            ax_r.plot(fpr, tpr)
            ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
            ax_r.set_title(f"ROC (Val)\nAUROC={auroc_val:.3f}")
            ax_r.grid(True, alpha=0.25)
            st.pyplot(fig_r, clear_figure=True)

    st.divider()

    # ---------- Threshold analysis (Test) ----------
    st.subheader("Threshold trade-offs (Test)", anchor=False)
    # persist threshold across reruns/tabs if you add more pages later
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.50
    st.session_state.threshold = st.slider(
        "Decision threshold",
        0.0, 1.0, float(st.session_state.threshold), 0.01, key="thresh_tradeoffs"
    )
    thr = float(st.session_state.threshold)
    yhat = (test_p >= thr).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score
    prec_t = precision_score(test_y, yhat, zero_division=0)
    rec_t  = recall_score(test_y, yhat, zero_division=0)
    f1_t   = f1_score(test_y, yhat, zero_division=0)
    fpr_t  = ( (yhat==1) & (test_y==0) ).sum() / max((test_y==0).sum(), 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{prec_t:.3f}")
    k2.metric("Recall", f"{rec_t:.3f}")
    k3.metric("F1", f"{f1_t:.3f}")
    k4.metric("FPR", f"{fpr_t:.4f}")

    # compact confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(test_y, yhat, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(1.5, 1.5), dpi=140)
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["True 0", "True 1"])
    ax_cm.set_title(f"Confusion Matrix (thr={thr:.2f})", fontsize=9)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    st.pyplot(fig_cm, use_container_width=False)

    with st.expander("Classification report (test)"):
        report = classification_report(test_y, yhat, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().style.format(precision=3), use_container_width=True)

    st.divider()

    # ---------- Risk Explorer (unchanged but compact) ----------
    st.subheader("Risk Explorer (Top accounts)", anchor=False)
    if test_index is not None:
        N = st.number_input("Top N", min_value=5, max_value=1000, value=50, step=5, key="topn")
        top = test_index.sort_values("test_p", ascending=False).head(int(N))
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

        uniq_users = top["user_id"].astype(str).unique().tolist()
        if len(uniq_users):
            sel_user = st.selectbox("Pick a user id", uniq_users, key="user_sel")
            user_rows = test_index[test_index["user_id"].astype(str) == sel_user].sort_values("end_ts")
            if len(user_rows):
              fig4, ax4 = plt.subplots(figsize=(8, 2))
              ax4.plot(user_rows["end_ts"], user_rows["test_p"], marker="o")
              ax4.set_xlabel("End timestamp")
              ax4.set_ylabel("Fraud probability")
              ax4.set_title(f"Score over time – user {sel_user}", fontsize=10)

              # format x-axis
              ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
              fig4.autofmt_xdate()  # rotate labels automatically

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
    - **Size**: Extremely imbalanced with only **141 fraudulent events** out of over **33 million login records**.  
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
              
    Note: `index`, `Login Timestamp`, and `IP Address` were not used as features for prediction.
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
    - **Resulting shape**: 100k samples with 20 columns each (after windowing).
    """)

    st.markdown("### Model")
    st.markdown("""
    - **Architecture**: Embeddings + scaled numerics → GRU → dense layer → sigmoid.
    - **Loss**: Weighted BCE to address class imbalance.
    - **Window size**: Last 20 logins.

    I used a GRU network that combines categorical embeddings (for features like device and country) with normalized numeric signals (e.g., login timing, failed attempts). The embeddings let the model capture high-cardinality features without one-hot explosion, while the GRU processes sequences of the last 20 logins to learn temporal patterns in user behavior. A small feedforward head on the final hidden state outputs the fraud probability.
    """)

    st.subheader("Architecture (GRU + Embeddings)")

    dot = r'''
    digraph G {
      rankdir=LR;
      node [shape=rect, style="rounded,filled", fillcolor="#F7F9FB", color="#CBD5E1", fontname="Helvetica", fontsize=11];

      subgraph cluster_emb {
        label="Categorical Embeddings"; labelloc="t"; fontsize=12; fontname="Helvetica-Bold";
        style="rounded,dashed"; color="#94A3B8";
        u   [label="engineered features\nEmbedding)"];
        dev [label="device / browser\nEmbedding"];
        geo [label="country / region\nEmbedding"];
      }

      nums   [label="Numeric features\n(Standardized: hour, Δt, fails, etc.)"];
      concat [label="Concatenate\n[embeddings ⊕ numerics]"];
      gru    [label="GRU\nT=20, H=128, L=1"];
      head   [label="Dense head\nLinear → ReLU → Dropout → Linear"];
      sig    [label="Sigmoid\np(fraud)"];

      {u dev geo} -> concat;
      nums -> concat;
      concat -> gru -> head -> sig;
    }
    '''
    st.graphviz_chart(dot)

    st.caption("Windowing (per user)")
    dot2 = r'''
    digraph W {
      rankdir=LR;
      node [shape=rect, style="rounded,filled", fillcolor="#F7F9FB", color="#CBD5E1", fontname="Helvetica", fontsize=10];

      subgraph cluster_seq {
        label="User sequence (last 20 logins)"; labelloc="t"; fontsize=11; fontname="Helvetica-Bold";
        style="rounded,dashed"; color="#94A3B8";
        t1 [label="t-19"]; t2 [label="t-18"]; t3 [label="..."]; t20 [label="t"];
      }
      t1 -> t2 -> t3 -> t20 [color="#94A3B8"];
      t20o [label="final hidden state\n(used by head)"];
      t20 -> t20o;
    }
    '''
    st.graphviz_chart(dot2)


    st.markdown("### Training & Evaluation")
    st.markdown("""
    - **Split**: Chronological, stratified to preserve positives across train/val/test (prevents leakage).
    - **Efficiency**: Downsampling of negatives for faster training.
    - **Metrics**: **AUPRC** (primary) and AUROC (secondary).
    - **Threshold tuning**: Emphasis on precision/recall trade-offs for fraud ops.
    """)

    st.markdown("### Next steps & deployment ideas:")
    st.markdown("""
    - Stack with a metadata model (e.g., CatBoost on `p_hat` + static features).
    - Serve via FastAPI; this Streamlit app can call the API.
    """)

# ---------------- Router ----------------
if page == "Dashboard":
    render_dashboard()
else:
    render_methodology()
