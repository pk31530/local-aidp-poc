"""Phase 8: Streamlit dashboard — 5 views over the live platform state.

Run with:
    streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts only this file's own directory (src/dashboard/) on
# sys.path, not the project root — so `from src...` imports fail unless the
# root is added explicitly here. This must run before any `src.*` import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.dashboard import fraud_intel_tab
from src.dashboard.data import (
    get_active_model,
    get_confusion_matrix,
    get_executive_metrics,
    get_fraud_analysis,
    get_live_transactions,
    get_platform_health,
)

st.set_page_config(page_title="AiDP Fraud Detection", layout="wide")
st.title("Local AiDP POC — Fraud Detection Dashboard")

tab_exec, tab_live, tab_fraud, tab_model, tab_health, tab_fraud_intel = st.tabs(
    ["Executive Overview", "Live Transaction Feed", "Fraud Analysis", "Model Performance", "Platform Health", "Fraud Intelligence"]
)

# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------
with tab_exec:
    st.button("Refresh", key="refresh_exec")
    metrics = get_executive_metrics()
    model = get_active_model()
    health = get_platform_health()
    system_healthy = all(health.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions Processed", f"{metrics['transactions_processed']:,}")
    c2.metric("Transaction Value", f"₹{metrics['transaction_value']:,.2f}")
    c3.metric("Fraud Alerts", f"{metrics['fraud_alerts']:,}")
    c4.metric("Fraud Rate", f"{metrics['fraud_rate'] * 100:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Blocked Transactions", f"{metrics['blocked_transactions']:,}")
    c6.metric("Potential Fraud Value", f"₹{metrics['potential_fraud_value']:,.2f}")
    c7.metric("Active Model Version", model["model_version"] if model else "N/A")
    c8.metric("System Health", "🟢 Healthy" if system_healthy else "🔴 Degraded")

# ---------------------------------------------------------------------
# Live Transaction Feed
# ---------------------------------------------------------------------
with tab_live:
    st.button("Refresh", key="refresh_live")
    limit = st.slider("Rows to show", min_value=10, max_value=500, value=100, step=10)
    df = get_live_transactions(limit)
    if df.empty:
        st.info("No transactions scored yet. Run the API demo or the streaming producer/consumer.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Fraud Analysis
# ---------------------------------------------------------------------
with tab_fraud:
    st.button("Refresh", key="refresh_fraud")
    fraud_data = get_fraud_analysis()
    df = fraud_data["data"]

    if df.empty:
        st.info("No transactions scored yet.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud Score Distribution")
            bins = pd.cut(df["fraud_probability"], bins=10)
            dist = bins.value_counts().sort_index()
            dist.index = [f"{i.left:.1f}-{i.right:.1f}" for i in dist.index]
            st.bar_chart(dist)

            st.subheader("Fraud by Country")
            high_risk = df[df["decision"].isin(["REVIEW", "BLOCK"])]
            if not high_risk.empty:
                st.bar_chart(high_risk["country"].value_counts())
            else:
                st.caption("No REVIEW/BLOCK transactions yet.")

        with col2:
            st.subheader("Fraud by Merchant")
            if not high_risk.empty:
                st.bar_chart(high_risk["merchant"].value_counts())
            else:
                st.caption("No REVIEW/BLOCK transactions yet.")

            st.subheader("Fraud Over Time")
            ts = high_risk.copy()
            if not ts.empty:
                ts["hour"] = ts["transaction_timestamp"].dt.floor("h")
                st.line_chart(ts.groupby("hour").size())
            else:
                st.caption("No REVIEW/BLOCK transactions yet.")

        st.subheader("Amount vs. Fraud Score")
        st.scatter_chart(df, x="amount", y="fraud_probability", color="decision")

        st.subheader("Highest-Risk Transactions")
        top_risk = df.sort_values("fraud_probability", ascending=False).head(20)
        st.dataframe(
            top_risk[["transaction_id", "amount", "country", "merchant", "fraud_probability", "decision"]],
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------
with tab_model:
    st.button("Refresh", key="refresh_model")
    model = get_active_model()
    if model is None:
        st.warning("No active model registered yet. Run `python -m src.ml.train`.")
    else:
        st.caption(f"Active model: **{model['model_name']}** version **{model['model_version']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{model['precision_score']:.4f}" if model["precision_score"] is not None else "N/A")
        c2.metric("Recall", f"{model['recall_score']:.4f}" if model["recall_score"] is not None else "N/A")
        c3.metric("F1", f"{model['f1_score']:.4f}" if model["f1_score"] is not None else "N/A")
        c4.metric("ROC-AUC", f"{model['roc_auc_score']:.4f}" if model["roc_auc_score"] is not None else "N/A")

        cm = get_confusion_matrix()
        if cm:
            st.subheader("Confusion Matrix (held-out test set)")
            cm_df = pd.DataFrame(
                [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                index=["Actual: Not Fraud", "Actual: Fraud"],
                columns=["Predicted: Not Fraud", "Predicted: Fraud"],
            )
            st.table(cm_df)
        else:
            st.caption("No confusion matrix artifact found.")

# ---------------------------------------------------------------------
# Platform Health
# ---------------------------------------------------------------------
with tab_health:
    st.button("Refresh", key="refresh_health")
    health = get_platform_health()
    cols = st.columns(len(health))
    for col, (service, healthy) in zip(cols, health.items()):
        col.metric(service, "🟢 OK" if healthy else "🔴 FAIL")

# ---------------------------------------------------------------------
# Fraud Intelligence (v1.3, seven-channel) — read-only, aidp_test only
# ---------------------------------------------------------------------
with tab_fraud_intel:
    fraud_intel_tab.render()
