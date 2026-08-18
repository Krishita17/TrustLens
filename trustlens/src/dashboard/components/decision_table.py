"""
Decision Table Component for TrustLens Dashboard.

Renders a professional color-coded table of authentication decisions
with filtering, sorting, and search capabilities.
"""

import streamlit as st
import pandas as pd


def render_decision_table(decisions: pd.DataFrame):
    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        filter_decision = st.selectbox(
            "Filter by decision:", ["All", "ALLOW", "DENY"], index=0
        )
    with col_search:
        search_user = st.text_input("Search by User ID:", placeholder="e.g. USER_042")

    filtered = decisions.copy()
    if filter_decision != "All":
        filtered = filtered[filtered["decision"] == filter_decision]
    if search_user:
        filtered = filtered[filtered["user_id"].str.contains(search_user, case=False, na=False)]

    display_cols = [
        "request_id", "timestamp", "user_id", "decision",
        "trust_score", "auth_method", "location_risk",
        "device_trust_score", "anomaly_score",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].copy()

    def color_decision(row):
        if row["decision"] == "ALLOW":
            return ["background-color: #1b5e20; color: #a5d6a7"] * len(row)
        elif row["decision"] == "DENY":
            return ["background-color: #b71c1c; color: #ef9a9a"] * len(row)
        return [""] * len(row)

    def format_float(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        return val

    styled = (
        display_df.style
        .apply(color_decision, axis=1)
        .format({
            "trust_score": "{:.4f}",
            "device_trust_score": "{:.3f}",
            "anomaly_score": "{:.3f}",
        }, na_rep="N/A")
    )
    st.dataframe(styled, use_container_width=True, height=450)

    col1, col2, col3 = st.columns(3)
    total = len(filtered)
    if total > 0:
        allow_pct = (filtered["decision"] == "ALLOW").mean() * 100
        deny_pct = 100 - allow_pct
        avg_trust = filtered["trust_score"].mean()
        col1.success(f"Allow rate: {allow_pct:.1f}%")
        col2.error(f"Deny rate: {deny_pct:.1f}%")
        col3.info(f"Avg trust: {avg_trust:.3f}")
    else:
        col1.warning("No matching records")
