"""
Counterfactual Analysis Component.

Shows what changes would flip an authentication decision from DENY to ALLOW.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_counterfactual_panel(row, feature_names, feature_values):
    if row["decision"] == "ALLOW":
        st.success(
            "This request was **ALLOWED**. No counterfactual needed — "
            "all trust conditions were satisfied."
        )
        _show_margin_analysis(row)
        return

    st.warning("This request was **DENIED**. Here's what would change the decision:")

    suggestions = _compute_counterfactuals(row, feature_names, feature_values)

    for i, suggestion in enumerate(suggestions):
        improvement = suggestion["improvement"]
        color = "#00e676" if improvement > 0.1 else "#ffc107"
        icon = {
            "auth_method": "🔑",
            "device_trust_score": "🛡️",
            "failed_attempts": "🔄",
            "vpn_active": "🌐",
            "patch_level": "📦",
            "location_risk": "📍",
            "anomaly_score": "🔍",
        }.get(suggestion["feature"], "📊")

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{icon} {suggestion['action']}**  \n"
                    f"`{suggestion['feature']}`: "
                    f"{suggestion['current']} → {suggestion['target']}"
                )
            with col2:
                st.markdown(
                    f"<div style='text-align:center; color:{color}; font-size:1.3rem; font-weight:700;'>"
                    f"+{improvement:.3f}</div>"
                    f"<div style='text-align:center; font-size:0.75rem; color:#aaa;'>trust boost</div>",
                    unsafe_allow_html=True,
                )
            st.divider()

    total_possible = sum(s["improvement"] for s in suggestions)
    new_score = row["trust_score"] + total_possible
    gap = 0.65 - row["trust_score"]

    st.markdown(f"**Current trust score:** {row['trust_score']:.4f}")
    st.markdown(f"**Gap to threshold:** {gap:.4f}")
    st.markdown(f"**Potential score with all changes:** {min(new_score, 1.0):.4f}")

    if new_score >= 0.65:
        st.success("Applying the suggested changes would flip the decision to **ALLOW**.")
    else:
        st.error("Even with all changes, the trust score remains below threshold. Manual review required.")


def _compute_counterfactuals(row, feature_names, feature_values):
    suggestions = []

    if row.get("auth_method") == "password":
        suggestions.append({
            "feature": "auth_method",
            "action": "Enable Multi-Factor Authentication (MFA)",
            "current": "password",
            "target": "MFA",
            "improvement": 0.06,
        })

    if row.get("auth_method") in ["password", "MFA"]:
        suggestions.append({
            "feature": "auth_method",
            "action": "Switch to biometric authentication",
            "current": row.get("auth_method", "password"),
            "target": "biometric",
            "improvement": 0.045 if row.get("auth_method") == "MFA" else 0.105,
        })

    if row.get("device_trust_score", 0) < 0.7:
        improvement = (0.85 - row.get("device_trust_score", 0.5)) * 0.30
        suggestions.append({
            "feature": "device_trust_score",
            "action": "Update device security and run compliance check",
            "current": f"{row.get('device_trust_score', 0.5):.2f}",
            "target": "0.85",
            "improvement": round(max(improvement, 0.01), 3),
        })

    if not row.get("vpn_active", False):
        suggestions.append({
            "feature": "vpn_active",
            "action": "Connect via corporate VPN",
            "current": "False",
            "target": "True",
            "improvement": 0.04,
        })

    if row.get("patch_level", 0) > 30:
        improvement = min(row.get("patch_level", 30) / 365 * 0.10, 0.08)
        suggestions.append({
            "feature": "patch_level",
            "action": "Apply latest security patches",
            "current": f"{row.get('patch_level', 30)} days",
            "target": "0 days",
            "improvement": round(improvement, 3),
        })

    if row.get("location_risk") == "high":
        suggestions.append({
            "feature": "location_risk",
            "action": "Access from a trusted network location",
            "current": "high",
            "target": "low",
            "improvement": 0.07,
        })

    if row.get("failed_attempts", 0) > 2:
        improvement = min(row.get("failed_attempts", 0) * 0.015, 0.075)
        suggestions.append({
            "feature": "failed_attempts",
            "action": "Reset failed attempt counter (verify identity)",
            "current": str(row.get("failed_attempts", 0)),
            "target": "0",
            "improvement": round(improvement, 3),
        })

    return sorted(suggestions, key=lambda x: x["improvement"], reverse=True)


def _show_margin_analysis(row):
    margin = row["trust_score"] - 0.65
    color = "#00e676" if margin > 0.1 else "#ffc107" if margin > 0.05 else "#ff9800"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=margin * 100,
        title={"text": "Safety Margin Above Threshold"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 35]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 5], "color": "#ffebee"},
                {"range": [5, 15], "color": "#fff3e0"},
                {"range": [15, 35], "color": "#e8f5e9"},
            ],
        },
    ))
    fig.update_layout(height=250, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
