"""
Trust Score Gauge Component for TrustLens Dashboard.

Renders a professional gauge visualization showing the trust score.
"""

import streamlit as st
import plotly.graph_objects as go


def render_trust_gauge(trust_score: float, request_id: str = "",
                       threshold: float = 0.65):
    if trust_score >= threshold:
        color = "#00e676"
        decision = "ALLOW"
    elif trust_score >= threshold * 0.85:
        color = "#ffc107"
        decision = "DENY (borderline)"
    else:
        color = "#ff5252"
        decision = "DENY"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=trust_score * 100,
        delta={
            "reference": threshold * 100,
            "increasing": {"color": "#00e676"},
            "decreasing": {"color": "#ff5252"},
        },
        title={
            "text": (
                f"Trust Score — {request_id}<br>"
                f"<span style='font-size:0.85em;color:{color};font-weight:600'>{decision}</span>"
            ),
        },
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, threshold * 85], "color": "rgba(255,82,82,0.15)"},
                {"range": [threshold * 85, threshold * 100], "color": "rgba(255,193,7,0.15)"},
                {"range": [threshold * 100, 100], "color": "rgba(0,230,118,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ffc107", "width": 4},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))

    fig.update_layout(
        height=280,
        template="plotly_dark",
        margin=dict(t=80, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
