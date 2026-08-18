"""
SHAP Plot Components for TrustLens Dashboard.

Renders SHAP waterfall, force, and summary plots with dark theme styling.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render_shap_plot(shap_values: np.ndarray, feature_names: list,
                     request_id: str = ""):
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    sorted_values = shap_values[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    colors = ["#ff5252" if v > 0 else "#00e676" for v in sorted_values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_values,
        y=sorted_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in sorted_values],
        textposition="auto",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title=dict(
            text=f"SHAP Feature Contributions — {request_id}",
            font=dict(size=14),
        ),
        xaxis_title="SHAP Value (→ DENY | ← ALLOW)",
        yaxis_title="",
        height=max(350, len(feature_names) * 40),
        template="plotly_dark",
        margin=dict(l=160, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=2),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_summary(shap_values_matrix: np.ndarray, feature_names: list):
    mean_abs = np.abs(shap_values_matrix).mean(axis=0)
    sorted_indices = np.argsort(mean_abs)[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mean_abs[sorted_indices],
        y=[feature_names[i] for i in sorted_indices],
        orientation="h",
        marker=dict(
            color=mean_abs[sorted_indices],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
        text=[f"{v:.4f}" for v in mean_abs[sorted_indices]],
        textposition="auto",
    ))

    fig.update_layout(
        title="SHAP Global Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        height=max(350, len(feature_names) * 35),
        template="plotly_dark",
        margin=dict(l=160, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
