"""
LIME Plot Component for TrustLens Dashboard.

Renders LIME feature weight bar charts with professional dark theme styling.
"""

import streamlit as st
import plotly.graph_objects as go


def render_lime_plot(feature_weights: list, request_id: str = ""):
    sorted_weights = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)

    names = [fw[0] for fw in sorted_weights]
    weights = [fw[1] for fw in sorted_weights]
    colors = ["#ff5252" if w > 0 else "#00e676" for w in weights]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weights,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{w:+.4f}" for w in weights],
        textposition="auto",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title=dict(
            text=f"LIME Feature Weights — {request_id}",
            font=dict(size=14),
        ),
        xaxis_title="Feature Weight (→ DENY | ← ALLOW)",
        yaxis_title="",
        height=max(350, len(names) * 40),
        template="plotly_dark",
        margin=dict(l=160, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=2),
    )
    st.plotly_chart(fig, use_container_width=True)
