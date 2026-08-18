"""
Threat Intelligence Visualization Components.

Renders risk heatmaps, threat pattern analysis, and anomaly visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def render_threat_heatmap(decisions: pd.DataFrame):
    hours = list(range(24))
    risk_levels = ["low", "medium", "high"]

    heatmap_data = np.zeros((3, 24))
    for i, risk in enumerate(risk_levels):
        for h in hours:
            mask = (decisions["time_of_access"] == h) & (decisions["location_risk"] == risk)
            deny_count = ((decisions.loc[mask, "decision"] == "DENY").sum() if mask.any() else 0)
            heatmap_data[i, h] = deny_count

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{h:02d}:00" for h in hours],
        y=["Low Risk", "Medium Risk", "High Risk"],
        colorscale="YlOrRd",
        text=heatmap_data.astype(int),
        texttemplate="%{text}",
        hovertemplate="Hour: %{x}<br>Risk: %{y}<br>Denials: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Denial Heatmap: Time of Day vs Location Risk",
        xaxis_title="Hour of Day",
        yaxis_title="Location Risk Level",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_radar(decisions: pd.DataFrame):
    categories = [
        "Device Health",
        "Auth Strength",
        "Location Safety",
        "Behavior Score",
        "Patch Currency",
    ]

    denied = decisions[decisions["decision"] == "DENY"]
    allowed = decisions[decisions["decision"] == "ALLOW"]

    def avg_metrics(df):
        return [
            df["device_trust_score"].mean() if len(df) else 0,
            df["auth_method"].map({"password": 0.3, "MFA": 0.7, "biometric": 1.0}).mean() if len(df) else 0,
            1 - df["location_risk"].map({"low": 0.1, "medium": 0.5, "high": 0.9}).mean() if len(df) else 0,
            max(0, 1 - df["failed_attempts"].mean() / 10) if len(df) else 0,
            max(0, 1 - df["patch_level"].mean() / 365) if len(df) else 0,
        ]

    allow_vals = avg_metrics(allowed)
    deny_vals = avg_metrics(denied)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=allow_vals + [allow_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Allowed",
        line_color="#00e676",
        fillcolor="rgba(0,230,118,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=deny_vals + [deny_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Denied",
        line_color="#ff5252",
        fillcolor="rgba(255,82,82,0.2)",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Risk Profile: Allowed vs Denied Requests",
        template="plotly_dark",
        height=450,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
