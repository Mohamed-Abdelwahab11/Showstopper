"""
Showstopper — Sentiment & Bias Charts
Plotly visualizations rendered in dark mode.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from models.schemas import BiasResult, SentimentResult

# ── Color palette ─────────────────────────────────────────────
_POSITIVE_COLOR = "#00D4AA"     # cyan-green
_NEGATIVE_COLOR = "#FF4B6E"     # rose-red
_NEUTRAL_COLOR  = "#7B8FA1"     # steel blue-grey
_SUBJ_COLOR     = "#FFB347"     # amber

_BIAS_COLORS = {
    "left":         "#4B9FFF",
    "center-left":  "#7BC8FF",
    "center":       "#A0C4A0",
    "center-right": "#FFB347",
    "right":        "#FF6B6B",
    "unknown":      "#555555",
}

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#C9D6E3", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)


def render_sentiment_donut(overall: SentimentResult) -> None:
    """Donut chart: overall positive / negative / neutral split."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [overall.positive, overall.neutral, overall.negative]
    colors = [_POSITIVE_COLOR, _NEUTRAL_COLOR, _NEGATIVE_COLOR]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0A0F1E", width=2)),
        textfont=dict(size=12, color="#C9D6E3"),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))

    compound = overall.compound
    compound_color = _POSITIVE_COLOR if compound >= 0.05 else (_NEGATIVE_COLOR if compound <= -0.05 else _NEUTRAL_COLOR)

    fig.add_annotation(
        text=f"<b>{compound:+.2f}</b><br><span style='font-size:10px'>compound</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color=compound_color),
        showarrow=False,
    )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Overall Sentiment", font=dict(size=14), x=0.5),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=280,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_sentiment_by_source(sentiments: list[SentimentResult]) -> None:
    """Horizontal grouped bar chart: sentiment per source."""
    if not sentiments:
        st.info("No sentiment data available.")
        return

    sources = [s.source_name for s in sentiments]
    positives = [s.positive for s in sentiments]
    negatives = [s.negative for s in sentiments]
    neutrals  = [s.neutral for s in sentiments]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive", x=positives, y=sources, orientation="h",
        marker_color=_POSITIVE_COLOR,
        hovertemplate="%{y}: %{x:.1%}<extra>Positive</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Neutral", x=neutrals, y=sources, orientation="h",
        marker_color=_NEUTRAL_COLOR,
        hovertemplate="%{y}: %{x:.1%}<extra>Neutral</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Negative", x=negatives, y=sources, orientation="h",
        marker_color=_NEGATIVE_COLOR,
        hovertemplate="%{y}: %{x:.1%}<extra>Negative</extra>",
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        barmode="stack",
        title=dict(text="Sentiment by Source", font=dict(size=14), x=0.5),
        xaxis=dict(tickformat=".0%", gridcolor="#1E2D40"),
        yaxis=dict(gridcolor="#1E2D40"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        height=max(200, len(sources) * 40 + 80),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_bias_spectrum(bias_results: list[BiasResult]) -> None:
    """Horizontal bias spectrum showing political lean per source."""
    if not bias_results:
        st.info("No bias data available.")
        return

    lean_order = ["left", "center-left", "center", "center-right", "right", "unknown"]
    lean_positions = {lean: i for i, lean in enumerate(lean_order)}

    sources = [b.source_name for b in bias_results if b.lean != "unknown"]
    positions = [lean_positions.get(b.lean, 5) for b in bias_results if b.lean != "unknown"]
    colors = [_BIAS_COLORS.get(b.lean, "#555") for b in bias_results if b.lean != "unknown"]
    labels = [b.lean.replace("-", " ").title() for b in bias_results if b.lean != "unknown"]

    if not sources:
        st.info("No recognized sources for bias analysis.")
        return

    fig = go.Figure(go.Scatter(
        x=positions, y=sources,
        mode="markers+text",
        marker=dict(color=colors, size=14, line=dict(color="#0A0F1E", width=1)),
        text=labels,
        textposition="middle right",
        textfont=dict(size=10, color="#C9D6E3"),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Media Bias Spectrum", font=dict(size=14), x=0.5),
        xaxis=dict(
            tickvals=list(range(len(lean_order))),
            ticktext=["◀ Left", "Center-Left", "Center", "Center-Right", "Right ▶", "Unknown"],
            gridcolor="#1E2D40",
            range=[-0.5, 5.5],
        ),
        yaxis=dict(gridcolor="#1E2D40"),
        height=max(200, len(sources) * 40 + 80),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_subjectivity_gauge(overall: SentimentResult) -> None:
    """Gauge chart showing subjectivity score."""
    val = overall.subjectivity * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "%", "font": {"size": 20, "color": _SUBJ_COLOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#C9D6E3", "tickfont": {"size": 10}},
            "bar": {"color": _SUBJ_COLOR, "thickness": 0.3},
            "bgcolor": "#1E2D40",
            "steps": [
                {"range": [0, 33], "color": "#0A1628"},
                {"range": [33, 66], "color": "#0F1F35"},
                {"range": [66, 100], "color": "#162440"},
            ],
            "threshold": {
                "line": {"color": "#FF4B6E", "width": 2},
                "thickness": 0.75,
                "value": 65,
            },
        },
        title={"text": "Subjectivity", "font": {"size": 13, "color": "#C9D6E3"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(**_DARK_LAYOUT, height=200)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
