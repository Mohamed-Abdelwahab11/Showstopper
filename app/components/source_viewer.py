"""
Showstopper — Source Viewer
Renders expandable glassmorphism source cards with
relevance scores, bias badges, and snippet text.
"""

from __future__ import annotations

import streamlit as st

from models.schemas import BiasResult, SourceCitation

_BIAS_BADGE_COLORS = {
    "left":         ("#4B9FFF", "◀ Left"),
    "center-left":  ("#7BC8FF", "◀ Ctr-Left"),
    "center":       ("#A0C4A0", "⬤ Center"),
    "center-right": ("#FFB347", "Ctr-Right ▶"),
    "right":        ("#FF6B6B", "Right ▶"),
    "unknown":      ("#666666", "? Unknown"),
}

_CREDIBILITY_BADGE = {
    "high":    ("🟢", "High Credibility"),
    "medium":  ("🟡", "Medium Credibility"),
    "low":     ("🔴", "Low Credibility"),
    "unknown": ("⚪", "Unknown"),
}


def _relevance_bar(relevance: float) -> str:
    """Generate an HTML progress bar for relevance score."""
    pct = int(relevance * 100)
    color = "#00D4AA" if pct >= 70 else ("#FFB347" if pct >= 40 else "#FF4B6E")
    return (
        f'<div style="background:#1E2D40;border-radius:4px;height:6px;width:100%;margin:4px 0">'
        f'<div style="background:{color};height:6px;border-radius:4px;width:{pct}%"></div>'
        f'</div>'
        f'<small style="color:#7B8FA1">Relevance: {pct}%</small>'
    )


def render_source_cards(
    sources: list[SourceCitation],
    bias_map: dict[str, BiasResult] | None = None,
) -> None:
    """
    Render a list of source citation cards in Streamlit.

    Parameters
    ----------
    sources  : List of SourceCitation from the RAG response
    bias_map : Optional dict mapping source_name → BiasResult
    """
    if not sources:
        st.info("No sources retrieved for this query.")
        return

    st.markdown(f"#### 📚 Retrieved Sources ({len(sources)})")

    for i, source in enumerate(sources):
        bias = bias_map.get(source.source_name) if bias_map else None
        bias_color, bias_label = _BIAS_BADGE_COLORS.get(
            bias.lean if bias else "unknown", ("#666", "? Unknown")
        )
        cred_icon, cred_label = _CREDIBILITY_BADGE.get(
            bias.credibility_tier if bias else "unknown", ("⚪", "Unknown")
        )

        # Build card header
        header_cols = st.columns([3, 1, 1])
        with header_cols[0]:
            st.markdown(
                f"**{source.source_name}** &nbsp;"
                f'<span style="background:{bias_color}22;color:{bias_color};'
                f'padding:2px 8px;border-radius:10px;font-size:11px;border:1px solid {bias_color}44">'
                f'{bias_label}</span>',
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown(f"{cred_icon} {cred_label}", help="Credibility tier based on Ad Fontes ratings")
        with header_cols[2]:
            date = source.published_at[:10] if source.published_at else "N/A"
            st.markdown(f"🗓️ {date}")

        with st.expander(f"📰 {source.title or 'Untitled Article'}", expanded=(i == 0)):
            st.markdown(_relevance_bar(source.relevance), unsafe_allow_html=True)

            if source.snippet:
                st.markdown(
                    f'<div style="background:#0D1929;border-left:3px solid #00D4FF;'
                    f'padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0;'
                    f'font-size:14px;color:#B8CCD8;line-height:1.6">'
                    f'{source.snippet}</div>',
                    unsafe_allow_html=True,
                )

            if bias and bias.flagged_phrases:
                st.markdown("**⚠️ Flagged Language:**")
                badges = " ".join(
                    f'<span style="background:#FF4B6E22;color:#FF4B6E;'
                    f'padding:2px 8px;border-radius:10px;font-size:11px;margin:2px">'
                    f'"{p}"</span>'
                    for p in bias.flagged_phrases[:5]
                )
                st.markdown(badges, unsafe_allow_html=True)

            st.markdown(
                f'<a href="{source.url}" target="_blank" style="'
                f'color:#00D4FF;text-decoration:none;font-size:13px">'
                f'🔗 Read full article →</a>',
                unsafe_allow_html=True,
            )

        st.divider()
