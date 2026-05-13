"""
Showstopper — Chat Component
Renders the conversation history and handles message display.
"""

from __future__ import annotations

import streamlit as st

from models.schemas import ChatMessage, RAGResponse

# ── Avatar HTML ───────────────────────────────────────────────
_USER_AVATAR = "🧑‍💻"
_BOT_AVATAR = "🗞️"


def _render_sources_pill(response: RAGResponse) -> None:
    """Render compact source pills below an assistant message."""
    if not response.sources:
        return

    pills = []
    for src in response.sources[:5]:
        pills.append(
            f'<a href="{src.url}" target="_blank" style="'
            f'text-decoration:none;background:#0D2038;color:#00D4FF;'
            f'border:1px solid #00D4FF44;border-radius:12px;'
            f'padding:3px 10px;font-size:11px;margin:2px;display:inline-block">'
            f'📰 {src.source_name}</a>'
        )

    st.markdown(
        '<div style="margin-top:8px;flex-wrap:wrap">' + " ".join(pills) + "</div>",
        unsafe_allow_html=True,
    )


def _render_meta_row(response: RAGResponse) -> None:
    """Render metadata row: model, latency, chunk count."""
    model = response.model_used or "demo"
    latency = f"{response.latency_ms:.0f}ms" if response.latency_ms else "—"
    n_chunks = len(response.retrieved_chunks)

    st.markdown(
        f'<div style="font-size:11px;color:#4A6080;margin-top:6px">'
        f'🤖 {model} &nbsp;·&nbsp; ⚡ {latency} &nbsp;·&nbsp; '
        f'📄 {n_chunks} chunks retrieved'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_chat_history(messages: list[ChatMessage]) -> None:
    """Render the full conversation history."""
    for msg in messages:
        if msg.role == "user":
            with st.chat_message("user", avatar=_USER_AVATAR):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant", avatar=_BOT_AVATAR):
                st.markdown(msg.content)
                if msg.rag_response:
                    _render_sources_pill(msg.rag_response)
                    _render_meta_row(msg.rag_response)


def render_streaming_placeholder() -> "st.delta_generator.DeltaGenerator":
    """Return a placeholder for streaming output."""
    with st.chat_message("assistant", avatar=_BOT_AVATAR):
        return st.empty()
