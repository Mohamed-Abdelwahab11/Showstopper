"""
Showstopper — AI News Analyst
Main Streamlit application entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ensure project root is in Python path ─────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from analysis.bias_detector import analyze_sources_bias, bias_spectrum_summary
from analysis.sentiment import aggregate_by_source, overall_sentiment
from app.components.charts import (
    render_bias_spectrum,
    render_sentiment_by_source,
    render_sentiment_donut,
    render_subjectivity_gauge,
)
from app.components.chat import render_chat_history, render_streaming_placeholder
from app.components.source_viewer import render_source_cards
from ingestion.chunker import chunk_articles
from ingestion.cleaner import clean_articles
from ingestion.news_fetcher import fetch_articles, fetch_top_headlines
from models.schemas import ChatMessage, RAGResponse
from rag.chain import run_rag_chain
from vectorstore.embedder import embed_texts
from vectorstore.index_manager import FAISSIndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Showstopper — AI News Analyst",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark Mode CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #060D1A !important;
    color: #C9D6E3 !important;
}

/* Main container */
.stApp {
    background: linear-gradient(135deg, #060D1A 0%, #0A1628 50%, #060D1A 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080F1E 0%, #0D1929 100%) !important;
    border-right: 1px solid #1E3050;
}

/* Header banner */
.showstopper-header {
    background: linear-gradient(135deg, #0A1628 0%, #0D2240 50%, #0A1628 100%);
    border: 1px solid #1E3050;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.showstopper-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, #00D4FF08 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, #7B61FF08 0%, transparent 50%);
    pointer-events: none;
}
.showstopper-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00D4FF, #7B61FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.showstopper-subtitle {
    color: #7B8FA1;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}
.status-ready {
    background: #00D4AA22;
    color: #00D4AA;
    border: 1px solid #00D4AA44;
}
.status-empty {
    background: #FFB34722;
    color: #FFB347;
    border: 1px solid #FFB34744;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #0D1929 !important;
    border-radius: 12px !important;
    border: 1px solid #1E3050 !important;
    margin-bottom: 12px !important;
}

/* Input box */
[data-testid="stChatInputTextArea"] {
    background: #0D1929 !important;
    border: 1px solid #1E3050 !important;
    color: #C9D6E3 !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF22, #7B61FF22) !important;
    border: 1px solid #00D4FF44 !important;
    color: #00D4FF !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00D4FF44, #7B61FF44) !important;
    border-color: #00D4FF88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D4FF, #7B61FF) !important;
    color: #060D1A !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0D1929 !important;
    border: 1px solid #1E3050 !important;
    border-radius: 10px !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #0D1929;
    border: 1px solid #1E3050;
    border-radius: 10px;
    padding: 16px;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #7B8FA1 !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom-color: #00D4FF !important;
}

/* Divider */
hr { border-color: #1E3050 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080F1E; }
::-webkit-scrollbar-thumb { background: #1E3050; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A4060; }

/* Spinner */
.stSpinner { color: #00D4FF !important; }

/* Info/Warning/Success boxes */
.stAlert { border-radius: 10px !important; border: 1px solid !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "messages": [],          # list[ChatMessage]
        "index_manager": None,   # FAISSIndexManager | None
        "current_topic": "",
        "articles_loaded": 0,
        "chunks_loaded": 0,
        "last_rag_response": None,
        "analysis_tab": "sentiment",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="showstopper-header">
    <h1 class="showstopper-title">🗞️ Showstopper</h1>
    <p class="showstopper-subtitle">
        AI-Powered News Analyst · RAG + Semantic Search + Media Bias Detection
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Sidebar — Ingestion Panel
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.divider()

    # ── Status ───────────────────────────────────────────────
    mgr: FAISSIndexManager | None = st.session_state.index_manager
    if mgr and mgr.is_ready:
        st.markdown(
            f'<div class="status-badge status-ready">✅ Index Ready — '
            f'{mgr.num_vectors:,} vectors</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Topic:** {st.session_state.current_topic}  \n"
            f"**Articles:** {st.session_state.articles_loaded}  \n"
            f"**Chunks:** {st.session_state.chunks_loaded}",
        )
    else:
        st.markdown(
            '<div class="status-badge status-empty">⚠️ No Index — Fetch articles first</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Fetch Panel ───────────────────────────────────────────
    st.markdown("### 📡 Fetch Articles")

    topic = st.text_input(
        "Search Topic",
        placeholder='e.g. "AI regulation", "climate change"',
        help="Enter a news topic to fetch and index articles from NewsAPI.",
    )

    col1, col2 = st.columns(2)
    with col1:
        max_articles = st.slider("Max Articles", 10, 100, 50, 10)
    with col2:
        days_back = st.slider("Days Back", 1, 30, 7, 1)

    include_headlines = st.checkbox("+ Top Headlines", value=True,
                                    help="Supplement /everything with /top-headlines for fresher results")

    fetch_btn = st.button("🚀 Fetch & Index", type="primary", use_container_width=True,
                          disabled=not topic.strip())

    if fetch_btn and topic.strip():
        with st.spinner("📡 Fetching articles…"):
            api_key = os.getenv("NEWS_API_KEY", "")
            articles = fetch_articles(
                topic=topic.strip(),
                api_key=api_key,
                max_articles=max_articles,
                days_back=days_back,
            )
            if include_headlines:
                headlines = fetch_top_headlines(topic=topic.strip(), api_key=api_key, max_articles=20)
                articles = articles + headlines

        if not articles:
            st.error("No articles found. Try a different topic or check your API key.")
        else:
            with st.spinner(f"🧹 Cleaning {len(articles)} articles…"):
                cleaned = clean_articles(articles)

            with st.spinner("✂️ Chunking…"):
                chunks = chunk_articles(cleaned)

            if not chunks:
                st.error("No valid chunks produced. Articles may be too short.")
            else:
                with st.spinner(f"🧮 Embedding {len(chunks)} chunks…"):
                    texts = [c.text for c in chunks]
                    embeddings = embed_texts(texts, show_progress=False)

                with st.spinner("🗄️ Building FAISS index…"):
                    new_mgr = FAISSIndexManager()
                    new_mgr.build(chunks, embeddings)
                    new_mgr.save()

                st.session_state.index_manager = new_mgr
                st.session_state.current_topic = topic.strip()
                st.session_state.articles_loaded = len(cleaned)
                st.session_state.chunks_loaded = len(chunks)
                st.session_state.messages = []  # Reset chat
                st.session_state.last_rag_response = None

                st.success(
                    f"✅ Indexed **{len(cleaned)}** articles → **{len(chunks)}** chunks"
                )
                st.rerun()

    st.divider()

    # ── LLM Config ────────────────────────────────────────────
    st.markdown("### 🤖 LLM Provider")
    provider_display = os.getenv("LLM_PROVIDER", "demo").upper()
    st.markdown(f"**Active:** `{provider_display}`")
    st.caption("Set `LLM_PROVIDER` in your `.env` file:  \n`openai` · `groq` · `demo`")

    st.divider()

    # ── Clear Chat ────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_rag_response = None
        st.rerun()

    # ── Load Saved Index ──────────────────────────────────────
    if st.button("📂 Load Saved Index", use_container_width=True):
        load_mgr = FAISSIndexManager()
        if load_mgr.load():
            st.session_state.index_manager = load_mgr
            st.success(f"Loaded index: {load_mgr.num_vectors:,} vectors")
            st.rerun()
        else:
            st.warning("No saved index found. Fetch articles first.")


# ─────────────────────────────────────────────────────────────
#  Main Layout: Chat (left) + Analysis (right)
# ─────────────────────────────────────────────────────────────

chat_col, analysis_col = st.columns([3, 2], gap="large")

# ── LEFT: Chat ────────────────────────────────────────────────
with chat_col:
    st.markdown("### 💬 Ask the Analyst")

    mgr = st.session_state.index_manager

    if not mgr or not mgr.is_ready:
        st.markdown("""
        <div style="background:#0D1929;border:1px solid #1E3050;border-radius:12px;
                    padding:32px;text-align:center;color:#7B8FA1">
            <div style="font-size:2.5rem;margin-bottom:12px">📡</div>
            <div style="font-size:1.1rem;font-weight:600;color:#C9D6E3;margin-bottom:8px">
                No Articles Indexed Yet
            </div>
            <div>Enter a topic in the sidebar and click <strong style="color:#00D4FF">
            Fetch & Index</strong> to get started.</div>
            <div style="margin-top:16px;font-size:13px">
                Example topics: <em>artificial intelligence</em> · 
                <em>climate change</em> · <em>Ukraine war</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Render chat history
        render_chat_history(st.session_state.messages)

        # Suggested questions
        if not st.session_state.messages:
            topic = st.session_state.current_topic
            suggestions = [
                f"What is the media's overall stance on {topic}?",
                f"Which outlets are most critical of {topic}?",
                f"Summarize the key viewpoints from different sources on {topic}.",
                f"What language do journalists use when discussing {topic}?",
            ]
            st.markdown("**💡 Suggested Questions:**")
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                        st.session_state["_pending_question"] = q
                        st.rerun()

        # Chat input
        user_input = st.chat_input(
            "Ask about media coverage, bias, sentiment…",
            key="main_chat_input",
        )

        # Handle pending suggestion clicks
        if "_pending_question" in st.session_state:
            user_input = st.session_state.pop("_pending_question")

        if user_input:
            # Add user message
            user_msg = ChatMessage(role="user", content=user_input)
            st.session_state.messages.append(user_msg)

            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)

            # Generate answer
            with st.chat_message("assistant", avatar="🗞️"):
                with st.spinner("🔍 Retrieving & Analyzing…"):
                    response: RAGResponse = run_rag_chain(
                        query=user_input,
                        index_manager=mgr,
                    )

                st.markdown(response.answer)

                # Source pills
                if response.sources:
                    pills = " ".join(
                        f'<a href="{s.url}" target="_blank" style="'
                        f'text-decoration:none;background:#0D2038;color:#00D4FF;'
                        f'border:1px solid #00D4FF44;border-radius:12px;'
                        f'padding:3px 10px;font-size:11px;margin:2px;display:inline-block">'
                        f'📰 {s.source_name}</a>'
                        for s in response.sources[:6]
                    )
                    st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)

                # Meta row
                model = response.model_used or "demo"
                latency = f"{response.latency_ms:.0f}ms" if response.latency_ms else "—"
                st.markdown(
                    f'<div style="font-size:11px;color:#4A6080;margin-top:6px">'
                    f'🤖 {model} &nbsp;·&nbsp; ⚡ {latency} &nbsp;·&nbsp; '
                    f'📄 {len(response.retrieved_chunks)} chunks retrieved'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Save to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=response.answer,
                rag_response=response,
            )
            st.session_state.messages.append(assistant_msg)
            st.session_state.last_rag_response = response


# ── RIGHT: Analysis Dashboard ─────────────────────────────────
with analysis_col:
    st.markdown("### 📊 Analysis Dashboard")

    last_response: RAGResponse | None = st.session_state.last_rag_response

    if not last_response:
        st.markdown("""
        <div style="background:#0D1929;border:1px solid #1E3050;border-radius:12px;
                    padding:32px;text-align:center;color:#7B8FA1">
            <div style="font-size:2.5rem;margin-bottom:12px">📊</div>
            <div style="font-size:1rem;font-weight:600;color:#C9D6E3;margin-bottom:8px">
                Analysis will appear here
            </div>
            <div>Ask a question to see sentiment analysis,<br>
            bias detection, and source breakdowns.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chunks = last_response.retrieved_chunks

        # Tabs
        tab_sentiment, tab_bias, tab_sources = st.tabs([
            "😊 Sentiment", "⚖️ Bias", "📚 Sources"
        ])

        with tab_sentiment:
            overall = overall_sentiment(chunks)
            per_source = aggregate_by_source(chunks)

            col_donut, col_gauge = st.columns(2)
            with col_donut:
                render_sentiment_donut(overall)
            with col_gauge:
                render_subjectivity_gauge(overall)

            if per_source:
                render_sentiment_by_source(per_source)

        with tab_bias:
            bias_results = analyze_sources_bias(chunks)
            spectrum = bias_spectrum_summary(bias_results)

            if bias_results:
                render_bias_spectrum(bias_results)

                st.markdown("#### Source Bias Summary")
                for br in bias_results[:8]:
                    lean_emoji = {
                        "left": "◀◀", "center-left": "◀", "center": "⬤",
                        "center-right": "▶", "right": "▶▶"
                    }.get(br.lean, "?")
                    st.markdown(
                        f"**{lean_emoji} {br.source_name}** — "
                        f"{br.lean.replace('-', ' ').title()} "
                        f"(confidence: {br.confidence:.0%})"
                    )
            else:
                st.info("No bias data available for retrieved sources.")

        with tab_sources:
            bias_map = {
                br.source_name: br
                for br in analyze_sources_bias(chunks)
            }
            render_source_cards(last_response.sources, bias_map=bias_map)


# ─────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#2A4060;font-size:12px">'
    'Showstopper · AI News Analyst · Built with NewsAPI + FAISS + LangChain + Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
