"""
Showstopper — AI News Analyst
Premium Streamlit UI — Bloomberg × ChatGPT dark aesthetic
"""

from __future__ import annotations
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from analysis.bias_detector import analyze_sources_bias
from analysis.sentiment import aggregate_by_source, overall_sentiment
from app.components.charts import (
    render_bias_spectrum, render_sentiment_by_source,
    render_sentiment_donut, render_subjectivity_gauge,
)
from app.components.source_viewer import render_source_cards
from ingestion.chunker import chunk_articles
from ingestion.cleaner import clean_articles
from ingestion.news_fetcher import fetch_articles, fetch_top_headlines
from models.schemas import ChatMessage, RAGResponse
from rag.chain import run_rag_chain
from vectorstore.embedder import embed_texts
from vectorstore.index_manager import FAISSIndexManager

logging.basicConfig(level=logging.WARNING)

# ─── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Showstopper — AI News Analyst",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Background ── */
.stApp {
    background: #060D1A !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,180,255,.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(123,97,255,.05) 0%, transparent 60%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#08101F 0%,#0B1628 100%) !important;
    border-right: 1px solid rgba(0,212,255,.12) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg,#080F1E,#0C1E38,#081525);
    border: 1px solid rgba(0,212,255,.15);
    border-radius: 20px;
    padding: 28px 36px 22px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; inset:0;
    background:
        radial-gradient(circle at 15% 50%, rgba(0,212,255,.08) 0%,transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(123,97,255,.07) 0%,transparent 50%);
    pointer-events:none;
}
.hero-title {
    font-size:2.6rem; font-weight:800; letter-spacing:-1px; margin:0;
    background:linear-gradient(90deg,#00D4FF 0%,#7B61FF 50%,#00D4AA 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { color:#4A7090; font-size:.9rem; margin-top:6px; letter-spacing:.3px; }
.hero-tags { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.hero-tag {
    padding:3px 12px; border-radius:20px; font-size:11px; font-weight:500;
    border:1px solid; letter-spacing:.3px;
}

/* ── Status pill ── */
.pill-ready { background:#00D4AA18;color:#00D4AA;border-color:#00D4AA33; }
.pill-empty  { background:#FFB34718;color:#FFB347;border-color:#FFB34733; }
.pill-info   { background:#00D4FF18;color:#00D4FF;border-color:#00D4FF33; }
.pill-right  { background:#7B61FF18;color:#7B61FF;border-color:#7B61FF33; }

/* ── Stat card ── */
.stat-row { display:flex; gap:12px; margin:14px 0 0; }
.stat-card {
    flex:1; background:#0C1828; border:1px solid #1A2E45;
    border-radius:12px; padding:12px 16px; text-align:center;
}
.stat-num { font-size:1.6rem; font-weight:700; color:#00D4FF; line-height:1; }
.stat-lbl { font-size:11px; color:#4A7090; margin-top:3px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background:#0C1828 !important;
    border:1px solid #1A2E45 !important;
    border-radius:14px !important;
    margin-bottom:10px !important;
}
[data-testid="stChatMessageContent"] p { color:#C5D5E5 !important; line-height:1.7; }

/* ── Chat input ── */
[data-testid="stChatInputTextArea"] {
    background:#0C1828 !important; border:1px solid #1E3550 !important;
    color:#C5D5E5 !important; border-radius:14px !important; font-size:14px !important;
}
[data-testid="stChatInputTextArea"]:focus { border-color:#00D4FF66 !important; }

/* ── Buttons ── */
.stButton>button {
    background:linear-gradient(135deg,#00D4FF18,#7B61FF18) !important;
    border:1px solid #00D4FF33 !important; color:#00D4FF !important;
    border-radius:10px !important; font-weight:500 !important; font-size:13px !important;
    transition:all .25s ease !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#00D4FF30,#7B61FF30) !important;
    border-color:#00D4FF66 !important; transform:translateY(-1px) !important;
    box-shadow:0 6px 20px rgba(0,212,255,.15) !important;
}
[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#00D4FF,#7B61FF) !important;
    color:#050C18 !important; border:none !important; font-weight:700 !important;
}
[data-testid="baseButton-primary"]:hover {
    box-shadow:0 8px 25px rgba(0,212,255,.35) !important;
    transform:translateY(-2px) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background:#0C1828 !important; border:1px solid #1A2E45 !important;
    border-radius:12px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { color:#4A7090 !important; font-weight:500 !important; }
[data-testid="stTabs"] [aria-selected="true"] { color:#00D4FF !important; }
[data-testid="stTabs"] [aria-selected="true"] + span { background:#00D4FF !important; }

/* ── Inputs / Sliders ── */
[data-testid="stTextInput"] input {
    background:#0C1828 !important; border:1px solid #1E3550 !important;
    color:#C5D5E5 !important; border-radius:10px !important;
}
[data-testid="stTextInput"] input:focus { border-color:#00D4FF66 !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color:#4A7090 !important; }

/* ── Alerts ── */
.stAlert { border-radius:12px !important; }

/* ── Divider ── */
hr { border-color:#1A2E45 !important; }

/* ── Sidebar section header ── */
.sb-section {
    font-size:10px; font-weight:600; letter-spacing:1.5px;
    color:#2A4A65; text-transform:uppercase; margin:18px 0 8px;
}

/* ── Empty state ── */
.empty-state {
    background:#0C1828; border:1px dashed #1A2E45; border-radius:16px;
    padding:40px 20px; text-align:center;
}
.empty-icon { font-size:3rem; margin-bottom:12px; }
.empty-title { font-size:1.1rem; font-weight:600; color:#C5D5E5; margin-bottom:6px; }
.empty-sub { color:#4A7090; font-size:.875rem; line-height:1.6; }

/* ── Suggestion chips ── */
.chip-btn .stButton>button {
    background:#0C1828 !important; border:1px solid #1A2E45 !important;
    color:#7B9BB5 !important; font-size:12px !important; text-align:left !important;
    border-radius:10px !important; padding:8px 12px !important;
    white-space:normal !important; height:auto !important; min-height:48px !important;
}
.chip-btn .stButton>button:hover {
    border-color:#00D4FF44 !important; color:#00D4FF !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#070E1C; }
::-webkit-scrollbar-thumb { background:#1A2E45; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#243D57; }

/* ── Footer ── */
.footer { text-align:center; color:#1A2E45; font-size:11px; padding:24px 0 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ───────────────────────────────────────────
for k, v in {
    "messages": [], "index_manager": None,
    "current_topic": "", "articles_loaded": 0,
    "chunks_loaded": 0, "last_rag_response": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Hero Header ────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1 class="hero-title">🗞️ Showstopper</h1>
  <p class="hero-sub">AI-Powered Media Analyst · Ask any question about news coverage</p>
  <div class="hero-tags">
    <span class="hero-tag pill-info">RAG Pipeline</span>
    <span class="hero-tag pill-ready">FAISS Vector Search</span>
    <span class="hero-tag pill-right">Sentiment Analysis</span>
    <span class="hero-tag pill-empty">Bias Detection</span>
    <span class="hero-tag pill-info">NewsAPI Live</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    # Logo area
    st.markdown("""
    <div style="padding:20px 0 4px;text-align:center">
      <div style="font-size:2rem">🗞️</div>
      <div style="font-size:1.1rem;font-weight:700;background:linear-gradient(90deg,#00D4FF,#7B61FF);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:-.3px">Showstopper</div>
      <div style="font-size:10px;color:#2A4A65;margin-top:2px">AI NEWS ANALYST</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Index status ─────────────────────────────────────────
    mgr: FAISSIndexManager | None = st.session_state.index_manager
    if mgr and mgr.is_ready:
        st.markdown(f"""
        <div style="background:#00D4AA0D;border:1px solid #00D4AA25;border-radius:12px;padding:14px 16px">
          <div style="color:#00D4AA;font-size:12px;font-weight:600;margin-bottom:8px">✅ INDEX READY</div>
          <div style="color:#7B9BB5;font-size:12px">Topic: <span style="color:#C5D5E5;font-weight:500">
            {st.session_state.current_topic}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card">
            <div class="stat-num">{st.session_state.articles_loaded}</div>
            <div class="stat-lbl">Articles</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{st.session_state.chunks_loaded}</div>
            <div class="stat-lbl">Chunks</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{mgr.num_vectors:,}</div>
            <div class="stat-lbl">Vectors</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FFB3470D;border:1px solid #FFB34725;border-radius:12px;
                    padding:14px 16px;text-align:center">
          <div style="font-size:1.4rem;margin-bottom:4px">📡</div>
          <div style="color:#FFB347;font-size:12px;font-weight:600">No Index Loaded</div>
          <div style="color:#4A7090;font-size:11px;margin-top:4px">Fetch articles to get started</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Fetch section ─────────────────────────────────────────
    st.markdown('<div class="sb-section">📡 News Ingestion</div>', unsafe_allow_html=True)

    topic = st.text_input(
        "Search Topic", placeholder='e.g. "AI regulation", "Ukraine war"',
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        max_articles = st.slider("Articles", 10, 100, 50, 10)
    with col_b:
        days_back = st.slider("Days", 1, 30, 7, 1)

    include_headlines = st.checkbox("Include Top Headlines", value=True)

    fetch_btn = st.button(
        "🚀 Fetch & Index Articles", type="primary",
        use_container_width=True, disabled=not topic.strip()
    )

    if fetch_btn and topic.strip():
        api_key = os.getenv("NEWS_API_KEY", "")

        prog = st.progress(0, text="📡 Fetching from NewsAPI…")
        articles = fetch_articles(topic.strip(), api_key=api_key,
                                  max_articles=max_articles, days_back=days_back)
        if include_headlines:
            prog.progress(20, text="📰 Fetching headlines…")
            articles += fetch_top_headlines(topic.strip(), api_key=api_key, max_articles=20)

        prog.progress(35, text="🧹 Cleaning articles…")
        cleaned = clean_articles(articles)

        if not cleaned:
            prog.empty()
            st.error("No valid articles found. Try a different topic.")
        else:
            prog.progress(55, text="✂️ Chunking text…")
            chunks = chunk_articles(cleaned)

            prog.progress(70, text="🧮 Embedding chunks…")
            embeddings = embed_texts([c.text for c in chunks], show_progress=False)

            prog.progress(90, text="⚡ Building FAISS index…")
            new_mgr = FAISSIndexManager()
            new_mgr.build(chunks, embeddings)
            new_mgr.save()

            prog.progress(100, text="✅ Done!")
            time.sleep(0.5)
            prog.empty()

            st.session_state.update({
                "index_manager": new_mgr,
                "current_topic": topic.strip(),
                "articles_loaded": len(cleaned),
                "chunks_loaded": len(chunks),
                "messages": [],
                "last_rag_response": None,
            })
            st.success(f"✅ {len(cleaned)} articles → {len(chunks)} chunks indexed!")
            st.rerun()

    st.divider()

    # ── LLM status ────────────────────────────────────────────
    st.markdown('<div class="sb-section">🤖 LLM Provider</div>', unsafe_allow_html=True)
    provider = os.getenv("LLM_PROVIDER", "demo").upper()
    color = {"OPENAI": "#00D4AA", "GROQ": "#7B61FF"}.get(provider, "#FFB347")
    st.markdown(f"""
    <div style="background:{color}11;border:1px solid {color}33;border-radius:10px;
                padding:10px 14px;display:flex;align-items:center;gap:10px">
      <div style="width:8px;height:8px;background:{color};border-radius:50%;
                  box-shadow:0 0 6px {color}"></div>
      <div>
        <div style="color:{color};font-size:12px;font-weight:600">{provider}</div>
        <div style="color:#4A7090;font-size:10px">Set LLM_PROVIDER in .env</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Actions ───────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_rag_response = None
            st.rerun()
    with col_d:
        if st.button("📂 Load Index", use_container_width=True):
            load_mgr = FAISSIndexManager()
            if load_mgr.load():
                st.session_state.index_manager = load_mgr
                st.success(f"Loaded {load_mgr.num_vectors:,} vectors")
                st.rerun()
            else:
                st.warning("No saved index found.")


# ─── Main Layout ─────────────────────────────────────────────
chat_col, dash_col = st.columns([11, 9], gap="large")

# ══════════════════════════════════════════════════════════════
#  LEFT — Chat
# ══════════════════════════════════════════════════════════════
with chat_col:
    st.markdown("""
    <div style="font-size:13px;font-weight:600;color:#4A7090;letter-spacing:1px;
                text-transform:uppercase;margin-bottom:14px">
      💬 Media Analyst Chat
    </div>
    """, unsafe_allow_html=True)

    mgr = st.session_state.index_manager

    if not mgr or not mgr.is_ready:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📡</div>
          <div class="empty-title">No Articles Indexed Yet</div>
          <div class="empty-sub">
            Enter a news topic in the sidebar and click<br>
            <strong style="color:#00D4FF">Fetch &amp; Index Articles</strong> to begin.<br><br>
            Try: <em>AI regulation</em> · <em>climate change</em> · <em>Ukraine war</em>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat history
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg.role == "user" else "🗞️"
            with st.chat_message(msg.role, avatar=avatar):
                st.markdown(msg.content)
                if msg.role == "assistant" and msg.rag_response:
                    resp = msg.rag_response
                    # Source pills
                    if resp.sources:
                        pills = " ".join(
                            f'<a href="{s.url}" target="_blank" style="text-decoration:none;'
                            f'background:#0C1828;color:#00D4FF;border:1px solid #00D4FF33;'
                            f'border-radius:20px;padding:3px 10px;font-size:11px;'
                            f'display:inline-block;margin:2px">📰 {s.source_name}</a>'
                            for s in resp.sources[:6]
                        )
                        st.markdown(f'<div style="margin-top:8px">{pills}</div>',
                                    unsafe_allow_html=True)
                    # Meta
                    lat = f"{resp.latency_ms:.0f}ms" if resp.latency_ms else "—"
                    st.markdown(
                        f'<div style="font-size:10px;color:#2A4060;margin-top:6px">'
                        f'🤖 {resp.model_used} &nbsp;·&nbsp; ⚡ {lat} '
                        f'&nbsp;·&nbsp; 📄 {len(resp.retrieved_chunks)} chunks'
                        f'</div>', unsafe_allow_html=True)

        # Suggestions (only when no messages yet)
        if not st.session_state.messages:
            t = st.session_state.current_topic
            sugs = [
                f"What is the media's overall stance on {t}?",
                f"Which outlets are most critical of {t}?",
                f"Summarize key viewpoints from different sources on {t}.",
                f"What emotional language do journalists use about {t}?",
            ]
            st.markdown(
                '<div style="font-size:11px;color:#2A4060;font-weight:600;'
                'letter-spacing:.5px;margin-bottom:8px">💡 SUGGESTED QUESTIONS</div>',
                unsafe_allow_html=True)
            cols = st.columns(2)
            for i, q in enumerate(sugs):
                with cols[i % 2]:
                    with st.container():
                        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                        if st.button(q, key=f"sug_{i}", use_container_width=True):
                            st.session_state["_pending"] = q
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

        # Handle pending suggestion
        if "_pending" in st.session_state:
            pending_q = st.session_state.pop("_pending")
        else:
            pending_q = None

        # Chat input
        user_input = st.chat_input("Ask about media coverage, bias, sentiment…") or pending_q

        if user_input:
            st.session_state.messages.append(ChatMessage(role="user", content=user_input))
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🗞️"):
                with st.spinner("🔍 Retrieving & analyzing…"):
                    t0 = time.time()
                    response: RAGResponse = run_rag_chain(user_input, mgr)

                st.markdown(response.answer)

                if response.sources:
                    pills = " ".join(
                        f'<a href="{s.url}" target="_blank" style="text-decoration:none;'
                        f'background:#0C1828;color:#00D4FF;border:1px solid #00D4FF33;'
                        f'border-radius:20px;padding:3px 10px;font-size:11px;'
                        f'display:inline-block;margin:2px">📰 {s.source_name}</a>'
                        for s in response.sources[:6]
                    )
                    st.markdown(f'<div style="margin-top:8px">{pills}</div>',
                                unsafe_allow_html=True)

                lat = f"{response.latency_ms:.0f}ms" if response.latency_ms else "—"
                st.markdown(
                    f'<div style="font-size:10px;color:#2A4060;margin-top:6px">'
                    f'🤖 {response.model_used} &nbsp;·&nbsp; ⚡ {lat} '
                    f'&nbsp;·&nbsp; 📄 {len(response.retrieved_chunks)} chunks'
                    f'</div>', unsafe_allow_html=True)

            st.session_state.messages.append(
                ChatMessage(role="assistant", content=response.answer, rag_response=response))
            st.session_state.last_rag_response = response


# ══════════════════════════════════════════════════════════════
#  RIGHT — Analysis Dashboard
# ══════════════════════════════════════════════════════════════
with dash_col:
    st.markdown("""
    <div style="font-size:13px;font-weight:600;color:#4A7090;letter-spacing:1px;
                text-transform:uppercase;margin-bottom:14px">
      📊 Analysis Dashboard
    </div>
    """, unsafe_allow_html=True)

    last: RAGResponse | None = st.session_state.last_rag_response

    if not last:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-title">Waiting for Analysis</div>
          <div class="empty-sub">
            Ask a question in the chat to see<br>
            sentiment scores, bias detection,<br>
            and source breakdowns here.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chunks = last.retrieved_chunks

        tab1, tab2, tab3 = st.tabs(["😊 Sentiment", "⚖️ Bias", "📚 Sources"])

        with tab1:
            overall = overall_sentiment(chunks)
            per_src  = aggregate_by_source(chunks)
            c1, c2 = st.columns(2)
            with c1:
                render_sentiment_donut(overall)
            with c2:
                render_subjectivity_gauge(overall)
            if per_src:
                render_sentiment_by_source(per_src)

        with tab2:
            bias_results = analyze_sources_bias(chunks)
            if bias_results:
                render_bias_spectrum(bias_results)
                st.markdown("#### Source Bias Summary")
                for br in bias_results[:8]:
                    emoji = {"left":"◀◀","center-left":"◀","center":"⬤",
                             "center-right":"▶","right":"▶▶"}.get(br.lean,"?")
                    conf_color = "#00D4AA" if br.confidence > .7 else "#FFB347"
                    st.markdown(
                        f'<div style="background:#0C1828;border:1px solid #1A2E45;'
                        f'border-radius:10px;padding:10px 14px;margin-bottom:8px;'
                        f'display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="color:#C5D5E5;font-size:13px">'
                        f'{emoji} <strong>{br.source_name}</strong></span>'
                        f'<span style="color:{conf_color};font-size:11px;'
                        f'background:{conf_color}18;padding:2px 8px;border-radius:8px">'
                        f'{br.lean.replace("-"," ").title()} · {br.confidence:.0%}</span>'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.info("No bias data available for retrieved sources.")

        with tab3:
            bias_map = {br.source_name: br for br in analyze_sources_bias(chunks)}
            render_source_cards(last.sources, bias_map=bias_map)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Showstopper · AI News Analyst · NewsAPI + FAISS + LangChain + Streamlit
</div>
""", unsafe_allow_html=True)
