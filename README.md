# 🗞️ Showstopper — AI News Analyst

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00D4AA?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-1C3C3C?style=for-the-badge)
![NewsAPI](https://img.shields.io/badge/NewsAPI-Live_News-0066CC?style=for-the-badge)

**A RAG-powered AI chatbot that reads thousands of news articles and answers:**
> *"What is the media's stance on Issue X?"*

[Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Configuration](#configuration) · [Usage](#usage)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📡 **Live News Ingestion** | Fetches articles from NewsAPI.org (`/v2/everything` + `/v2/top-headlines`) |
| 🧹 **Smart Cleaning** | Strips HTML, removes boilerplate, deduplicates by URL |
| ✂️ **Semantic Chunking** | RecursiveCharacterTextSplitter with configurable size & overlap |
| 🧮 **Free Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` — no API key needed |
| ⚡ **FAISS Vector Search** | Persistent local index with MMR diversity reranking |
| 🤖 **Multi-LLM Support** | OpenAI GPT-4o-mini · Groq Llama-3 (free) · Demo mode (no key needed) |
| 😊 **Sentiment Analysis** | VADER + TextBlob composite — per-source breakdown |
| ⚖️ **Bias Detection** | Source lean database (AllSides-inspired) + framing keyword detection |
| 💬 **Premium Chat UI** | Dark-mode Streamlit with streaming answers, source pills, suggested questions |
| 📊 **Visual Dashboard** | Plotly sentiment donut · bias spectrum · subjectivity gauge · source cards |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                 Ingestion Pipeline                   │
│  NewsAPI → Cleaner → Chunker → Embedder → FAISS     │
└─────────────────────────────────────────────────────┘
     │                              ▲
     │                              │ build index
     ▼                              │
┌──────────────┐          ┌─────────────────┐
│  retriever   │──top-k──▶│  FAISS Index    │
└──────────────┘          └─────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  prompt_builder → LLM (chain)    │
│  OpenAI / Groq / Demo            │
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  analysis/sentiment.py           │  VADER + TextBlob
│  analysis/bias_detector.py       │  Source DB + Framing Keywords
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  Streamlit UI (app/main.py)      │
│  Chat · Charts · Source Cards    │
└──────────────────────────────────┘
```

---

## 📁 Project Structure

```
Showstopper/
├── ingestion/
│   ├── news_fetcher.py       # NewsAPI wrapper + demo fallback
│   ├── cleaner.py            # HTML strip, dedup, boilerplate removal
│   └── chunker.py            # RecursiveCharacterTextSplitter
├── vectorstore/
│   ├── embedder.py           # sentence-transformers batch encoding
│   └── index_manager.py      # FAISS build / save / load / search
├── rag/
│   ├── retriever.py          # Top-k retrieval with MMR deduplication
│   ├── prompt_builder.py     # System + context + user prompt templates
│   └── chain.py              # OpenAI / Groq / Demo chain with streaming
├── analysis/
│   ├── sentiment.py          # VADER + TextBlob composite sentiment
│   └── bias_detector.py      # Source lean DB + framing keyword detection
├── app/
│   ├── main.py               # Streamlit entry point
│   └── components/
│       ├── chat.py           # Chat history renderer + streaming
│       ├── charts.py         # Plotly dark-mode visualizations
│       └── source_viewer.py  # Expandable source cards with bias badges
├── models/
│   └── schemas.py            # Pydantic v2 data models
├── tests/
│   ├── test_ingestion.py     # Cleaner + chunker tests
│   ├── test_rag.py           # FAISS index + prompt builder tests
│   └── test_analysis.py      # Sentiment + bias detector tests
├── data/
│   └── sample_articles.json  # Demo articles (works without API key)
├── config.yaml               # All tunable parameters
├── .env.example              # API key template
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Showstopper.git
cd Showstopper

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required for live news
NEWS_API_KEY=your_newsapi_org_key_here

# Choose ONE LLM provider (or use "demo" — no key needed)
LLM_PROVIDER=demo           # demo | openai | groq

OPENAI_API_KEY=...          # if LLM_PROVIDER=openai
GROQ_API_KEY=...            # if LLM_PROVIDER=groq (FREE at console.groq.com)
```

### 3. Run

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎮 Usage

### Step 1 — Fetch Articles
Enter a topic in the sidebar (e.g. `"artificial intelligence regulation"`) and click **Fetch & Index**.

The pipeline will:
1. 📡 Fetch up to 100 articles from NewsAPI
2. 🧹 Clean and deduplicate them
3. ✂️ Split into ~500-character chunks
4. 🧮 Embed with `all-MiniLM-L6-v2`
5. ⚡ Build a FAISS index

### Step 2 — Ask Questions

Example questions:
- *"What is the media's overall stance on AI regulation?"*
- *"Which outlets are most critical?"*
- *"What language do journalists use when discussing this topic?"*
- *"Summarize the key viewpoints from left and right media."*

### Step 3 — Explore the Dashboard

The **Analysis Dashboard** on the right shows:
- 😊 **Sentiment Donut** — overall positive/negative/neutral split
- 📊 **Sentiment by Source** — stacked bar per outlet
- ⚖️ **Bias Spectrum** — political lean per source
- 🎯 **Subjectivity Gauge** — objective vs opinionated coverage
- 📚 **Source Cards** — expandable cards with snippets and bias badges

---

## ⚙️ Configuration

All parameters are in `config.yaml`:

```yaml
ingestion:
  max_articles: 100      # articles per fetch
  days_back: 7           # historical window
  language: en

chunking:
  chunk_size: 500        # characters per chunk
  chunk_overlap: 60      # context overlap

retrieval:
  top_k: 8              # chunks returned per query
  use_mmr: true         # diversity reranking

llm:
  provider: demo        # openai | groq | demo
  openai_model: gpt-4o-mini
  groq_model: llama-3.1-8b-instant
  temperature: 0.3

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage:
- `test_ingestion.py` — HTML cleaning, deduplication, chunking, metadata
- `test_analysis.py` — Sentiment scores in valid ranges, bias labels
- `test_rag.py` — FAISS build/search, prompt templates

---

## 🔑 API Keys

| Key | Required | Free? | Get it |
|---|---|---|---|
| `NEWS_API_KEY` | For live news | Free tier (100 req/day) | [newsapi.org](https://newsapi.org) |
| `OPENAI_API_KEY` | If using OpenAI | No | [platform.openai.com](https://platform.openai.com) |
| `GROQ_API_KEY` | If using Groq | **Yes, FREE** | [console.groq.com](https://console.groq.com) |

> **Demo Mode**: Without any LLM key, the app shows retrieved article excerpts with sentiment and bias analysis. Set `LLM_PROVIDER=demo` in `.env`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | Streamlit |
| **News Source** | NewsAPI.org |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS (CPU, local) |
| **LLM** | OpenAI GPT-4o-mini / Groq Llama-3 / Demo |
| **Orchestration** | LangChain (text splitter) |
| **Sentiment** | VADER + TextBlob |
| **Visualization** | Plotly |
| **Validation** | Pydantic v2 |
| **Config** | PyYAML + python-dotenv |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
</div>
