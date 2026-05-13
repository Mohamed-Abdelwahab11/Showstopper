"""
Showstopper — RAG Chain
Orchestrates the full Retrieval-Augmented Generation pipeline.
Supports OpenAI, Groq, and Demo (no LLM) providers.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import yaml
from dotenv import load_dotenv

from models.schemas import Chunk, RAGResponse, SourceCitation
from rag.prompt_builder import build_demo_answer, build_prompt
from rag.retriever import retrieve
from vectorstore.index_manager import FAISSIndexManager

load_dotenv()
logger = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_llm_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["llm"]


def _get_provider() -> str:
    """Determine active LLM provider from env var, falling back to config."""
    env_val = os.getenv("LLM_PROVIDER", "").lower()
    if env_val in ("openai", "groq", "demo"):
        return env_val
    cfg = _load_llm_config()
    return cfg.get("provider", "demo").lower()


def _chunks_to_citations(chunks: list[Chunk]) -> list[SourceCitation]:
    """Convert retrieved chunks to deduplicated SourceCitation objects."""
    seen: set[str] = set()
    citations: list[SourceCitation] = []
    for chunk in chunks:
        url = chunk.metadata.article_url
        if url in seen:
            continue
        seen.add(url)
        citations.append(SourceCitation(
            title=chunk.metadata.article_title or "Untitled",
            url=url,
            source_name=chunk.metadata.source_name,
            published_at=chunk.metadata.published_at,
            relevance=chunk.relevance,
            snippet=chunk.text[:200] + ("…" if len(chunk.text) > 200 else ""),
        ))
    return sorted(citations, key=lambda c: c.relevance, reverse=True)


# ── OpenAI Chain ──────────────────────────────────────────────

def _run_openai(system_prompt: str, user_prompt: str, cfg: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=cfg.get("openai_model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 1200),
    )
    return response.choices[0].message.content or ""


def _stream_openai(system_prompt: str, user_prompt: str, cfg: dict) -> Generator[str, None, None]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model=cfg.get("openai_model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 1200),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Groq Chain ────────────────────────────────────────────────

def _run_groq(system_prompt: str, user_prompt: str, cfg: dict) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=cfg.get("groq_model", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 1200),
    )
    return response.choices[0].message.content or ""


def _stream_groq(system_prompt: str, user_prompt: str, cfg: dict) -> Generator[str, None, None]:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    stream = client.chat.completions.create(
        model=cfg.get("groq_model", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 1200),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Public API ────────────────────────────────────────────────

def run_rag_chain(
    query: str,
    index_manager: FAISSIndexManager,
    top_k: int | None = None,
) -> RAGResponse:
    """
    Run the complete RAG pipeline synchronously.

    1. Retrieve relevant chunks
    2. Build prompt
    3. Call LLM (or demo mode)
    4. Return structured RAGResponse

    Parameters
    ----------
    query         : User's question
    index_manager : Loaded FAISSIndexManager
    top_k         : Override for number of retrieved chunks
    """
    start = time.time()
    cfg = _load_llm_config()
    provider = _get_provider()

    # Step 1: Retrieve
    chunks = retrieve(query, index_manager, top_k=top_k)

    # Step 2: Build prompt
    system_prompt, user_prompt = build_prompt(query, chunks)

    # Step 3: Generate answer
    answer = ""
    model_used = "demo"

    try:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            answer = _run_openai(system_prompt, user_prompt, cfg)
            model_used = cfg.get("openai_model", "gpt-4o-mini")
        elif provider == "groq" and os.getenv("GROQ_API_KEY"):
            answer = _run_groq(system_prompt, user_prompt, cfg)
            model_used = cfg.get("groq_model", "llama-3.1-8b-instant")
        else:
            answer = build_demo_answer(query, chunks)
            model_used = "demo"
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = build_demo_answer(query, chunks)
        model_used = "demo (fallback)"

    latency = round((time.time() - start) * 1000, 1)

    return RAGResponse(
        query=query,
        answer=answer,
        sources=_chunks_to_citations(chunks),
        retrieved_chunks=chunks,
        timestamp=datetime.utcnow(),
        model_used=model_used,
        latency_ms=latency,
    )


def stream_rag_chain(
    query: str,
    index_manager: FAISSIndexManager,
    top_k: int | None = None,
) -> Generator[str, None, None]:
    """
    Streaming version of run_rag_chain.
    Yields token strings for real-time display in Streamlit.
    After streaming, yields a special sentinel: '__DONE__'
    """
    cfg = _load_llm_config()
    provider = _get_provider()

    chunks = retrieve(query, index_manager, top_k=top_k)
    system_prompt, user_prompt = build_prompt(query, chunks)

    try:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            yield from _stream_openai(system_prompt, user_prompt, cfg)
        elif provider == "groq" and os.getenv("GROQ_API_KEY"):
            yield from _stream_groq(system_prompt, user_prompt, cfg)
        else:
            # Demo: yield word by word
            demo_answer = build_demo_answer(query, chunks)
            for word in demo_answer.split(" "):
                yield word + " "
    except Exception as e:
        logger.error(f"Streaming LLM call failed: {e}")
        yield build_demo_answer(query, chunks)

    yield "__DONE__"
