"""
Showstopper — Prompt Builder
Constructs the system + user prompt for the RAG chain,
injecting retrieved chunks as numbered context blocks.
"""

from __future__ import annotations

from models.schemas import Chunk

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Showstopper, an expert AI media analyst. \
Your role is to analyze news coverage and answer questions about how \
different media outlets report on specific topics.

You have access to a curated set of news article excerpts retrieved \
specifically for the user's question. Use ONLY this provided context \
to form your answer — do not rely on general knowledge from your training.

Guidelines:
- Synthesize perspectives from multiple sources when available
- Note when sources agree or disagree
- Point out any apparent bias, framing differences, or emotional language
- Always cite the source name and article title when referencing a claim
- If the context is insufficient to answer the question, say so clearly
- Be objective and analytical — avoid personal opinions
- Write in clear, professional English"""

# ── Context Template ──────────────────────────────────────────
CONTEXT_HEADER = """
---
RETRIEVED NEWS CONTEXT ({n_chunks} excerpts from {n_sources} sources):
---
"""

CHUNK_TEMPLATE = """[{idx}] SOURCE: {source} | DATE: {date} | TITLE: {title}
EXCERPT: {text}
URL: {url}
"""

# ── User Prompt Template ──────────────────────────────────────
USER_PROMPT_TEMPLATE = """{context}

---
USER QUESTION: {question}

Please provide a thorough analysis based strictly on the news excerpts above. \
Cite sources by their name and article title in your response."""


def build_prompt(question: str, chunks: list[Chunk]) -> tuple[str, str]:
    """
    Build a (system_prompt, user_prompt) tuple for the LLM.

    Parameters
    ----------
    question : User's natural language question
    chunks   : Retrieved Chunk objects from FAISS

    Returns
    -------
    (system_prompt, user_prompt) — both as plain strings
    """
    if not chunks:
        context_text = "No relevant news articles were found for this query."
    else:
        n_sources = len(set(c.metadata.source_name for c in chunks))
        context_header = CONTEXT_HEADER.format(
            n_chunks=len(chunks),
            n_sources=n_sources,
        )

        chunk_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            block = CHUNK_TEMPLATE.format(
                idx=i,
                source=chunk.metadata.source_name or "Unknown",
                date=chunk.metadata.published_at[:10] if chunk.metadata.published_at else "N/A",
                title=chunk.metadata.article_title or "Untitled",
                text=chunk.text,
                url=chunk.metadata.article_url or "N/A",
            )
            chunk_blocks.append(block)

        context_text = context_header + "\n".join(chunk_blocks)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question,
    )

    return SYSTEM_PROMPT, user_prompt


def build_demo_answer(question: str, chunks: list[Chunk]) -> str:
    """
    Generate a structured demo answer without calling any LLM.
    Used when LLM_PROVIDER=demo or no LLM key is configured.
    """
    if not chunks:
        return (
            "⚠️ **No articles found** for this query. "
            "Try fetching articles first using the sidebar, "
            "or adjust your search topic."
        )

    sources = list({c.metadata.source_name for c in chunks if c.metadata.source_name})
    titles = list({c.metadata.article_title for c in chunks if c.metadata.article_title})

    lines = [
        f"## 📰 Media Coverage Analysis: *{question}*\n",
        f"> **Demo Mode** — No LLM API key configured. Showing retrieved excerpts only.\n",
        f"Found **{len(chunks)} relevant excerpts** from **{len(sources)} sources**:\n",
    ]

    for source in sources[:5]:
        lines.append(f"- {source}")

    lines.append("\n### Top Retrieved Excerpts\n")
    for i, chunk in enumerate(chunks[:4], start=1):
        lines.append(
            f"**[{i}] {chunk.metadata.article_title or 'Untitled'}** "
            f"— *{chunk.metadata.source_name}*\n"
        )
        lines.append(f"> {chunk.text[:300]}{'…' if len(chunk.text) > 300 else ''}\n")

    lines.append(
        "\n---\n*To enable full AI analysis, add an OpenAI or Groq API key "
        "to your `.env` file and set `LLM_PROVIDER=openai` or `LLM_PROVIDER=groq`.*"
    )

    return "\n".join(lines)
