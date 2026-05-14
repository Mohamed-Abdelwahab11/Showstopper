"""
Showstopper — Article Chunker
Splits cleaned articles into fixed-size overlapping chunks using
LangChain's RecursiveCharacterTextSplitter. Preserves article metadata
in every chunk for traceability back to the source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from models.schemas import Article, Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_chunk_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["chunking"]


def chunk_article(
    article: Article,
    chunk_size: int = 500,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    """
    Split a single article's full text into overlapping chunks.

    The full text is built from: title + description + content.
    Each chunk carries the article's metadata so it can be traced back.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    full_text = article.full_text
    if not full_text.strip():
        return []

    raw_chunks = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        if len(text.strip()) < 30:
            continue  # skip slivers
        chunks.append(Chunk(
            text=text.strip(),
            metadata=ChunkMetadata(
                article_url=article.url,
                article_title=article.title,
                source_name=article.source_name,
                published_at=article.published_at or "",
                query_topic=article.query_topic,
                chunk_index=idx,
            ),
        ))

    return chunks


def chunk_articles(
    articles: list[Article],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Chunk a list of cleaned articles using config.yaml defaults.

    Parameters
    ----------
    articles      : List of cleaned Article objects
    chunk_size    : Override config chunk_size
    chunk_overlap : Override config chunk_overlap

    Returns
    -------
    List of Chunk objects ready for embedding
    """
    cfg = _load_chunk_config()
    size = chunk_size or cfg.get("chunk_size", 500)
    overlap = chunk_overlap or cfg.get("chunk_overlap", 60)

    all_chunks: list[Chunk] = []
    for article in articles:
        chunks = chunk_article(article, chunk_size=size, chunk_overlap=overlap)
        all_chunks.extend(chunks)
        logger.debug(f"  '{article.title[:50]}' → {len(chunks)} chunks")

    logger.info(f"Total chunks produced: {len(all_chunks)} from {len(articles)} articles.")
    return all_chunks
