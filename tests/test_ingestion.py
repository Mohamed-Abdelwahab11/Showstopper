"""
Tests for ingestion pipeline: cleaner and chunker.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.schemas import Article, ArticleSource
from ingestion.cleaner import clean_article, clean_articles
from ingestion.chunker import chunk_article, chunk_articles


# ── Fixtures ──────────────────────────────────────────────────

def make_article(
    url="https://example.com/news/1",
    title="Test Article About AI",
    description="A description of the article.",
    content="This is the full content of the article. " * 20,
    source_name="Test Source",
    query_topic="artificial intelligence",
) -> Article:
    return Article(
        url=url,
        title=title,
        description=description,
        content=content,
        source=ArticleSource(name=source_name),
        query_topic=query_topic,
    )


# ── Cleaner Tests ─────────────────────────────────────────────

class TestCleaner:

    def test_strips_html(self):
        article = make_article(
            content="<p>Hello <strong>World</strong></p><br/>More text here for content."
        )
        result = clean_article(article)
        assert result is not None
        assert "<p>" not in (result.content or "")
        assert "<strong>" not in (result.content or "")
        assert "Hello" in (result.content or "") or "Hello" in result.full_text

    def test_removes_newsapi_truncation(self):
        article = make_article(
            content="This is important news. The story continues. [+1234 chars]"
        )
        result = clean_article(article)
        assert result is not None
        assert "[+" not in result.full_text

    def test_filters_short_articles(self):
        article = make_article(title="Hi", description="", content="Too short.")
        result = clean_article(article)
        assert result is None  # below MIN_CONTENT_LENGTH

    def test_deduplication(self):
        articles = [make_article(url="https://example.com/1")] * 5
        cleaned = clean_articles(articles)
        assert len(cleaned) == 1

    def test_dedup_different_urls(self):
        articles = [
            make_article(url=f"https://example.com/{i}")
            for i in range(5)
        ]
        cleaned = clean_articles(articles)
        assert len(cleaned) == 5

    def test_removed_articles_skipped(self):
        article = make_article(title="[Removed]", content="", description="")
        result = clean_article(article)
        assert result is None


# ── Chunker Tests ─────────────────────────────────────────────

class TestChunker:

    def test_produces_chunks(self):
        article = make_article()
        chunks = chunk_article(article, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 0

    def test_chunk_metadata_preserved(self):
        article = make_article(url="https://example.com/a", source_name="BBC")
        chunks = chunk_article(article)
        for chunk in chunks:
            assert chunk.metadata.article_url == "https://example.com/a"
            assert chunk.metadata.source_name == "BBC"

    def test_chunk_size_respected(self):
        article = make_article()
        chunks = chunk_article(article, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            # Allow some tolerance for splitter behavior
            assert len(chunk.text) <= 250

    def test_empty_article_yields_no_chunks(self):
        article = make_article(title="", description="", content="")
        chunks = chunk_article(article)
        assert len(chunks) == 0

    def test_chunk_list(self):
        articles = [make_article(url=f"https://example.com/{i}") for i in range(3)]
        chunks = chunk_articles(articles)
        assert len(chunks) > 3  # should be multiple chunks per article

    def test_chunk_index_sequential(self):
        article = make_article()
        chunks = chunk_article(article, chunk_size=100, chunk_overlap=10)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))
