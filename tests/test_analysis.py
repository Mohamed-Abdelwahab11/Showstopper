"""
Tests for NLP analysis: sentiment and bias detector.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.schemas import Article, ArticleSource, Chunk, ChunkMetadata
from analysis.sentiment import analyze_text, analyze_chunks, overall_sentiment, aggregate_by_source
from analysis.bias_detector import analyze_chunk_bias, analyze_sources_bias, _detect_framing_keywords


# ── Fixtures ──────────────────────────────────────────────────

def make_chunk(text: str, source: str = "Test Source", url: str = "https://example.com") -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            article_url=url,
            article_title="Test Article",
            source_name=source,
            published_at="2025-01-01",
            query_topic="test",
        ),
    )


# ── Sentiment Tests ───────────────────────────────────────────

class TestSentiment:

    def test_positive_text(self):
        result = analyze_text("This is a wonderful, amazing, outstanding success!")
        assert result.label == "positive"
        assert result.compound > 0.05

    def test_negative_text(self):
        result = analyze_text("This is a terrible disaster. Everything is failing badly.")
        assert result.label == "negative"
        assert result.compound < -0.05

    def test_neutral_text(self):
        result = analyze_text("The company released its quarterly earnings report today.")
        assert result.label == "neutral"

    def test_scores_sum_to_one(self):
        result = analyze_text("The government announced a new policy on climate change.")
        total = result.positive + result.negative + result.neutral
        assert abs(total - 1.0) < 0.01  # sum ≈ 1

    def test_compound_in_range(self):
        result = analyze_text("Some sample text for testing.")
        assert -1.0 <= result.compound <= 1.0

    def test_subjectivity_in_range(self):
        result = analyze_text("The sky is blue and the grass is green.")
        assert 0.0 <= result.subjectivity <= 1.0

    def test_analyze_chunks_returns_per_chunk(self):
        chunks = [make_chunk("Good news today!"), make_chunk("Bad disaster here.")]
        results = analyze_chunks(chunks)
        assert len(results) == 2

    def test_overall_sentiment_empty(self):
        result = overall_sentiment([])
        assert result.label == "neutral"
        assert result.compound == 0.0

    def test_aggregate_by_source(self):
        chunks = [
            make_chunk("Great success!", source="BBC"),
            make_chunk("Terrible failure!", source="CNN"),
            make_chunk("Another good result.", source="BBC"),
        ]
        results = aggregate_by_source(chunks)
        sources = {r.source_name for r in results}
        assert "BBC" in sources
        assert "CNN" in sources
        assert len(results) == 2  # one per unique source


# ── Bias Tests ────────────────────────────────────────────────

class TestBias:

    def test_known_source_left(self):
        chunk = make_chunk("Article text here.", source="The Guardian")
        result = analyze_chunk_bias(chunk)
        assert result.lean in ("left", "center-left")
        assert result.confidence > 0.5

    def test_known_source_center(self):
        chunk = make_chunk("Article text here.", source="Reuters")
        result = analyze_chunk_bias(chunk)
        assert result.lean == "center"

    def test_known_source_right(self):
        chunk = make_chunk("Article text here.", source="Fox News")
        result = analyze_chunk_bias(chunk)
        assert result.lean == "right"

    def test_unknown_source(self):
        chunk = make_chunk("Article text here.", source="Local Village Gazette")
        result = analyze_chunk_bias(chunk)
        # Should not crash, may return "unknown"
        assert result.lean is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_framing_keywords_right(self):
        text = "The woke mob is pushing its deep state socialist agenda again."
        phrases, lean = _detect_framing_keywords(text)
        assert lean in ("right", "center-right")

    def test_analyze_sources_bias_deduplication(self):
        chunks = [
            make_chunk("text", source="Reuters"),
            make_chunk("more text", source="Reuters"),
            make_chunk("other text", source="Fox News"),
        ]
        results = analyze_sources_bias(chunks)
        sources = [r.source_name for r in results]
        assert len(sources) == len(set(sources))  # no duplicates

    def test_bias_result_fields_valid(self):
        chunk = make_chunk("Article text.", source="BBC News")
        result = analyze_chunk_bias(chunk)
        assert result.lean in ("left", "center-left", "center", "center-right", "right", "unknown")
        assert result.credibility_tier in ("high", "medium", "low", "unknown")
        assert isinstance(result.flagged_phrases, list)
