"""
Showstopper — Sentiment Analyzer
Composite sentiment analysis using VADER (primary) + TextBlob (subjectivity).
VADER is specifically tuned for social media / news text and handles
negations, punctuation, and capitalization natively.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models.schemas import Chunk, SentimentResult

logger = logging.getLogger(__name__)

# ── Singleton analyzer ────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()


def _label_from_compound(compound: float) -> str:
    """Convert VADER compound score to text label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_text(text: str, source_name: str = "Unknown") -> SentimentResult:
    """
    Analyze sentiment of a single text string.

    Returns a SentimentResult with:
    - pos/neg/neu  : VADER proportional scores [0, 1] summing to 1
    - compound     : VADER aggregate score [-1, 1]
    - subjectivity : TextBlob subjectivity [0=objective, 1=subjective]
    - label        : "positive" | "negative" | "neutral"
    """
    vader_scores = _vader.polarity_scores(text)

    try:
        blob = TextBlob(text)
        subjectivity = float(blob.sentiment.subjectivity)
    except Exception:
        subjectivity = 0.5

    compound = float(vader_scores["compound"])

    return SentimentResult(
        positive=round(float(vader_scores["pos"]), 4),
        negative=round(float(vader_scores["neg"]), 4),
        neutral=round(float(vader_scores["neu"]), 4),
        compound=round(compound, 4),
        subjectivity=round(subjectivity, 4),
        label=_label_from_compound(compound),
        source_name=source_name,
    )


def analyze_chunks(chunks: list[Chunk]) -> list[SentimentResult]:
    """
    Analyze sentiment for each chunk individually.
    Returns one SentimentResult per chunk.
    """
    results = []
    for chunk in chunks:
        result = analyze_text(chunk.text, source_name=chunk.metadata.source_name)
        results.append(result)
    return results


def aggregate_by_source(chunks: list[Chunk]) -> list[SentimentResult]:
    """
    Aggregate sentiment per news source across all retrieved chunks.
    Returns one averaged SentimentResult per unique source.
    """
    # Group chunks by source
    source_chunks: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        source_chunks[chunk.metadata.source_name].append(chunk)

    aggregated: list[SentimentResult] = []

    for source_name, src_chunks in source_chunks.items():
        individual = [analyze_text(c.text, source_name=source_name) for c in src_chunks]

        n = len(individual)
        avg = SentimentResult(
            positive=round(sum(r.positive for r in individual) / n, 4),
            negative=round(sum(r.negative for r in individual) / n, 4),
            neutral=round(sum(r.neutral for r in individual) / n, 4),
            compound=round(sum(r.compound for r in individual) / n, 4),
            subjectivity=round(sum(r.subjectivity for r in individual) / n, 4),
            label=_label_from_compound(sum(r.compound for r in individual) / n),
            source_name=source_name,
        )
        aggregated.append(avg)

    # Sort by compound score descending (most positive first)
    aggregated.sort(key=lambda r: r.compound, reverse=True)
    return aggregated


def overall_sentiment(chunks: list[Chunk]) -> SentimentResult:
    """
    Compute overall sentiment across ALL retrieved chunks combined.
    """
    if not chunks:
        return SentimentResult(
            positive=0.0, negative=0.0, neutral=1.0,
            compound=0.0, subjectivity=0.0,
            label="neutral", source_name="Overall",
        )

    combined_text = " ".join(c.text for c in chunks)
    result = analyze_text(combined_text, source_name="Overall")
    return result
