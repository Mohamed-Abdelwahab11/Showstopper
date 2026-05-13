"""
Showstopper — Media Bias Detector
Detects potential media bias using:
1. Known source lean from the AllSides / Ad Fontes media bias ratings
2. Loaded/charged keyword detection in article text
"""

from __future__ import annotations

import logging
import re

from models.schemas import BiasResult, Chunk

logger = logging.getLogger(__name__)

# ── Source Bias Database (AllSides / Ad Fontes inspired) ───────
# Format: source_name_lower → (lean, credibility_tier)
_SOURCE_BIAS_DB: dict[str, tuple[str, str]] = {
    # Left
    "the guardian": ("left", "high"),
    "mother jones": ("left", "medium"),
    "the nation": ("left", "medium"),
    "huffpost": ("left", "medium"),
    "vox": ("center-left", "high"),
    "msnbc": ("left", "medium"),
    "slate": ("left", "medium"),
    "new republic": ("left", "medium"),
    "buzzfeed news": ("center-left", "medium"),
    # Center-Left
    "new york times": ("center-left", "high"),
    "washington post": ("center-left", "high"),
    "the atlantic": ("center-left", "high"),
    "politico": ("center-left", "high"),
    "nprnews": ("center-left", "high"),
    "npr": ("center-left", "high"),
    "time": ("center-left", "high"),
    "associated press": ("center", "high"),
    "ap": ("center", "high"),
    "reuters": ("center", "high"),
    "bloomberg": ("center", "high"),
    # Center
    "bbc news": ("center", "high"),
    "bbc": ("center", "high"),
    "the economist": ("center", "high"),
    "axios": ("center", "high"),
    "the hill": ("center", "high"),
    "pbs": ("center", "high"),
    "csmonitor": ("center", "high"),
    # Center-Right
    "wall street journal": ("center-right", "high"),
    "wsj": ("center-right", "high"),
    "national review": ("center-right", "medium"),
    "the dispatch": ("center-right", "high"),
    "the federalist": ("right", "medium"),
    # Right
    "fox news": ("right", "medium"),
    "new york post": ("right", "medium"),
    "breitbart": ("right", "low"),
    "daily wire": ("right", "medium"),
    "newsmax": ("right", "low"),
    "oann": ("right", "low"),
    "daily mail": ("center-right", "medium"),
}

# ── Framing Keywords (loaded language detection) ───────────────
_LEFT_FRAMING = [
    r"\bsystemic racism\b", r"\bclimate justice\b", r"\bincome inequality\b",
    r"\bsocial justice\b", r"\bequity\b", r"\bmarginalized\b",
    r"\bprivilege\b", r"\bintersectionality\b", r"\boppressive\b",
]

_RIGHT_FRAMING = [
    r"\bread tape\b", r"\bsocialist\b", r"\belite media\b",
    r"\bopen borders\b", r"\bwoke\b", r"\bcanceled\b",
    r"\bdeep state\b", r"\bfake news\b", r"\banti-american\b",
]

_HIGH_EMOTION = [
    r"\bshocking\b", r"\boutrageous\b", r"\bunbelievable\b",
    r"\binfuriating\b", r"\bscandal\b", r"\bcatastrophic\b",
    r"\bexplodes?\b", r"\bslams?\b", r"\bdestroy\b",
    r"\bcrushing\b", r"\bblasts?\b",
]


def _compile_patterns(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_LEFT_RE = _compile_patterns(_LEFT_FRAMING)
_RIGHT_RE = _compile_patterns(_RIGHT_FRAMING)
_EMOTION_RE = _compile_patterns(_HIGH_EMOTION)


def _detect_source_bias(source_name: str) -> tuple[str, str, float]:
    """
    Look up source lean and credibility from the database.
    Returns (lean, credibility_tier, confidence).
    """
    key = source_name.strip().lower()

    # Exact match
    if key in _SOURCE_BIAS_DB:
        lean, tier = _SOURCE_BIAS_DB[key]
        return lean, tier, 0.85

    # Partial match
    for db_key, (lean, tier) in _SOURCE_BIAS_DB.items():
        if db_key in key or key in db_key:
            return lean, tier, 0.65

    return "unknown", "unknown", 0.0


def _detect_framing_keywords(text: str) -> tuple[list[str], str]:
    """
    Detect loaded/charged language in text.
    Returns (flagged_phrases, detected_lean).
    """
    flagged: list[str] = []
    left_hits = _LEFT_RE.findall(text)
    right_hits = _RIGHT_RE.findall(text)
    emotion_hits = _EMOTION_RE.findall(text)

    flagged.extend([h.strip() for h in emotion_hits[:5]])

    if len(left_hits) > len(right_hits):
        flagged.extend([h.strip() for h in left_hits[:3]])
        lean = "center-left" if len(left_hits) < 3 else "left"
    elif len(right_hits) > len(left_hits):
        flagged.extend([h.strip() for h in right_hits[:3]])
        lean = "center-right" if len(right_hits) < 3 else "right"
    else:
        lean = "center"

    return list(set(flagged))[:8], lean


def analyze_chunk_bias(chunk: Chunk) -> BiasResult:
    """Analyze a single chunk for source bias and framing language."""
    source_lean, credibility, source_confidence = _detect_source_bias(
        chunk.metadata.source_name
    )
    flagged_phrases, text_lean = _detect_framing_keywords(chunk.text)

    # Combine source lean and text lean
    if source_confidence >= 0.65:
        final_lean = source_lean
        confidence = source_confidence
    elif flagged_phrases:
        final_lean = text_lean
        confidence = 0.40
    else:
        final_lean = "unknown"
        confidence = 0.0

    return BiasResult(
        lean=final_lean,
        confidence=round(confidence, 2),
        flagged_phrases=flagged_phrases,
        source_name=chunk.metadata.source_name,
        credibility_tier=credibility,
    )


def analyze_sources_bias(chunks: list[Chunk]) -> list[BiasResult]:
    """
    Aggregate bias analysis per unique source across all retrieved chunks.
    Returns one BiasResult per unique source, sorted by confidence.
    """
    seen: dict[str, BiasResult] = {}
    for chunk in chunks:
        source = chunk.metadata.source_name
        if source not in seen:
            seen[source] = analyze_chunk_bias(chunk)

    results = list(seen.values())
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


def bias_spectrum_summary(bias_results: list[BiasResult]) -> dict:
    """
    Summarize the overall political lean distribution across sources.
    Returns a dict: {lean_label: count}.
    """
    counts: dict[str, int] = {}
    for r in bias_results:
        if r.lean != "unknown":
            counts[r.lean] = counts.get(r.lean, 0) + 1
    return counts
