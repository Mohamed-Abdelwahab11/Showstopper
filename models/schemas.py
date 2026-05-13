"""
Showstopper — AI News Analyst
Pydantic v2 schemas for all data models across the pipeline.
"""

from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator


# ─────────────────────────────────────────────────────────────
#  Ingestion Layer
# ─────────────────────────────────────────────────────────────

class ArticleSource(BaseModel):
    """NewsAPI source object."""
    id: Optional[str] = None
    name: str = "Unknown"


class Article(BaseModel):
    """A single news article from NewsAPI or local demo data."""
    url: str
    title: str = ""
    description: Optional[str] = None
    content: Optional[str] = None
    source: ArticleSource = Field(default_factory=ArticleSource)
    author: Optional[str] = None
    published_at: Optional[str] = None   # ISO 8601 string
    query_topic: str = ""                # topic that triggered this fetch

    @field_validator("title", "description", "content", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @property
    def full_text(self) -> str:
        """Combine all text fields into one string for indexing."""
        parts = filter(None, [self.title, self.description, self.content])
        return " ".join(parts)

    @property
    def source_name(self) -> str:
        return self.source.name or "Unknown"


# ─────────────────────────────────────────────────────────────
#  Vector Store Layer
# ─────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk stored in FAISS."""
    article_url: str = ""
    article_title: str = ""
    source_name: str = "Unknown"
    published_at: str = ""
    query_topic: str = ""
    chunk_index: int = 0


class Chunk(BaseModel):
    """A text chunk with its associated metadata."""
    text: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    faiss_id: Optional[int] = None
    score: Optional[float] = None   # L2 distance (lower = more similar)

    @property
    def relevance(self) -> float:
        """Convert L2 distance to a [0, 1] relevance score (higher = better)."""
        if self.score is None:
            return 0.0
        return round(1.0 / (1.0 + self.score), 4)


# ─────────────────────────────────────────────────────────────
#  Analysis Layer
# ─────────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    """VADER + TextBlob composite sentiment analysis."""
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    compound: float = Field(ge=-1.0, le=1.0)      # VADER compound [-1, 1]
    subjectivity: float = Field(ge=0.0, le=1.0)   # TextBlob [0=objective, 1=subjective]
    label: Literal["positive", "negative", "neutral"] = "neutral"
    source_name: str = "Unknown"


class BiasResult(BaseModel):
    """Media bias / framing analysis result."""
    lean: Literal["left", "center-left", "center", "center-right", "right", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    flagged_phrases: list[str] = Field(default_factory=list)
    source_name: str = "Unknown"
    credibility_tier: Literal["high", "medium", "low", "unknown"] = "unknown"


# ─────────────────────────────────────────────────────────────
#  RAG Layer
# ─────────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A single source cited in the RAG answer."""
    title: str
    url: str
    source_name: str
    published_at: str = ""
    relevance: float = 0.0
    snippet: str = ""


class RAGResponse(BaseModel):
    """Full response object returned by the RAG chain."""
    query: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    sentiment: Optional[SentimentResult] = None
    bias_results: list[BiasResult] = Field(default_factory=list)
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = "demo"
    latency_ms: Optional[float] = None


class ChatMessage(BaseModel):
    """A single message in the chat history."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rag_response: Optional[RAGResponse] = None
