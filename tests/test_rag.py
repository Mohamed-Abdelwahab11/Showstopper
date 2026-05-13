"""
Tests for RAG pipeline: retriever and prompt builder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from models.schemas import Chunk, ChunkMetadata
from rag.prompt_builder import build_prompt, build_demo_answer
from vectorstore.index_manager import FAISSIndexManager


# ── Fixtures ──────────────────────────────────────────────────

def make_chunk(text: str, source: str = "BBC", url: str = "https://example.com") -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            article_url=url,
            article_title="Test Article",
            source_name=source,
            published_at="2025-01-15",
            query_topic="AI regulation",
        ),
    )


def build_test_index(n_chunks: int = 20, dim: int = 384) -> tuple[FAISSIndexManager, list[Chunk]]:
    """Build a small FAISS index for testing."""
    chunks = [make_chunk(f"Test chunk number {i} about AI regulation.", url=f"https://example.com/{i}") for i in range(n_chunks)]
    embeddings = np.random.rand(n_chunks, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    mgr = FAISSIndexManager()
    mgr.build(chunks, embeddings)
    return mgr, chunks


# ── Index Manager Tests ───────────────────────────────────────

class TestFAISSIndexManager:

    def test_build_and_search(self):
        mgr, chunks = build_test_index(n_chunks=20)
        query = np.random.rand(384).astype(np.float32)
        query /= np.linalg.norm(query)

        results = mgr.search(query, top_k=5)
        assert len(results) == 5

    def test_search_returns_chunks_with_scores(self):
        mgr, _ = build_test_index(n_chunks=10)
        query = np.random.rand(384).astype(np.float32)
        query /= np.linalg.norm(query)

        results = mgr.search(query, top_k=3)
        for chunk in results:
            assert chunk.score is not None
            assert chunk.text

    def test_is_ready_after_build(self):
        mgr, _ = build_test_index(n_chunks=5)
        assert mgr.is_ready is True

    def test_is_not_ready_before_build(self):
        mgr = FAISSIndexManager()
        assert mgr.is_ready is False

    def test_num_vectors(self):
        mgr, _ = build_test_index(n_chunks=15)
        assert mgr.num_vectors == 15

    def test_clear(self):
        mgr, _ = build_test_index(n_chunks=5)
        mgr.clear()
        assert mgr.is_ready is False
        assert mgr.num_vectors == 0

    def test_mismatch_raises(self):
        mgr = FAISSIndexManager()
        chunks = [make_chunk("text") for _ in range(5)]
        embeddings = np.random.rand(3, 384).astype(np.float32)  # wrong count
        with pytest.raises(ValueError):
            mgr.build(chunks, embeddings)

    def test_relevance_score_in_range(self):
        mgr, _ = build_test_index(n_chunks=10)
        query = np.random.rand(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = mgr.search(query, top_k=5)
        for chunk in results:
            assert 0.0 <= chunk.relevance <= 1.0


# ── Prompt Builder Tests ──────────────────────────────────────

class TestPromptBuilder:

    def test_returns_two_strings(self):
        chunks = [make_chunk("Some news text.")]
        sys_p, user_p = build_prompt("What is the media stance?", chunks)
        assert isinstance(sys_p, str)
        assert isinstance(user_p, str)
        assert len(sys_p) > 0
        assert len(user_p) > 0

    def test_user_prompt_contains_question(self):
        chunks = [make_chunk("Content here.")]
        _, user_p = build_prompt("What is AI regulation?", chunks)
        assert "What is AI regulation?" in user_p

    def test_user_prompt_contains_source_name(self):
        chunks = [make_chunk("Content.", source="Reuters")]
        _, user_p = build_prompt("Question?", chunks)
        assert "Reuters" in user_p

    def test_empty_chunks_handled(self):
        sys_p, user_p = build_prompt("What is happening?", [])
        assert "No relevant" in user_p or len(user_p) > 0

    def test_system_prompt_contains_analyst_persona(self):
        chunks = [make_chunk("text")]
        sys_p, _ = build_prompt("question", chunks)
        assert "analyst" in sys_p.lower() or "showstopper" in sys_p.lower()

    def test_demo_answer_non_empty(self):
        chunks = [make_chunk("News content here.")]
        answer = build_demo_answer("What is the stance?", chunks)
        assert len(answer) > 0
        assert "Demo Mode" in answer or "retrieved" in answer.lower()

    def test_demo_answer_empty_chunks(self):
        answer = build_demo_answer("Question?", [])
        assert "No articles" in answer or "fetch" in answer.lower()
