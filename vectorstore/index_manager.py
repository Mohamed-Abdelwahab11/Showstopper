"""
Showstopper — FAISS Index Manager
Builds, saves, loads, and searches a FAISS flat L2 index.
All chunk metadata is stored in a companion pickle file.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import yaml

from models.schemas import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _get_paths() -> tuple[Path, Path]:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["vectorstore"]
    base = Path(__file__).parent.parent
    idx_path = base / cfg.get("index_path", "data/faiss_index")
    meta_path = base / cfg.get("metadata_path", "data/faiss_metadata.pkl")
    return idx_path, meta_path


class FAISSIndexManager:
    """
    Manages the FAISS vector index and its associated chunk metadata.

    Usage
    -----
    mgr = FAISSIndexManager()
    mgr.build(chunks, embeddings)
    mgr.save()
    mgr.search(query_embedding, top_k=8)
    """

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []
        self._dim: int = 0
        idx_path, meta_path = _get_paths()
        self._index_dir: Path = idx_path
        self._meta_file: Path = meta_path

    # ── Build ────────────────────────────────────────────────

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Build a new FAISS index from a list of chunks and their embeddings.
        Replaces any existing in-memory index.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        self._dim = embeddings.shape[1]
        # IndexFlatIP works with L2-normalized embeddings (equivalent to cosine similarity)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)

        self._metadata = [
            {
                "faiss_id": i,
                "text": c.text,
                "article_url": c.metadata.article_url,
                "article_title": c.metadata.article_title,
                "source_name": c.metadata.source_name,
                "published_at": c.metadata.published_at,
                "query_topic": c.metadata.query_topic,
                "chunk_index": c.metadata.chunk_index,
            }
            for i, c in enumerate(chunks)
        ]

        logger.info(f"FAISS index built: {self._index.ntotal} vectors (dim={self._dim}).")

    # ── Persist ──────────────────────────────────────────────

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save. Call build() first.")

        self._index_dir.mkdir(parents=True, exist_ok=True)
        index_file = self._index_dir / "index.faiss"

        faiss.write_index(self._index, str(index_file))
        with open(self._meta_file, "wb") as f:
            pickle.dump(self._metadata, f)

        logger.info(f"Index saved → {index_file}")
        logger.info(f"Metadata saved → {self._meta_file}")

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True on success."""
        index_file = self._index_dir / "index.faiss"

        if not index_file.exists() or not self._meta_file.exists():
            logger.warning("No saved index found. Run build() first.")
            return False

        self._index = faiss.read_index(str(index_file))
        with open(self._meta_file, "rb") as f:
            self._metadata = pickle.load(f)

        self._dim = self._index.d
        logger.info(f"Index loaded: {self._index.ntotal} vectors (dim={self._dim}).")
        return True

    # ── Search ───────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 8) -> list[Chunk]:
        """
        Search the FAISS index for the nearest chunks to a query embedding.

        Parameters
        ----------
        query_embedding : 1D numpy array (embedding_dim,)
        top_k           : Number of results to return

        Returns
        -------
        List of Chunk objects sorted by relevance (highest first)
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, top_k)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            m = self._metadata[idx]
            chunk = Chunk(
                text=m["text"],
                metadata=ChunkMetadata(
                    article_url=m["article_url"],
                    article_title=m["article_title"],
                    source_name=m["source_name"],
                    published_at=m["published_at"],
                    query_topic=m["query_topic"],
                    chunk_index=m["chunk_index"],
                ),
                faiss_id=int(idx),
                score=float(score),
            )
            results.append(chunk)

        # Sort by score descending (higher inner product = more similar for normalized vectors)
        results.sort(key=lambda c: c.score or 0.0, reverse=True)
        return results

    # ── Utilities ─────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def num_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def clear(self) -> None:
        """Reset the in-memory index and metadata."""
        self._index = None
        self._metadata = []
        self._dim = 0
        logger.info("Index cleared from memory.")


# ── Module-level singleton ─────────────────────────────────────
_manager: FAISSIndexManager | None = None


def get_index_manager() -> FAISSIndexManager:
    """Return the module-level singleton FAISSIndexManager."""
    global _manager
    if _manager is None:
        _manager = FAISSIndexManager()
    return _manager
