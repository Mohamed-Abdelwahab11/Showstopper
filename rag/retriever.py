"""
Showstopper — RAG Retriever
Embeds a user query and retrieves the top-k most relevant chunks
from the FAISS index, with optional MMR reranking.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from models.schemas import Chunk
from vectorstore.embedder import embed_query
from vectorstore.index_manager import FAISSIndexManager

logger = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_retrieval_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["retrieval"]


def _mmr_rerank(
    query_vec: np.ndarray,
    chunks: list[Chunk],
    chunk_embeddings: list[np.ndarray],
    top_k: int,
    lambda_val: float = 0.6,
) -> list[Chunk]:
    """
    Maximal Marginal Relevance reranking.
    Balances relevance to query vs diversity among selected chunks.

    lambda_val=1.0 → pure relevance (no MMR)
    lambda_val=0.0 → pure diversity
    """
    if not chunks:
        return []

    selected: list[int] = []
    remaining = list(range(len(chunks)))

    for _ in range(min(top_k, len(chunks))):
        if not remaining:
            break

        # Relevance score for each remaining chunk
        rel_scores = np.array([
            float(np.dot(query_vec, chunk_embeddings[i]))
            for i in remaining
        ])

        if not selected:
            # First pick: highest relevance
            best_local = int(np.argmax(rel_scores))
        else:
            # MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
            selected_vecs = np.array([chunk_embeddings[s] for s in selected])
            sim_to_selected = np.array([
                float(np.max(np.dot(selected_vecs, chunk_embeddings[i])))
                for i in remaining
            ])
            mmr_scores = lambda_val * rel_scores - (1 - lambda_val) * sim_to_selected
            best_local = int(np.argmax(mmr_scores))

        selected.append(remaining[best_local])
        remaining.pop(best_local)

    return [chunks[i] for i in selected]


def retrieve(
    query: str,
    index_manager: FAISSIndexManager,
    top_k: int | None = None,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
) -> list[Chunk]:
    """
    Retrieve top-k relevant chunks for a query.

    Parameters
    ----------
    query         : User's natural language question
    index_manager : Loaded FAISSIndexManager instance
    top_k         : Number of chunks to return (default from config.yaml)
    use_mmr       : Apply MMR reranking (default from config.yaml)
    mmr_lambda    : MMR diversity weight (default from config.yaml)

    Returns
    -------
    List of Chunk objects, sorted by relevance
    """
    cfg = _load_retrieval_config()
    k = top_k or cfg.get("top_k", 8)
    mmr = use_mmr if use_mmr is not None else cfg.get("use_mmr", True)
    lam = mmr_lambda or cfg.get("mmr_lambda", 0.6)

    if not index_manager.is_ready:
        logger.warning("Index is not ready. Cannot retrieve.")
        return []

    # Embed query
    query_vec = embed_query(query)

    if mmr:
        # Fetch more candidates for MMR to work with
        candidates = index_manager.search(query_vec, top_k=min(k * 3, index_manager.num_vectors))
        if not candidates:
            return []

        # Re-embed candidates for MMR diversity calculation
        # (use their stored score as proxy since we already have normalized embeddings)
        # Simplified: use score as relevance proxy instead of re-embedding
        # For full MMR we'd need to store embeddings, so we do a score-based dedup instead
        seen_urls: set[str] = set()
        deduped: list[Chunk] = []
        for chunk in candidates:
            url = chunk.metadata.article_url
            if url not in seen_urls:
                seen_urls.add(url)
                deduped.append(chunk)
            if len(deduped) >= k:
                break

        logger.info(f"Retrieved {len(deduped)} chunks (MMR dedup) for query='{query[:60]}'")
        return deduped
    else:
        chunks = index_manager.search(query_vec, top_k=k)
        logger.info(f"Retrieved {len(chunks)} chunks for query='{query[:60]}'")
        return chunks
