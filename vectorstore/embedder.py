"""
Showstopper — Embedder
Converts text chunks into dense vector embeddings using
sentence-transformers/all-MiniLM-L6-v2.
Fully local — no API key required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# ── Singleton model cache ─────────────────────────────────────
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, device: str = "cpu") -> SentenceTransformer:
    """Load and cache the embedding model (singleton per model name)."""
    key = f"{model_name}:{device}"
    if key not in _model_cache:
        logger.info(f"Loading embedding model: {model_name} on {device}…")
        _model_cache[key] = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model loaded.")
    return _model_cache[key]


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.

    Parameters
    ----------
    texts        : List of raw text strings to embed
    model_name   : Sentence-transformers model (default from config.yaml)
    batch_size   : Encoding batch size (default from config.yaml)
    device       : "cpu" or "cuda" (default from config.yaml)
    show_progress: Show tqdm progress bar

    Returns
    -------
    numpy array of shape (len(texts), embedding_dim)
    """
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["embedding"]

    mn = model_name or cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    bs = batch_size or cfg.get("batch_size", 64)
    dev = device or cfg.get("device", "cpu")

    model = _get_model(mn, dev)

    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    logger.info(f"Embedding {len(texts)} texts (batch_size={bs})…")
    embeddings = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
    )
    logger.info(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def embed_query(
    query: str,
    model_name: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    Embed a single query string into a 1D vector.
    Returns shape (embedding_dim,).
    """
    result = embed_texts([query], model_name=model_name, device=device, show_progress=False)
    return result[0]
