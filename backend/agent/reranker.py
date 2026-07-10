"""
Priority 2: Cross-encoder Re-ranking
=======================================
Architecture decision: two-stage retrieval.

Stage 1 (rag.py): bi-encoder similarity search
  → Fast, retrieves top-20 candidate chunks
  → Uses dot product of pre-computed embeddings
  → Weakness: independent encodings miss query-chunk interaction

Stage 2 (this file): cross-encoder re-ranking
  → Slow but precise — scores each (query, chunk) pair together
  → The cross-encoder reads BOTH query and chunk at once, so it
    understands context like "when the query says 'fast' it means
    inference speed, not car speed"
  → Takes top-20 candidates, outputs top-5 most relevant

Why this matters in interviews:
  "I use a bi-encoder for recall (get all possibly relevant chunks)
   and a cross-encoder for precision (pick the best ones). This is
   the same architecture used in Google's search ranking."

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22MB download, runs on CPU in ~50ms per batch
  - Trained on MS MARCO (Microsoft's 500k QA dataset)
  - Industry standard for passage re-ranking

Degradation: if sentence-transformers isn't installed or model
  fails to load, falls back to original cosine-similarity order.
"""

import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

_reranker = None
_reranker_available = None  # None = untested, True/False = tested


def _load_reranker():
    """Lazy load the cross-encoder model. Caches result."""
    global _reranker, _reranker_available
    if _reranker_available is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder

        model_name = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Loading cross-encoder: {model_name}")
        _reranker = CrossEncoder(model_name, max_length=512)
        _reranker_available = True
        logger.info("Cross-encoder loaded successfully")
    except Exception as exc:
        logger.warning(
            f"Cross-encoder unavailable ({exc}). "
            "Install: pip install sentence-transformers. "
            "Falling back to cosine-similarity ordering."
        )
        _reranker_available = False
        _reranker = None

    return _reranker


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Re-rank retrieved chunks using a cross-encoder.

    Args:
        query: The research question
        chunks: List of chunk dicts from rag.retrieve_relevant_chunks()
        top_k: How many chunks to keep after re-ranking

    Returns:
        Top-k chunks sorted by cross-encoder relevance score (descending).
        Falls back to original order if cross-encoder unavailable.
    """
    if not chunks:
        return []

    reranker = _load_reranker()

    if reranker is None:
        # Graceful fallback: return top_k chunks in original order
        logger.debug("Using fallback ordering (no cross-encoder)")
        return chunks[:top_k]

    try:
        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs — cross-encoder returns logits (higher = more relevant)
        scores = reranker.predict(pairs)

        # Attach scores to chunks
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(score)
            # Keep original cosine similarity for comparison/display
            chunk_copy["cosine_similarity"] = chunk.get("relevance", 0.0)
            scored_chunks.append(chunk_copy)

        # Boost score by source credibility
        from .credibility import score_url

        for chunk in scored_chunks:
            cred = score_url(chunk.get("url", "")) / 100.0  # normalize 0-1
            # Weighted combination: 70% cross-encoder, 30% credibility
            chunk["final_score"] = (chunk["rerank_score"] * 0.70) + (cred * 0.30)

        # Sort by final combined score
        scored_chunks.sort(key=lambda x: x["final_score"], reverse=True)

        # Log improvement for observability
        original_top = chunks[0].get("title", "?") if chunks else "?"
        reranked_top = scored_chunks[0].get("title", "?") if scored_chunks else "?"
        if original_top != reranked_top:
            logger.info(f"Re-ranking changed top result: '{original_top}' → '{reranked_top}'")

        return scored_chunks[:top_k]

    except Exception as exc:
        logger.error(f"Re-ranking failed: {exc}. Using fallback ordering.")
        return chunks[:top_k]


def is_reranker_available() -> bool:
    """Check if re-ranking is available without loading the model."""
    if _reranker_available is None:
        _load_reranker()
    return bool(_reranker_available)
