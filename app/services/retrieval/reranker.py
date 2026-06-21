"""
retrieval/reranker.py

Cross-encoder reranking for precision.

A bi-encoder (the BGE embedder) scores query and chunk *separately*, which is
fast but approximate. A cross-encoder feeds (query, chunk) together through the
model and outputs one relevance score — far more accurate, but too slow to run
over the whole corpus. So we use it only to re-score the ~20 fused candidates,
which reliably pushes the truly-relevant chunk to the top.

Model: BAAI/bge-reranker-base. Loaded lazily and cached. If it can't load
(offline / no download), rerank() returns the candidates unchanged so the
pipeline degrades gracefully to fused order.
"""

import math
from typing import List, Dict, Any, Optional

from app.utils.paths import load_settings

_RERANKER = None          # cached CrossEncoder
_LOAD_FAILED = False      # don't retry a failed load every call


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def get_reranker():
    """Lazily load and cache the CrossEncoder. Returns None if unavailable."""
    global _RERANKER, _LOAD_FAILED
    if _RERANKER is not None or _LOAD_FAILED:
        return _RERANKER

    s = load_settings()
    model_name = s["reranker"]["model"]
    try:
        from sentence_transformers import CrossEncoder
        # device auto-detected by sentence-transformers
        print(f"[reranker] Loading {model_name} (downloads on first run, cached after)")
        _RERANKER = CrossEncoder(model_name)
    except Exception as e:
        print(f"  [reranker] Could not load {model_name} ({type(e).__name__}); "
              f"falling back to fused order.")
        _LOAD_FAILED = True
        _RERANKER = None
    return _RERANKER


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Re-score candidates with the cross-encoder and return the top_k, best-first.
    Adds 'rerank_score' (raw logit) and 'rerank_prob' (sigmoid, 0-1) to each.
    If the model is unavailable, returns the first top_k unchanged.
    """
    if not candidates:
        return []

    s = load_settings()
    if not s["reranker"]["enabled"]:
        return candidates[:top_k]

    model = get_reranker()
    if model is None:
        return candidates[:top_k]

    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
        c["rerank_prob"] = _sigmoid(float(score))

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
