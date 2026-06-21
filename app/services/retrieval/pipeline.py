"""
retrieval/pipeline.py

Phase 2 retrieval: hybrid (BM25 + vector) -> RRF fusion -> cross-encoder rerank.
Returns the top-k chunks with citation metadata as plain dicts. The interface
is unchanged from Phase 1 so generation doesn't care how retrieval works.

Flow:
  hybrid_search()  -> ~20 fused candidates (dense + sparse)
  rerank()         -> precise re-scoring, keep top_k
Both hybrid and rerank are individually toggleable in settings.yaml and
degrade gracefully (rerank off/unavailable -> fused order; hybrid off ->
vector-only).
"""

from typing import List, Dict, Any

from langchain_chroma import Chroma

from app.utils.paths import load_settings
from app.services.retrieval.hybrid import hybrid_search
from app.services.retrieval.reranker import rerank


def _primary_score(c: Dict[str, Any]) -> float:
    """Single 'higher = better' score for the gate, whatever stage produced it."""
    if c.get("rerank_prob") is not None:
        return c["rerank_prob"]
    if c.get("similarity") is not None:
        return c["similarity"]
    return c.get("rrf_score") or 0.0


def retrieve(store: Chroma, query: str) -> List[Dict[str, Any]]:
    """
    Return top-k relevant chunks (best-first) as dicts:
      {
        content, source, citation, language, repo,
        score,            # primary ranking score, higher = better
        rerank_prob,      # cross-encoder relevance 0-1 (None if not reranked)
        rerank_score,     # raw cross-encoder logit (None if not reranked)
        similarity,       # cosine similarity (None if chunk only came from BM25)
        bm25_score,       # BM25 weight (None if only from vector)
        rrf_score,        # fusion score (None if hybrid disabled)
      }
    """
    settings = load_settings()
    top_k = settings["retrieval"]["top_k"]

    candidates = hybrid_search(store, query)
    ranked = rerank(query, candidates, top_k)

    results = []
    for c in ranked:
        meta = c["metadata"]
        results.append({
            "content":      c["content"],
            "source":       meta.get("source", "unknown"),
            "citation":     meta.get("citation", ""),
            "language":     meta.get("language", ""),
            "repo":         meta.get("repo", ""),
            "score":        _primary_score(c),
            "rerank_prob":  c.get("rerank_prob"),
            "rerank_score": c.get("rerank_score"),
            "similarity":   c.get("similarity"),
            "bm25_score":   c.get("bm25_score"),
            "rrf_score":    c.get("rrf_score"),
        })
    return results
