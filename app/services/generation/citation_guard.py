"""
generation/citation_guard.py

Citation enforcement: an architectural guarantee (not a prompt plea) that the
model only answers when retrieval actually found supporting evidence. This is
the main anti-hallucination lever.

The guard inspects the retrieved chunks and decides:
  - which chunks are trustworthy enough to cite, and
  - whether there's enough of them to answer at all.

When reranking is on, it gates on the cross-encoder probability (a real
relevance signal). When rerank is off, it falls back to cosine similarity.
"""

from typing import List, Dict, Any, Tuple

from app.utils.paths import load_settings


def filter_trusted(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns (trusted_chunks, gate_kind).
    gate_kind is "rerank" or "similarity" depending on which signal was used.
    """
    s = load_settings()["citation"]

    # Prefer the reranker probability when it's present on the chunks.
    use_rerank = any(c.get("rerank_prob") is not None for c in chunks)
    if use_rerank:
        thresh = s["min_rerank_prob"]
        trusted = [c for c in chunks if (c.get("rerank_prob") or 0.0) >= thresh]
        return trusted, "rerank"

    thresh = s["min_similarity"]
    trusted = [c for c in chunks if (c.get("similarity") or c.get("score") or 0.0) >= thresh]
    return trusted, "similarity"


def has_sufficient_evidence(chunks: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    True if at least `min_evidence` trusted chunks exist.
    Returns (ok, trusted_chunks).
    """
    s = load_settings()["citation"]
    trusted, _ = filter_trusted(chunks)
    return (len(trusted) >= s["min_evidence"]), trusted
