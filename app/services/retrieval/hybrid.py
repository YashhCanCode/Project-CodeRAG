"""
retrieval/hybrid.py

Hybrid retrieval = dense (vector) + sparse (BM25), combined with
Reciprocal Rank Fusion (RRF).

Why RRF: the two retrievers produce scores on totally different scales
(cosine distance vs BM25 term weights). RRF ignores raw scores and fuses by
RANK instead, so neither signal dominates and no fragile normalization is
needed. A chunk that ranks high in *either* list bubbles up; a chunk high in
*both* wins. Formula per chunk: sum over lists of 1 / (rrf_k + rank).
"""

from typing import List, Dict, Any

from langchain_chroma import Chroma

from app.utils.paths import load_settings
from app.services.vector_store.store import similarity_search
from app.services.retrieval.bm25 import bm25_search


def _chunk_key(meta: Dict[str, Any]) -> str:
    """Stable identity for fusion/dedup across the two result lists."""
    return meta.get("chunk_id") or f"{meta.get('source','?')}:{meta.get('chunk_index','?')}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int,
    key_fn,
) -> List[Dict[str, Any]]:
    """
    ranked_lists: each is a best-first list of chunk dicts.
    Returns one fused list (best-first) with an added 'rrf_score'.
    """
    fused: Dict[str, Dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            k = key_fn(item["metadata"])
            if k not in fused:
                fused[k] = {**item, "rrf_score": 0.0}
            fused[k]["rrf_score"] += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def hybrid_search(store: Chroma, query: str) -> List[Dict[str, Any]]:
    """
    Run vector + BM25, fuse with RRF, return the top `rerank_candidates`
    chunk dicts (each carrying content, metadata, and diagnostic scores).
    If hybrid is disabled in settings, returns vector-only candidates.
    """
    s = load_settings()
    rrf_k = s["hybrid"]["rrf_k"]
    n_candidates = s["retrieval"]["rerank_candidates"]
    vector_k = s["retrieval"]["vector_search_k"]
    bm25_k = s["retrieval"]["bm25_k"]
    hybrid_on = s["hybrid"]["enabled"]

    # Dense list
    vec_raw = similarity_search(store, query, k=vector_k)  # [(Document, distance)]
    vec_list = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "distance": float(dist),
        "similarity": 1.0 - float(dist),
    } for doc, dist in vec_raw]

    if not hybrid_on:
        for r in vec_list:
            r["rrf_score"] = None
        return vec_list[:n_candidates]

    # Sparse list
    bm25_raw = bm25_search(store, query, k=bm25_k)  # [(chunk_dict, score)]
    bm25_list = [{
        "content": c["content"],
        "metadata": c["metadata"],
        "bm25_score": float(score),
    } for c, score in bm25_raw]

    fused = reciprocal_rank_fusion([vec_list, bm25_list], rrf_k=rrf_k, key_fn=_chunk_key)
    return fused[:n_candidates]
