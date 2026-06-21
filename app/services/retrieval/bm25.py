"""
retrieval/bm25.py

Classic keyword retrieval (BM25) over the SAME chunks stored in ChromaDB.
This captures exact terms (e.g. "thermal", "mongoose", ".env") that a dense
embedder can miss.

Chroma stays the single source of truth: we pull all chunk texts/metadata
from the store and build an in-memory BM25 index, cached per store so it's
built once per process (not per query).
"""

import re
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

# Cache: id(store) -> (BM25Okapi, [chunk_dict, ...])
_BM25_CACHE: Dict[int, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}

# split camelCase boundaries: "AuthCallback" -> "Auth Callback"
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> List[str]:
    """
    Code-aware tokenizer: lowercases, splits identifiers on camelCase and
    non-alphanumerics, so `create_agent` -> [create, agent] and
    `AuthCallbackPage` -> [auth, callback, page]. This makes keyword search
    match the way developers think about identifiers.
    """
    text = _CAMEL.sub(" ", text)
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]


def _chunks_from_store(store: Chroma) -> List[Dict[str, Any]]:
    """Pull every chunk (text + metadata) out of the Chroma collection."""
    raw = store.get(include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    chunks = []
    for content, meta in zip(docs, metas):
        meta = meta or {}
        chunks.append({"content": content, "metadata": meta})
    return chunks


def build_bm25(store: Chroma) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """Build (or reuse cached) BM25 index over the store's chunks."""
    key = id(store)
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]

    chunks = _chunks_from_store(store)
    corpus = [tokenize(c["content"]) for c in chunks]
    # BM25Okapi needs a non-empty corpus; guard against an empty collection.
    bm25 = BM25Okapi(corpus) if corpus else None
    _BM25_CACHE[key] = (bm25, chunks)
    print(f"[bm25] Indexed {len(chunks)} chunks")
    return bm25, chunks


def bm25_search(store: Chroma, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
    """Return top-k (chunk_dict, bm25_score) for the query, best-first."""
    bm25, chunks = build_bm25(store)
    if not bm25 or not chunks:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]
