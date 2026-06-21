"""
retrieval/vector_store.py

ChromaDB wrapper with two modes:
  - build(): ingest chunks and persist to disk (idempotent)
  - load(): load an existing store for querying

Distance metric is COSINE (hnsw:space=cosine). Chroma's
similarity_search_with_score therefore returns a cosine DISTANCE
(0 = identical, larger = less similar). The pipeline converts this to a
similarity (higher = better) before any thresholding. Do NOT treat the raw
score as a similarity.
"""

from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.utils.paths import load_settings, resolve_path
from app.services.embeddings.embedder import get_embedder


def _config():
    s = load_settings()
    return (
        resolve_path(s["vector_store"]["persist_directory"]),
        s["vector_store"]["collection_name"],
    )


def build_vector_store(chunks: List[Document], persist_dir: str = None,
                       collection: str = None) -> Chroma:
    """
    Embed all chunks and persist to ChromaDB.

    Idempotent: the collection is reset first and chunks are written with
    stable ids (chunk_id), so re-ingesting the same codebase replaces rather
    than duplicates vectors.

    persist_dir/collection override the configured defaults — used by the
    evaluation harness to build an isolated store without clobbering the
    user's main collection.
    """
    cfg_dir, cfg_col = _config()
    persist_dir = persist_dir or cfg_dir
    collection = collection or cfg_col
    embedder = get_embedder()

    # Reset the collection for a clean rebuild (avoids stale/duplicate vectors).
    existing = Chroma(
        collection_name=collection,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )
    try:
        existing.delete_collection()
    except Exception:
        pass

    ids = [
        c.metadata.get("chunk_id") or f"{c.metadata.get('source','unknown')}:{i}"
        for i, c in enumerate(chunks)
    ]

    print(f"[vector_store] Embedding {len(chunks)} chunks -> {persist_dir}/{collection} (cosine)")

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        ids=ids,
        collection_name=collection,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"[vector_store] Done. Collection has {store._collection.count()} vectors.")
    return store


def load_vector_store() -> Chroma:
    """Load an existing persisted ChromaDB collection for querying."""
    persist_dir, collection = _config()
    embedder = get_embedder()

    store = Chroma(
        collection_name=collection,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )

    count = store._collection.count()
    print(f"[vector_store] Loaded collection '{collection}' with {count} vectors")
    return store


def similarity_search(store: Chroma, query: str, k: int = None) -> List[Tuple[Document, float]]:
    """
    Returns top-k (document, cosine_distance) pairs, sorted best-first.
    NOTE: the float is a DISTANCE (lower = better). Convert in the pipeline.
    """
    settings = load_settings()
    k = k or settings["retrieval"]["vector_search_k"]
    return store.similarity_search_with_score(query, k=k)
