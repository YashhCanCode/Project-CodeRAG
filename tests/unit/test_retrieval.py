"""Offline unit tests for retrieval building blocks (no network/models)."""

from app.services.retrieval.hybrid import reciprocal_rank_fusion
from app.services.retrieval.bm25 import tokenize


def test_code_aware_tokenizer_splits_identifiers():
    assert tokenize("create_agent") == ["create", "agent"]
    assert tokenize("AuthCallbackPage") == ["auth", "callback", "page"]


def test_rrf_surfaces_a_vector_missed_chunk():
    key = lambda m: m["chunk_id"]
    chunk = lambda cid: {"metadata": {"chunk_id": cid}}
    vector = [chunk("A"), chunk("B"), chunk("C")]   # vector missed D
    bm25 = [chunk("D"), chunk("A")]                 # BM25 ranks D first
    fused = reciprocal_rank_fusion([vector, bm25], rrf_k=60, key_fn=key)
    ids = [f["metadata"]["chunk_id"] for f in fused]
    assert ids[0] == "A"        # strong in both -> wins
    assert "D" in ids           # vector-missed chunk still surfaces
