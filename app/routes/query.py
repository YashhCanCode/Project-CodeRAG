"""Query route: retrieve relevant chunks and (optionally) generate a cited answer."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.retrieval.pipeline import retrieve
from app.services.generation.chain import generate_answer
from app.services.vector_store.store import load_vector_store
from app.observability.tracer import RequestTrace
from app import state

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    generate: bool = True                  # set False for retrieval-only (no LLM/key)


def _retrieved_view(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "citation":    c.get("citation"),
        "source":      c.get("source"),
        "language":    c.get("language"),
        "score":       c.get("score"),
        "rerank_prob": c.get("rerank_prob"),
        "similarity":  c.get("similarity"),
        "bm25_score":  c.get("bm25_score"),
        "preview":     (c.get("content") or "")[:240],
        "content":     c.get("content") or "",
    } for c in chunks]


def _get_store():
    store = state.get_store()
    if store is None:
        try:
            store = load_vector_store()
            state.set_store(store)
        except Exception:
            raise HTTPException(400, "No codebase ingested yet. Call POST /ingest first.")
    return store


@router.post("/query")
def query(req: QueryRequest):
    store = _get_store()

    with RequestTrace("query", question=req.question[:200]) as tr:
        with tr.span("retrieval"):
            chunks = retrieve(store, req.question)
        tr.set_retrieval(chunks)

        response: Dict[str, Any] = {
            "question":  req.question,
            "retrieved": _retrieved_view(chunks),
        }

        if req.generate:
            try:
                with tr.span("generation"):
                    result = generate_answer(req.question, chunks)
                tr.record_usage(result.get("model"), result.get("usage"))
                tr.set_refused(result["refused"])
                tr.set_prompt(result.get("prompt"))
                tr.set_answer(result.get("answer"), result.get("citations"))
                response.update({
                    "answer":      result["answer"],
                    "citations":   result["citations"],
                    "refused":     result["refused"],
                    "chunks_used": result["chunks_used"],
                })
            except Exception as e:
                msg = str(e)
                note = "rate limit / quota" if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else "generation error"
                tr.set_refused(False)
                tr.set_error(True)
                response.update({
                    "answer":      None,
                    "citations":   [],
                    "refused":     False,
                    "chunks_used": 0,
                    "answer_error": f"Answer unavailable ({note}).",
                })

    return response
