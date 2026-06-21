"""Health + service-info routes."""

from fastapi import APIRouter

from app import __version__
from app import state

router = APIRouter()


@router.get("/")
def root():
    """Service info. Visit /docs for the interactive UI."""
    return {
        "service": "CodeRAG",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "POST /ingest": "load a local path, GitHub repo, or web page",
            "POST /ingest/upload": "ingest uploaded files",
            "POST /query": "ask a question (set generate=false for retrieval-only)",
            "GET /health": "liveness + whether a store is loaded",
        },
    }


@router.get("/health")
def health():
    return {"status": "ok", "store_loaded": state.store_loaded()}
