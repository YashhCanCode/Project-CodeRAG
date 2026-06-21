"""
retrieval/embedder.py

Wraps BAAI/bge-small-en-v1.5 as a LangChain-compatible embedder.
Runs locally via sentence-transformers — no API key needed.

Device is auto-detected (cuda > mps > cpu) so the same code runs on an
Apple-Silicon Mac, an NVIDIA box, or a CPU-only CI runner. (The old code
hardcoded "mps", which raises on non-Mac machines.)

BGE models need a query prefix for retrieval. HuggingFaceBgeEmbeddings adds
the default query instruction automatically; documents are embedded without
a prefix, which is correct.
"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from app.utils.paths import load_settings


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_embedder() -> HuggingFaceBgeEmbeddings:
    """
    Returns a BGE embedder. Downloads the model on first call (~130MB), cached after.
    normalize_embeddings=True so cosine similarity is well-behaved.
    """
    settings = load_settings()
    model_name = settings["embedding"]["model"]
    device = _detect_device()

    print(f"[embedder] Loading {model_name} on '{device}' "
          f"(downloads on first run, cached after)")

    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
