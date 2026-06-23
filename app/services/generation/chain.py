"""
generation/chain.py

Generation: formats retrieved chunks into a prompt, calls Gemini,
returns answer + citations.

Citation enforcement (see generation/citation_guard.py) runs BEFORE the LLM:
if retrieval didn't surface enough trustworthy evidence, we refuse without
calling the model. This is an architectural guarantee, not a prompt plea.
"""

import os
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.utils.paths import load_prompts as _load_prompts_file, load_settings
from app.services.generation.citation_guard import has_sufficient_evidence


def _load_prompts(version: str = "v1") -> Dict[str, str]:
    return _load_prompts_file()[version]


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks for the prompt.
    Each chunk is prefixed with its citation so the model can reference it.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Chunk {i} [{chunk['citation']}] ---\n"
            f"{chunk['content']}\n"
        )
    return "\n".join(parts)


def _build_llm():
    """Build the generation LLM from config. Pluggable: gemini, groq, or ollama."""
    gen = load_settings()["generation"]
    provider = gen.get("provider", "gemini")
    model, temp = gen["model"], gen["temperature"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
        return ChatGroq(model=model, api_key=key, temperature=temp)

    if provider == "ollama":
        # Local, free, no API key. Requires Ollama running + `ollama pull <model>`.
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temp, base_url=base_url)

    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temp)


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate an answer from retrieved chunks.

    Returns:
      {
        "answer":    str,
        "citations": List[str],
        "refused":   bool,     # True if citation enforcement triggered
        "chunks_used": int,
      }
    """
    # ── Citation enforcement gate (see generation/citation_guard.py) ───────
    ok, trusted = has_sufficient_evidence(chunks)

    if not ok:
        return {
            "answer": "Insufficient evidence in codebase. No retrieved chunks met the confidence threshold.",
            "citations": [],
            "refused": True,
            "chunks_used": 0,
            "usage": {"input": 0, "output": 0, "total": 0},
            "model": None,
        }

    # ── Build prompt ───────────────────────────────────────────────────────
    prompts = _load_prompts()
    context_str = _format_context(trusted)

    system_msg = SystemMessage(content=prompts["rag_system"])
    human_msg  = HumanMessage(content=prompts["rag_user"].format(
        context=context_str,
        question=query,
    ))

    # ── Call LLM ──────────────────────────────────────────────────────────
    llm = _build_llm()
    response = llm.invoke([system_msg, human_msg])

    um = getattr(response, "usage_metadata", None) or {}
    usage = {
        "input":  um.get("input_tokens", 0),
        "output": um.get("output_tokens", 0),
        "total":  um.get("total_tokens", 0),
    }
    citations = [c["citation"] for c in trusted]

    return {
        "answer":      response.content,
        "citations":   citations,
        "refused":     False,
        "chunks_used": len(trusted),
        "usage":       usage,
        "model":       load_settings()["generation"]["model"],
    }