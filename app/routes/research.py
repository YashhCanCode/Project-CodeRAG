"""Research-landscape route: topic -> arXiv -> rerank -> extract -> synthesize."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.agents.research_landscape import build_landscape

router = APIRouter()


class ResearchRequest(BaseModel):
    topic: str
    max_papers: int = 8          # papers kept after reranking


@router.post("/research")
def research(req: ResearchRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(400, "topic is required")

    top_k = max(3, min(req.max_papers, 12))   # keep quota + latency reasonable
    try:
        result = build_landscape(topic, max_candidates=max(top_k * 2, 15), top_k=top_k)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower():
            raise HTTPException(429, "LLM rate limit / quota exhausted — try fewer papers or later.")
        raise HTTPException(502, f"Research failed: {msg[:200]}")

    if not result["papers"]:
        raise HTTPException(404, result.get("note") or f"No arXiv papers found for: {topic}")
    return result
