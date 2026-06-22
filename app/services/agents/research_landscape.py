"""
app/services/agents/research_landscape.py

Research-landscape agent: turn a topic into a structured map of the literature.

Pipeline:
  topic
    -> arXiv search            (candidate papers, abstracts only — fast & free)
    -> cross-encoder rerank    (semantic relevance to the topic; reuses retrieval.reranker)
    -> structured extraction   (1 LLM call: problem/method/results/contribution per paper)
    -> cross-paper synthesis   (1 LLM call: clusters, open problems, tensions)

Two LLM calls total per topic (batched), to stay within free-tier limits.
Everything is grounded in the retrieved abstracts.
"""

import json
import re
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.utils.paths import load_prompts, load_settings
from app.services.ingestion.arxiv_loader import search_arxiv
from app.services.retrieval.reranker import rerank
from app.services.generation.chain import _build_llm


def _parse_json(text: str):
    """Tolerant JSON extraction from an LLM response (handles ``` fences / prose)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except Exception:
        # fall back to the first {...} or [...] block
        for open_c, close_c in (("[", "]"), ("{", "}")):
            i, j = t.find(open_c), t.rfind(close_c)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    pass
    return None


def _rerank_papers(topic: str, papers: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Order papers by semantic relevance to the topic, attaching a relevance score."""
    if not papers:
        return []
    candidates = [{"content": f"{p['title']}. {p['abstract']}", "metadata": p} for p in papers]
    ranked = rerank(topic, candidates, top_k)          # graceful: returns top_k if no model
    out = []
    for c in ranked:
        pp = c["metadata"]
        pp["relevance"] = c.get("rerank_prob")         # None when no reranker model is loaded
        out.append(pp)
    return out


def _llm_invoke(system: str, user: str) -> str:
    llm = _build_llm()
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


def _extract(papers: List[Dict[str, Any]]) -> None:
    """One batched LLM call: attach problem/method/results/contribution to each paper."""
    p = load_prompts()["research"]
    listing = "\n\n".join(
        f"[{i+1}] {pp['title']}\nAbstract: {pp['abstract']}" for i, pp in enumerate(papers)
    )
    raw = _llm_invoke(p["extract_system"], p["extract_user"].format(papers=listing))
    data = _parse_json(raw) or []
    for i, pp in enumerate(papers):
        item = data[i] if i < len(data) and isinstance(data[i], dict) else {}
        pp["problem"] = item.get("problem", "")
        pp["method"] = item.get("method", "")
        pp["results"] = item.get("results", "")
        pp["contribution"] = item.get("contribution", "")


def _synthesize(topic: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One LLM call: clusters, open problems, tensions across the papers."""
    p = load_prompts()["research"]
    listing = "\n".join(
        f"{pp['id']} — {pp['title']} — {pp.get('problem','')} {pp.get('method','')} "
        f"{pp.get('contribution','')}" for pp in papers
    )
    raw = _llm_invoke(p["synthesize_system"], p["synthesize_user"].format(topic=topic, papers=listing))
    data = _parse_json(raw) or {}
    return {
        "overview": data.get("overview", ""),
        "clusters": data.get("clusters", []),
        "open_problems": data.get("open_problems", []),
        "tensions": data.get("tensions", []),
    }


def _assign_clusters(papers: List[Dict[str, Any]], synthesis: Dict[str, Any]) -> None:
    """Tag each paper with the cluster theme it belongs to (for grouped display)."""
    id_to_theme = {}
    for cl in synthesis.get("clusters", []):
        for pid in cl.get("paper_ids", []):
            id_to_theme[pid] = cl.get("theme", "")
    for pp in papers:
        pp["cluster"] = id_to_theme.get(pp["id"], "Other")


def build_landscape(topic: str, max_candidates: int = 20, top_k: int = 8) -> Dict[str, Any]:
    """Run the full pipeline and return {topic, papers, synthesis, note?}."""
    candidates = search_arxiv(topic, max_results=max_candidates)
    papers = _rerank_papers(topic, candidates, top_k)
    if not papers:
        return {"topic": topic, "papers": [], "synthesis": {},
                "note": f"No arXiv papers found for '{topic}'."}

    # Relevance gate: refuse to synthesize from irrelevant papers (only when the
    # reranker produced scores; otherwise we can't judge relevance).
    min_rel = load_settings().get("research", {}).get("min_relevance", 0.10)
    scored = [p for p in papers if p.get("relevance") is not None]
    if scored:
        relevant = [p for p in scored if p["relevance"] >= min_rel]
        if not relevant:
            best = max(p["relevance"] for p in scored)
            return {"topic": topic, "papers": [], "synthesis": {},
                    "note": (f"No sufficiently relevant arXiv papers for '{topic}' "
                             f"(best match {best:.0%}). Try a research area like "
                             f"'retrieval augmented generation' rather than a tool name or question.")}
        papers = relevant

    _extract(papers)
    synthesis = _synthesize(topic, papers)
    _assign_clusters(papers, synthesis)
    return {"topic": topic, "papers": papers, "synthesis": synthesis}
