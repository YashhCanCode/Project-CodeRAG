"""
app/services/ingestion/arxiv_loader.py

Query the public arXiv API and return paper metadata + abstracts.

The arXiv API returns Atom XML; we parse it with the stdlib (no extra deps).
Abstracts are enough for the research-landscape feature, so no PDF download
is needed — fast and free.

API docs: https://info.arxiv.org/help/api/user-manual.html
"""

import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

_last_call = 0.0


def _polite_delay():
    """arXiv asks for ~3s between requests; be a good citizen."""
    global _last_call
    wait = 3.0 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _arxiv_id(entry_id: str) -> str:
    # entry_id looks like http://arxiv.org/abs/2005.11401v1
    return entry_id.rstrip("/").split("/abs/")[-1]


def search_arxiv(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search arXiv by topic. Returns papers (best-first by relevance):
      {id, title, authors, abstract, url, pdf_url, published, categories}
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    _polite_delay()
    resp = requests.get(url, headers={"User-Agent": "CodeRAG/0.3 research-landscape"}, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers: List[Dict[str, Any]] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{_ATOM}summary") or "").strip().replace("\n", " ")
        eid = (entry.findtext(f"{_ATOM}id") or "").strip()
        published = (entry.findtext(f"{_ATOM}published") or "")[:10]
        authors = [a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")]
        cats = [c.get("term") for c in entry.findall(f"{_ATOM}category") if c.get("term")]

        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")

        if not title:
            continue
        papers.append({
            "id": _arxiv_id(eid),
            "title": title,
            "authors": [a for a in authors if a],
            "abstract": summary,
            "url": eid,
            "pdf_url": pdf_url,
            "published": published,
            "categories": cats,
        })
    return papers
