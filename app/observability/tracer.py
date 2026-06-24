"""
Per-request tracer: times pipeline stages, records token usage + cost, writes a
JSONL trace line, and feeds the in-memory metrics store. Also emits OpenTelemetry
spans when OTel is enabled (see otel.py).

Usage:
    with RequestTrace("query", question=q) as tr:
        with tr.span("retrieval"):
            chunks = retrieve(...)
        with tr.span("generation"):
            result = generate_answer(...)
        tr.record_usage(model, result.get("usage"))
        tr.set_refused(result["refused"])
"""

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from app.utils.paths import load_settings, resolve_path
from app.observability.metrics import METRICS
from app.observability.cost import cost_for
from app.observability import otel


class RequestTrace:
    def __init__(self, name: str, **meta: Any):
        self.name = name
        self.meta = meta
        self.stages: Dict[str, float] = {}
        self.usage: Dict[str, int] = {}
        self.model: Optional[str] = None
        self.cost: float = 0.0
        self.refused: Optional[bool] = None
        self.error: bool = False
        self.retrieved: list = []
        self.prompt: Optional[str] = None
        self.answer: Optional[str] = None
        self.num_citations: int = 0
        self.duration_ms: float = 0.0
        self._t0 = 0.0
        self._root = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._root = otel.start_span(self.name)
        return self

    @contextmanager
    def span(self, stage: str):
        start = time.perf_counter()
        osp = otel.start_span(stage)
        try:
            yield
        finally:
            self.stages[stage] = round((time.perf_counter() - start) * 1000, 2)
            otel.end_span(osp)

    def record_usage(self, model: Optional[str], usage: Optional[Dict[str, int]]) -> None:
        self.model = model
        self.usage = usage or {}
        self.cost = cost_for(model or "", self.usage.get("input", 0), self.usage.get("output", 0))

    def set_refused(self, refused: bool) -> None:
        self.refused = refused

    def set_error(self, errored: bool = True) -> None:
        self.error = errored

    def set_retrieval(self, chunks) -> None:
        # compact view: which chunks were retrieved + their scores (full visibility)
        self.retrieved = [{
            "citation": c.get("citation"),
            "source": c.get("source"),
            "score": c.get("score"),
            "rerank_prob": c.get("rerank_prob"),
            "bm25_score": c.get("bm25_score"),
        } for c in chunks]

    def set_prompt(self, prompt: Optional[str]) -> None:
        self.prompt = prompt

    def set_answer(self, answer: Optional[str], citations=None) -> None:
        self.answer = answer
        self.num_citations = len(citations or [])

    def __exit__(self, *exc):
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        otel.end_span(self._root)
        trace = {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "stages": self.stages,
            "model": self.model,
            "usage": self.usage,
            "cost_usd": round(self.cost, 6),
            "refused": self.refused,
            "error": self.error,
            "num_citations": self.num_citations,
            "retrieved": self.retrieved,    # which chunks + scores
            "prompt": self.prompt,          # exact prompt sent to the LLM
            "answer": self.answer,          # final response
            **self.meta,
        }
        _write_jsonl(trace)
        METRICS.record(trace)
        return False  # never suppress exceptions


def _write_jsonl(trace: Dict[str, Any]) -> None:
    try:
        path = resolve_path(load_settings().get("observability", {}).get("trace_log", "./logs/traces.jsonl"))
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(trace) + "\n")
    except Exception:
        pass  # tracing must never break the request
