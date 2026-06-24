"""In-memory metrics aggregator: latency percentiles, cost, tokens, refusal rate."""

import threading
from collections import deque
from typing import Dict, Any, List


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


class MetricsStore:
    """Ring buffer of recent request traces with on-demand aggregation."""

    def __init__(self, maxlen: int = 2000):
        self._traces = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, trace: Dict[str, Any]) -> None:
        with self._lock:
            self._traces.append(trace)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            traces = list(self._traces)
        if not traces:
            return {"requests": 0}

        latencies = [t["duration_ms"] for t in traces if "duration_ms" in t]
        costs = [t.get("cost_usd", 0.0) for t in traces]
        totals = [(t.get("usage") or {}).get("total", 0) for t in traces]
        # generation was attempted (refused set) and didn't hard-error
        errors = [t for t in traces if t.get("error")]
        attempts = [t for t in traces if t.get("refused") is not None and not t.get("error")]
        refused = [t for t in attempts if t.get("refused")]
        answered = [t for t in attempts if not t.get("refused")]
        grounded = [t for t in answered if (t.get("num_citations") or 0) > 0]

        # per-stage latency percentiles
        stages: Dict[str, List[float]] = {}
        for t in traces:
            for name, ms in (t.get("stages") or {}).items():
                stages.setdefault(name, []).append(ms)
        stage_latency = {
            name: {"p50_ms": _pct(v, 0.50), "p95_ms": _pct(v, 0.95)}
            for name, v in stages.items()
        }

        return {
            "requests": len(traces),
            "latency_ms": {
                "p50": _pct(latencies, 0.50),
                "p95": _pct(latencies, 0.95),
                "max": round(max(latencies), 2) if latencies else 0.0,
            },
            "stage_latency_ms": stage_latency,
            "tokens": {
                "total": sum(totals),
                "avg_per_request": round(sum(totals) / len(traces), 1),
            },
            "cost_usd": {
                "total": round(sum(costs), 6),
                "avg_per_request": round(sum(costs) / len(traces), 6),
            },
            "refusal_rate": round(len(refused) / len(attempts), 3) if attempts else 0.0,
            "citation_coverage": round(len(grounded) / len(attempts), 3) if attempts else 0.0,
            "error_rate": round(len(errors) / len(traces), 3) if traces else 0.0,
        }


# process-wide singleton
METRICS = MetricsStore()
