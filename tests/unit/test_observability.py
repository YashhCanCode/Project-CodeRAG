"""Offline unit tests for the observability math (no network/LLM)."""

from app.observability.metrics import _pct, MetricsStore
from app.observability.cost import cost_for


def test_percentile():
    assert _pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.50) == 5.5
    assert _pct([], 0.95) == 0.0


def test_metrics_snapshot():
    m = MetricsStore()
    for i in range(4):
        m.record({
            "duration_ms": (i + 1) * 100,
            "cost_usd": 0.001,
            "usage": {"total": 100},
            "refused": (i == 0),     # 1 of 4 refused
            "stages": {"retrieval": 10.0, "generation": (i + 1) * 50},
        })
    snap = m.snapshot()
    assert snap["requests"] == 4
    assert snap["refusal_rate"] == 0.25
    assert snap["tokens"]["total"] == 400
    assert snap["latency_ms"]["p50"] == 250.0
    assert "generation" in snap["stage_latency_ms"]


def test_cost_for():
    # gemini-2.5-flash priced 0.30/M input, 2.50/M output -> 1M each = 2.80
    assert round(cost_for("gemini-2.5-flash", 1_000_000, 1_000_000), 2) == 2.80
    assert cost_for("mistral:7b", 1_000_000, 1_000_000) == 0.0     # local/free
    assert cost_for("unknown-model", 999, 999) == 0.0              # no pricing


def test_quality_metrics_coverage_and_errors():
    m = MetricsStore()
    m.record({"duration_ms": 100, "refused": True,  "error": False, "num_citations": 0, "usage": {"total": 0}})
    m.record({"duration_ms": 100, "refused": False, "error": False, "num_citations": 2, "usage": {"total": 50}})
    m.record({"duration_ms": 100, "refused": False, "error": False, "num_citations": 3, "usage": {"total": 50}})
    m.record({"duration_ms": 100, "refused": False, "error": True,  "num_citations": 0, "usage": {"total": 0}})
    s = m.snapshot()
    # attempts = 3 non-error with refused set: 1 refused, 2 answered & grounded
    assert s["refusal_rate"] == round(1 / 3, 3)
    assert s["citation_coverage"] == round(2 / 3, 3)
    assert s["error_rate"] == 0.25
