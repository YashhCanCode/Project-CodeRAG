"""Observability route: aggregated request metrics (latency p50/p95, cost, tokens)."""

from fastapi import APIRouter
from app.observability.metrics import METRICS

router = APIRouter()


@router.get("/metrics")
def metrics():
    """Aggregated metrics over recent requests (in-memory ring buffer)."""
    return METRICS.snapshot()
