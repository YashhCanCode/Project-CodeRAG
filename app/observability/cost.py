"""Cost estimation from token usage, priced via settings.observability.pricing."""

from app.utils.paths import load_settings


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for one LLM call. Returns 0.0 if the model has no pricing entry."""
    pricing = load_settings().get("observability", {}).get("pricing", {})
    p = pricing.get(model)
    if not p:
        return 0.0
    return (input_tokens / 1_000_000) * p.get("input", 0.0) + \
           (output_tokens / 1_000_000) * p.get("output", 0.0)
