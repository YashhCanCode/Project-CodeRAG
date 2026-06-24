"""
Optional Langfuse integration for trace UI + dashboard (latency, cost, tokens,
prompt/response per request). Open-source and self-hostable.

No-op unless `langfuse` is installed AND keys are set:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY  (+ optional LANGFUSE_HOST)

We attach Langfuse's LangChain callback handler to LLM calls, so each generation
is traced with its prompt, completion, token usage, and latency automatically —
which is exactly what powers the Langfuse dashboard.
"""

import os

_handler = None
_checked = False


def get_callback_handler():
    """Return a Langfuse LangChain CallbackHandler, or None if not configured."""
    global _handler, _checked
    if _checked:
        return _handler
    _checked = True

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None  # not configured -> stay no-op

    try:
        try:
            from langfuse.langchain import CallbackHandler      # langfuse v3
        except Exception:
            from langfuse.callback import CallbackHandler        # langfuse v2
        _handler = CallbackHandler()
        print("[langfuse] tracing enabled")
    except Exception as e:
        print(f"[langfuse] disabled ({type(e).__name__}); in-house tracing still active.")
        _handler = None
    return _handler
