"""
app/state.py

Process-wide handle to the active vector store, shared across route modules.
"""

_store = None


def set_store(store) -> None:
    global _store
    _store = store


def get_store():
    return _store


def store_loaded() -> bool:
    return _store is not None
