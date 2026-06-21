"""
paths.py

Central path + settings resolution so every module works regardless of the
current working directory (fixes the old `open("config/settings.yaml")`
assumption that you must launch from the project root).
"""

from pathlib import Path
import yaml

# Project root = repo root (this file is app/utils/paths.py -> parents[2]).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SETTINGS_PATH = PROJECT_ROOT / "app" / "config" / "settings.yaml"
PROMPTS_PATH = PROJECT_ROOT / "prompts" / "templates" / "prompts.yaml"


def load_settings() -> dict:
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def load_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)
