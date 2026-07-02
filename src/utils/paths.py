
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "SPLITS_DIR",
    "TOKENIZED_DIR",
    "RESULTS_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "CONFIGS_DIR",
    "find_root",
    "ensure_dir",
]

_ROOT_MARKER = "pyproject.toml"

def find_root(start: Path | None = None) -> Path:
    here = (start if start is not None else Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root: no '{_ROOT_MARKER}' found in "
        f"'{here}' or any parent directory."
    )

def _resolve_dir(env_name: str, default_relative: str, root: Path) -> Path:
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip():
        candidate = Path(raw.strip())
        if candidate.is_absolute():
            return candidate.resolve()
        return (root / candidate).resolve()
    return (root / default_relative).resolve()

def _resolve_logs_dir(root: Path) -> Path:
    for env_name in ("LOG_DIR", "LOGS_DIR"):
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip():
            return _resolve_dir(env_name, "logs", root)
    return (root / "logs").resolve()

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

ROOT: Path = find_root()

DATA_DIR: Path = _resolve_dir("DATA_DIR", "data", ROOT)
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"
TOKENIZED_DIR: Path = DATA_DIR / "tokenized"

RESULTS_DIR: Path = _resolve_dir("RESULTS_DIR", "results", ROOT)
MODELS_DIR: Path = _resolve_dir("MODELS_DIR", "models", ROOT)
CONFIGS_DIR: Path = _resolve_dir("CONFIGS_DIR", "configs", ROOT)
LOGS_DIR: Path = _resolve_logs_dir(ROOT)
