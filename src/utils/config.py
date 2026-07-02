
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import ROOT

__all__ = ["load_config", "deep_merge", "ConfigError"]

_EXTENDS_KEY = "extends"

class ConfigError(ValueError):
    """Raised for malformed configs: bad ``extends`` targets or cycles."""

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(base)
    for key, override_value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = deep_merge(base_value, override_value)
        else:
            result[key] = copy.deepcopy(override_value)
    return result

def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config '{path}': {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config '{path}' must define a mapping at the top level, "
            f"got {type(data).__name__}."
        )
    return data

def _resolve_extends_path(raw_target: Any, child_path: Path) -> Path:
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise ConfigError(
            f"'{_EXTENDS_KEY}' in '{child_path}' must be a non-empty string "
            f"path relative to the repository root, got {raw_target!r}."
        )
    return (ROOT / raw_target.strip()).resolve()

def _load_resolved(path: Path, seen: list[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ConfigError(f"Cyclic config inheritance via '{_EXTENDS_KEY}': {chain}")

    raw = _read_yaml(path)
    extends_target = raw.get(_EXTENDS_KEY)

    child = {k: v for k, v in raw.items() if k != _EXTENDS_KEY}

    if extends_target is None:
        return copy.deepcopy(child)

    parent_path = _resolve_extends_path(extends_target, path)
    parent = _load_resolved(parent_path, [*seen, path])
    return deep_merge(parent, child)

def load_config(path: str | Path) -> dict[str, Any]:
    return _load_resolved(Path(path), seen=[])
