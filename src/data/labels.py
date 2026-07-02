
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.utils.config import load_config
from src.utils.paths import CONFIGS_DIR

__all__ = [
    "SUBTYPES",
    "load_tag_map",
    "subtype_from_tags",
]

SUBTYPES: tuple = (
    "sql_injection",
    "rce",
    "php_injection",
    "xss",
    "path_traversal",
    "lfi",
    "scanner",
    "protocol",
)

_DEFAULT_MAP_PATH = CONFIGS_DIR / "labels" / "crs_tag_map.yaml"

@lru_cache(maxsize=8)
def load_tag_map(path: Optional[str] = None) -> Dict:
    cfg_path = Path(path) if path is not None else _DEFAULT_MAP_PATH
    cfg = load_config(cfg_path)

    default_subtype = cfg.get("default_subtype", "protocol")
    priority = dict(cfg.get("subtype_priority", {}))
    tag_to_subtype = dict(cfg.get("tag_to_subtype", {}))

    raw_fuzzy = cfg.get("fuzzy_fallback") or {}
    fuzzy_enabled = bool(raw_fuzzy.get("enabled", False))
    min_prefix_len = int(raw_fuzzy.get("min_prefix_len", 12))
    keyword_rules = []
    for rule in raw_fuzzy.get("keyword_to_subtype", []) or []:
        kw = str(rule["keyword"]).lower()
        sub = str(rule["subtype"])
        keyword_rules.append((kw, sub))

    referenced = (
        list(priority)
        + list(tag_to_subtype.values())
        + [default_subtype]
        + [sub for _, sub in keyword_rules]
    )
    unknown = {s for s in referenced if s not in SUBTYPES}
    if unknown:
        raise ValueError(
            f"crs_tag_map references unknown subtype(s) {sorted(unknown)}; "
            f"known subtypes are {SUBTYPES}"
        )

    missing_pri = [s for s in SUBTYPES if s not in priority]
    if missing_pri:
        raise ValueError(
            f"subtype_priority is missing entries for {missing_pri}; "
            "every subtype needs a priority for deterministic resolution"
        )
    return {
        "default_subtype": default_subtype,
        "subtype_priority": priority,
        "tag_to_subtype": tag_to_subtype,
        "fuzzy_enabled": fuzzy_enabled,
        "min_prefix_len": min_prefix_len,
        "keyword_rules": keyword_rules,
    }

def _fuzzy_subtype(tag: str, tag_map: Dict) -> Optional[str]:
    if not tag_map.get("fuzzy_enabled"):
        return None
    tag_to_subtype: Dict[str, str] = tag_map["tag_to_subtype"]
    min_prefix_len: int = tag_map["min_prefix_len"]

    if len(tag) >= min_prefix_len:
        prefixed_subtypes = {
            sub for key, sub in tag_to_subtype.items() if key.startswith(tag)
        }
        if len(prefixed_subtypes) == 1:
            return next(iter(prefixed_subtypes))

    low = tag.lower()
    for keyword, subtype in tag_map["keyword_rules"]:
        if keyword in low:
            return subtype
    return None

def subtype_from_tags(
    tags: Iterable[str],
    tag_map: Optional[Dict] = None,
) -> str:
    m = tag_map if tag_map is not None else load_tag_map()
    tag_to_subtype: Dict[str, str] = m["tag_to_subtype"]
    priority: Dict[str, int] = m["subtype_priority"]

    matched: List[str] = []
    for t in tags:
        sub = tag_to_subtype.get(t)
        if sub is None:

            sub = _fuzzy_subtype(t, m)
        if sub is not None:
            matched.append(sub)

    if not matched:
        return m["default_subtype"]

    return min(matched, key=lambda s: (priority[s], SUBTYPES.index(s)))
