
from __future__ import annotations

__all__ = [
    "PAD",
    "UNK",
    "CLS",
    "SEP",
    "MASK",
    "FIELD_MASK",
    "FIELD_SEP",
    "FIELD_NAMES",
    "CORE_SPECIAL_TOKENS",
    "FIELD_MODELING_TOKENS",
    "SPECIAL_TOKENS",
    "field_sentinel",
    "field_sentinel_tokens",
    "is_special",
]

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"

FIELD_MASK = "[FIELD-MASK]"
FIELD_SEP = "[FIELD-SEP]"

FIELD_NAMES: tuple[str, ...] = ("method", "path", "query", "ua", "status", "timing")

def field_sentinel(name: str) -> str:
    return f"[FIELD={name}]"

def field_sentinel_tokens(field_names: "tuple[str, ...] | list[str]" = FIELD_NAMES) -> list[str]:
    return [field_sentinel(name) for name in field_names]

CORE_SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, CLS, SEP, MASK)
FIELD_MODELING_TOKENS: tuple[str, ...] = (FIELD_MASK, FIELD_SEP)

SPECIAL_TOKENS: list[str] = [
    *CORE_SPECIAL_TOKENS,
    *FIELD_MODELING_TOKENS,
    *field_sentinel_tokens(FIELD_NAMES),
]

assert len(SPECIAL_TOKENS) == 13, "SPECIAL_TOKENS count drifted from the documented set"
assert len(set(SPECIAL_TOKENS)) == len(SPECIAL_TOKENS), "duplicate special token"

_SPECIAL_SET = frozenset(SPECIAL_TOKENS)

def is_special(token: str) -> bool:
    return token in _SPECIAL_SET
