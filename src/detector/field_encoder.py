
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.baselines._common import FIELD_ORDER, _field_text
from src.detector.fasttext_embed import tokenize_field

__all__ = [
    "FIELD_ORDER",
    "GRANULARITY_GROUPS",
    "resolve_groups",
    "FieldEncoder",
]

GRANULARITY_GROUPS = {
    6: [("method",), ("path",), ("query",), ("ua",), ("status",), ("timing",)],
    3: [("method", "path", "query"), ("ua",), ("status", "timing")],
    1: [("method", "path", "query", "ua", "status", "timing")],
}

GRANULARITY_LABELS = {
    6: ["method", "path", "query", "ua", "status", "timing"],
    3: ["request_line", "client", "response"],
    1: ["record"],
}

def resolve_groups(
    granularity: "int | str | None",
    *,
    groups: Optional[Sequence[Sequence[str]]] = None,
) -> Tuple[List[Tuple[str, ...]], List[str]]:
    if groups is not None:
        resolved: List[Tuple[str, ...]] = []
        for g in groups:
            members = tuple(str(f) for f in g)
            bad = [f for f in members if f not in FIELD_ORDER]
            if bad:
                raise ValueError(
                    f"field group {members} names unknown field(s) {bad}; "
                    f"known fields are {FIELD_ORDER}"
                )
            resolved.append(members)
        if not resolved or all(len(g) == 0 for g in resolved):
            raise ValueError("explicit field groups must contain at least one field")
        labels = [
            "_".join(g) if len(g) > 1 else (g[0] if g else "empty") for g in resolved
        ]
        return resolved, labels

    n = _granularity_to_int(granularity)
    if n not in GRANULARITY_GROUPS:
        raise ValueError(
            f"unsupported granularity {granularity!r}; built-in options are "
            f"{sorted(GRANULARITY_GROUPS)} (or pass explicit groups)"
        )
    return list(GRANULARITY_GROUPS[n]), list(GRANULARITY_LABELS[n])

def _granularity_to_int(granularity: "int | str | None") -> int:
    if granularity is None:
        return 6
    if isinstance(granularity, int):
        return granularity
    g = str(granularity).strip().lower()
    aliases = {
        "6": 6,
        "6-field": 6,
        "six_field": 6,
        "six": 6,
        "3": 3,
        "3-field": 3,
        "three_field": 3,
        "three": 3,
        "1": 1,
        "flat": 1,
        "flat_no_typed_embed": 1,
        "one": 1,
    }
    if g not in aliases:
        raise ValueError(
            f"unrecognized granularity string {granularity!r}; use one of "
            f"{sorted(set(aliases))}"
        )
    return aliases[g]

class FieldEncoder:

    def __init__(
        self,
        fasttext_model,
        *,
        granularity: "int | str | None" = 6,
        groups: Optional[Sequence[Sequence[str]]] = None,
        pooling: str = "mean",
    ) -> None:
        if pooling not in ("mean", "max"):
            raise ValueError(f"pooling must be 'mean' or 'max', got {pooling!r}")
        self.model = fasttext_model
        self.wv = fasttext_model.wv
        self.vector_size = int(self.wv.vector_size)
        self.pooling = pooling
        self.groups, self.block_labels = resolve_groups(granularity, groups=groups)
        self.num_blocks = len(self.groups)
        self.feature_dim = self.num_blocks * self.vector_size

    def block_slice(self, block_index: int) -> slice:
        vs = self.vector_size
        return slice(block_index * vs, (block_index + 1) * vs)

    def _pool(self, tokens: Sequence[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        vecs = np.empty((len(tokens), self.vector_size), dtype=np.float32)
        for i, tok in enumerate(tokens):

            vecs[i] = self.wv[tok]
        if self.pooling == "max":
            return vecs.max(axis=0)
        return vecs.mean(axis=0)

    def _group_tokens(self, record: Mapping[str, Any], group: Tuple[str, ...]) -> List[str]:
        toks: List[str] = []
        for field in group:
            toks.extend(tokenize_field(_field_text(record, field)))
        return toks

    def encode_record(self, record: Mapping[str, Any]) -> np.ndarray:
        blocks = [self._pool(self._group_tokens(record, g)) for g in self.groups]
        return np.concatenate(blocks).astype(np.float32, copy=False)

    def encode_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        n = len(records)
        out = np.zeros((n, self.feature_dim), dtype=np.float32)
        for i, rec in enumerate(records):
            out[i] = self.encode_record(rec)
        return out

    def encode_record_blocks(self, record: Mapping[str, Any]) -> List[np.ndarray]:
        return [self._pool(self._group_tokens(record, g)) for g in self.groups]

def _smoke() -> None:
    from src.baselines._common import synthetic_records
    from src.detector.fasttext_embed import train_fasttext

    recs = synthetic_records(80, seed=2)
    model = train_fasttext(
        recs,
        {"vector_size": 8, "epochs": 2, "min_count": 1, "min_n": 2, "max_n": 4},
        seed=42,
    )
    for g in (6, 3, 1):
        enc = FieldEncoder(model, granularity=g)
        X = enc.encode_records(recs[:10])
        assert X.shape == (10, enc.feature_dim), X.shape
        assert X.shape[1] == enc.num_blocks * 8
        print(
            f"[field_encoder] smoke: granularity={g} blocks={enc.block_labels} "
            f"X.shape={X.shape}"
        )

    enc6 = FieldEncoder(model, granularity=6)
    empty = enc6.encode_record({"method": "", "path": "", "query": "", "ua": "",
                                "status": "", "timing": ""})
    print(f"[field_encoder] smoke: empty-record feature all_finite="
          f"{bool(np.isfinite(empty).all())} norm={float((empty**2).sum()**0.5):.4f}")

if __name__ == "__main__":
    _smoke()
