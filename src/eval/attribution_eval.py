
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.data.labels import SUBTYPES
from src.data.tokenizers.special_tokens import FIELD_NAMES
from src.utils.config import load_config
from src.utils.paths import CONFIGS_DIR

__all__ = ["load_attribution_truth", "evaluate_attribution"]

_DEFAULT_TRUTH_PATH = CONFIGS_DIR / "eval" / "attribution_truth.yaml"

RANDOM_FIELD_FLOOR = 100.0 / 6.0

_PROTOCOL_SUBTYPE = "protocol"

def _stratify_subtypes(truth: Mapping[str, "List[str]"]) -> Dict[str, List[str]]:
    query_borne: List[str] = []
    field_observable: List[str] = []
    for s in SUBTYPES:
        if s == _PROTOCOL_SUBTYPE:
            continue
        primary = (truth.get(s) or [None])[0]
        (query_borne if primary == "query" else field_observable).append(s)
    return {"query_borne": query_borne, "field_observable": field_observable}

def load_attribution_truth(path: Optional[str] = None) -> Dict[str, List[str]]:
    cfg_path = Path(path) if path is not None else _DEFAULT_TRUTH_PATH
    cfg = load_config(cfg_path)
    expected = dict(cfg.get("expected_fields", {}))

    field_set = set(FIELD_NAMES)
    missing = [s for s in SUBTYPES if s not in expected]
    if missing:
        raise ValueError(
            f"attribution_truth is missing expected_fields for subtypes {missing}; "
            f"every subtype in {SUBTYPES} needs an entry"
        )
    out: Dict[str, List[str]] = {}
    for subtype, fields in expected.items():
        if subtype not in SUBTYPES:
            raise ValueError(
                f"attribution_truth references unknown subtype {subtype!r}; "
                f"known subtypes are {SUBTYPES}"
            )
        names = list(fields)
        bad = [f for f in names if f not in field_set]
        if bad:
            raise ValueError(
                f"attribution_truth subtype {subtype!r} names unknown field(s) {bad}; "
                f"known fields are {sorted(field_set)}"
            )
        out[subtype] = names
    return out

def evaluate_attribution(
    detector,
    records: List[Mapping[str, Any]],
    *,
    truth_path: Optional[str] = None,
    max_failures: int = 50,
) -> Dict[str, Any]:
    truth = load_attribution_truth(truth_path)

    labels = list(getattr(detector.encoder, "block_labels", []))
    if sorted(labels) != sorted(FIELD_NAMES):
        raise ValueError(
            "field-attribution requires a 6-field detector (block labels must be "
            f"the canonical fields {sorted(FIELD_NAMES)}); got blocks {labels}. "
            "Use the six_field / proposed granularity."
        )

    attributions = detector.field_attribution(records)

    correct = {s: 0 for s in SUBTYPES}
    support = {s: 0 for s in SUBTYPES}
    failures: List[Dict[str, Any]] = []
    for rec, attr in zip(records, attributions):
        subtype = str(rec.get("attack_subtype", ""))
        if subtype not in support:
            continue
        support[subtype] += 1
        predicted_field = attr["field"]
        expected = truth.get(subtype, [])
        if predicted_field in expected:
            correct[subtype] += 1
        elif len(failures) < int(max_failures):
            failures.append(
                {
                    "subtype": subtype,
                    "predicted_field": predicted_field,
                    "expected_fields": expected,
                    "deltas": attr["deltas"],
                    "path": rec.get("path", ""),
                    "query": rec.get("query", ""),
                }
            )

    per_subtype: Dict[str, float] = {}
    for s in SUBTYPES:
        per_subtype[s] = (correct[s] / support[s]) if support[s] else float("nan")

    scored = [v for s, v in per_subtype.items() if support[s] > 0]
    macro = float(sum(scored) / len(scored)) if scored else float("nan")

    strata = _stratify_subtypes(truth)

    def _macro_over(subtypes: List[str]) -> float:
        vals = [per_subtype[s] for s in subtypes if support.get(s, 0) > 0]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    macro_field_observable = _macro_over(strata["field_observable"])
    macro_query_borne = _macro_over(strata["query_borne"])
    macro_excl_protocol = _macro_over(
        [s for s in SUBTYPES if s != _PROTOCOL_SUBTYPE]
    )

    populated: Dict[str, float] = {}
    field_populated_count: Dict[str, int] = {s: 0 for s in SUBTYPES}
    query_nonempty_total = 0
    for rec in records:
        subtype = str(rec.get("attack_subtype", ""))
        if str(rec.get("query", "")).strip():
            query_nonempty_total += 1
        if subtype in field_populated_count:
            primary = (truth.get(subtype) or [None])[0]
            if primary and str(rec.get(primary, "")).strip():
                field_populated_count[subtype] += 1
    for s in SUBTYPES:
        populated[s] = (field_populated_count[s] / support[s]) if support[s] else float("nan")

    n_total = sum(support.values())

    query_empty_rate = (
        1.0 - (query_nonempty_total / len(records)) if records else float("nan")
    )

    always_path: Dict[str, float] = {
        s: (1.0 if "path" in (truth.get(s) or []) else 0.0) for s in SUBTYPES
    }
    always_path_macro_excl_protocol = float(
        sum(always_path[s] for s in SUBTYPES if s != _PROTOCOL_SUBTYPE)
        / max(1, len([s for s in SUBTYPES if s != _PROTOCOL_SUBTYPE]))
    )

    return {
        "per_subtype": per_subtype,
        "macro_agreement": macro,
        "macro_excl_protocol": macro_excl_protocol,
        "macro_field_observable": macro_field_observable,
        "macro_query_borne": macro_query_borne,
        "strata": strata,
        "per_subtype_populated": populated,
        "query_empty_rate": query_empty_rate,
        "random_field_floor": RANDOM_FIELD_FLOOR / 100.0,
        "always_path_macro_excl_protocol": always_path_macro_excl_protocol,
        "n": n_total,
        "support": support,
        "failure_cases": failures,
    }

def _smoke() -> None:
    from src.baselines._common import subtype_labels, synthetic_records
    from src.detector.classifier import FieldAwareDetector
    from src.detector.fasttext_embed import train_fasttext
    from src.detector.field_encoder import FieldEncoder

    recs = synthetic_records(96, seed=7)
    model = train_fasttext(
        recs, {"vector_size": 16, "epochs": 3, "min_count": 1, "min_n": 2, "max_n": 4}, seed=42
    )
    det = FieldAwareDetector(
        FieldEncoder(model, granularity=6), mode="supervised", estimator="logreg", seed=42
    )
    det.fit(recs, subtype_labels(recs))
    out = evaluate_attribution(det, recs[:40])
    print(
        f"[attribution_eval] smoke: macro_agreement={out['macro_agreement']:.3f} "
        f"n={out['n']} per_subtype={ {k: round(v, 2) for k, v in out['per_subtype'].items()} }"
    )

if __name__ == "__main__":
    _smoke()
