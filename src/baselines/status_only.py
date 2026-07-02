
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.baselines._common import (
    SUBTYPES,
    accuracy,
    load_split_records,
    macro_f1,
    per_class_f1,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "fit_status_rule", "predict_status_rule"]

_LOGGER = get_logger("baseline.status_only")

def _status_of(record: Mapping[str, Any]) -> int:
    try:
        return int(record.get("status", 0) or 0)
    except (TypeError, ValueError):
        return 0

def fit_status_rule(
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> Tuple[Dict[int, int], int]:
    per_status: "defaultdict[int, Counter]" = defaultdict(Counter)
    overall: Counter = Counter()
    for rec, lab in zip(records, labels):
        s = _status_of(rec)
        per_status[s][int(lab)] += 1
        overall[int(lab)] += 1

    def _argmax(counter: Counter) -> int:

        return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    status_to_subtype = {status: _argmax(c) for status, c in per_status.items()}
    global_majority = _argmax(overall) if overall else 0
    return status_to_subtype, global_majority

def predict_status_rule(
    records: Sequence[Mapping[str, Any]],
    status_to_subtype: Mapping[int, int],
    global_majority: int,
) -> List[int]:
    return [
        int(status_to_subtype.get(_status_of(rec), global_majority)) for rec in records
    ]

def run(
    cfg: Mapping[str, Any],
    seed: int = 42,
    *,
    max_records: Optional[int] = None,
    write: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    data = cfg.get("data", {})
    budget = float(data.get("label_budget", 0.05))

    train_recs = load_split_records(
        str(data.get("train_split", "owasp_train")), max_records=max_records
    )
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    idx = select_label_budget(len(train_recs), budget)
    if not idx:
        raise ValueError(f"label_budget={budget} selected 0 of {len(train_recs)} train records")
    labeled = [train_recs[i] for i in idx]

    status_to_subtype, global_majority = fit_status_rule(labeled, subtype_labels(labeled))

    test_labels = subtype_labels(test_recs)
    preds = predict_status_rule(test_recs, status_to_subtype, global_majority)
    macro = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    acc = round(100.0 * accuracy(preds, test_labels), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    rule_view = {
        int(s): SUBTYPES[sub] for s, sub in sorted(status_to_subtype.items())
    }
    _LOGGER.info(
        "status_only: budget=%.3f (%d/%d labeled), test n=%d, macro_f1=%.4f acc=%.4f",
        budget, len(labeled), len(train_recs), len(test_recs), macro, acc,
    )
    _LOGGER.info("status_only learned rule status->subtype: %s (fallback=%s)",
                 rule_view, SUBTYPES[global_majority])

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "accuracy": acc,
        "status_rule": rule_view,
        "fallback_subtype": SUBTYPES[global_majority],
        "note": (
            "Status-code-only shortcut baseline: majority subtype per HTTP status, "
            "no content. Reported to quantify the status shortcut on Table 6."
        ),
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="Status-only (shortcut)", col=col, value=macro, seed=seed,
            extra_metadata=extra, per_class=pcf1,
            preds=preds, labels=test_labels,
            split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)

    return {
        "value": macro,
        "accuracy": acc,
        "per_class_f1": pcf1,
        "status_rule": rule_view,
        "results_path": str(out_path) if out_path else None,
    }

def _smoke() -> None:
    from src.baselines._common import synthetic_records, subtype_labels as _sl

    recs = synthetic_records(120, seed=5)
    labels = _sl(recs)
    rule, fallback = fit_status_rule(recs, labels)
    preds = predict_status_rule(recs[:32], rule, fallback)
    print(
        f"[status_only] smoke: fit {len(rule)} status->subtype entries, "
        f"32 preds, train macro_f1={round(macro_f1(preds, labels[:32], len(SUBTYPES)), 3)}"
    )

if __name__ == "__main__":
    _smoke()
