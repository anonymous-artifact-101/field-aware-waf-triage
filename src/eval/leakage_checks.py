
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

__all__ = [
    "LeakageError",
    "check_temporal_order",
    "assert_temporal_order",
    "check_no_overlap",
    "assert_no_overlap",
    "check_status_shortcut",
]

Record = Mapping[str, Any]
Splits = Mapping[str, Sequence[Record]]

_ORDER = ("train", "val", "test")

class LeakageError(AssertionError):
    """Raised by the ``assert_*`` helpers when a leakage check fails."""

def _parse_ts(value: Any) -> datetime:
    if value is None:
        raise LeakageError("Record is missing a 'timestamp'.")
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise LeakageError(f"Unparseable ISO-8601 timestamp: {value!r}") from exc

def _ts_bounds(records: Sequence[Record]) -> tuple[datetime, datetime] | None:
    parsed = [_parse_ts(r.get("timestamp")) for r in records]
    if not parsed:
        return None
    return (min(parsed), max(parsed))

def _present_order(splits: Splits) -> list[str]:
    return [name for name in _ORDER if name in splits]

def check_temporal_order(splits: Splits) -> dict[str, Any]:
    bounds: dict[str, Any] = {}
    for name in _present_order(splits):
        b = _ts_bounds(splits[name])
        if b is None:
            bounds[name] = {"min": None, "max": None, "n": 0}
        else:
            lo, hi = b
            bounds[name] = {"min": lo.isoformat(), "max": hi.isoformat(), "n": len(splits[name])}

    non_empty = [name for name in _present_order(splits) if bounds[name]["n"] > 0]
    comparisons: list[dict[str, Any]] = []
    for earlier, later in zip(non_empty, non_empty[1:]):
        e_max = _ts_bounds(splits[earlier])[1]
        l_min = _ts_bounds(splits[later])[0]
        ok = e_max <= l_min
        comparisons.append(
            {
                "earlier": earlier,
                "later": later,
                "earlier_max": e_max.isoformat(),
                "later_min": l_min.isoformat(),
                "ok": ok,
            }
        )

    violations = [c for c in comparisons if not c["ok"]]
    return {
        "passed": not violations,
        "bounds": bounds,
        "comparisons": comparisons,
        "violations": violations,
    }

def assert_temporal_order(splits: Splits) -> dict[str, Any]:
    report = check_temporal_order(splits)
    if not report["passed"]:
        msgs = [
            f"{v['earlier']}.max={v['earlier_max']} > {v['later']}.min={v['later_min']}"
            for v in report["violations"]
        ]
        raise LeakageError("Temporal order violated: " + "; ".join(msgs))
    return report

def check_no_overlap(splits: Splits) -> dict[str, Any]:
    present = _present_order(splits)

    id_sets: dict[str, set[str]] = {}
    dup_within: dict[str, list[str]] = {}
    for name in present:
        seen: set[str] = set()
        dups: set[str] = set()
        for record in splits[name]:
            uid = record.get("unique_id")
            uid = "" if uid is None else str(uid)
            if uid in seen:
                dups.add(uid)
            seen.add(uid)
        id_sets[name] = seen
        if dups:
            dup_within[name] = sorted(dups)

    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            shared = id_sets[a] & id_sets[b]
            if shared:
                overlaps[f"{a}|{b}"] = sorted(shared)

    overlapping_ids: set[str] = set()
    for ids in overlaps.values():
        overlapping_ids.update(ids)

    return {
        "passed": not overlaps and not dup_within,
        "duplicates_within_split": dup_within,
        "overlaps": overlaps,
        "n_overlapping_ids": len(overlapping_ids),
    }

def assert_no_overlap(splits: Splits) -> dict[str, Any]:
    report = check_no_overlap(splits)
    if not report["passed"]:
        parts: list[str] = []
        if report["overlaps"]:
            parts.append(
                "cross-split unique_id overlap in "
                + ", ".join(report["overlaps"].keys())
            )
        if report["duplicates_within_split"]:
            parts.append(
                "duplicate unique_id within "
                + ", ".join(report["duplicates_within_split"].keys())
            )
        raise LeakageError("; ".join(parts))
    return report

def _label_to_int(label: Any) -> int | None:
    if label == "attack":
        return 1
    if label == "benign":
        return 0
    return None

def _binary_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def check_status_shortcut(records: Sequence[Record]) -> dict[str, Any]:
    statuses: list[int] = []
    labels: list[int] = []
    for record in records:
        y = _label_to_int(record.get("label"))
        if y is None:
            continue
        status = record.get("status")
        try:
            status = int(status) if status is not None else 0
        except (TypeError, ValueError):
            status = 0
        statuses.append(status)
        labels.append(y)

    n = len(labels)
    n_attack = sum(labels)
    n_benign = n - n_attack

    per_status: dict[int, dict[str, Any]] = {}
    for status, y in zip(statuses, labels):
        cell = per_status.setdefault(status, {"n": 0, "attack": 0})
        cell["n"] += 1
        cell["attack"] += y
    for status, cell in per_status.items():
        cell["attack_rate"] = cell["attack"] / cell["n"] if cell["n"] else 0.0

    attack_statuses = sorted(s for s, c in per_status.items() if c["attack_rate"] > 0.5)
    attack_set = set(attack_statuses)
    tp = fp = fn = 0
    for status, y in zip(statuses, labels):
        pred = 1 if status in attack_set else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
    rule_f1 = _binary_f1(tp, fp, fn)

    result: dict[str, Any] = {
        "feature": "status",
        "n": n,
        "n_attack": n_attack,
        "n_benign": n_benign,

        "per_status": {
            str(s): per_status[s] for s in sorted(per_status)
        },
        "rule_predict_attack_statuses": attack_statuses,
        "rule_f1": rule_f1,
        "rule_confusion": {"tp": tp, "fp": fp, "fn": fn},
    }

    if n == 0 or n_attack == 0 or n_benign == 0:
        result["logreg"] = None
        result["logreg_note"] = (
            "Skipped logistic regression: need both classes present "
            f"(n_attack={n_attack}, n_benign={n_benign})."
        )
        return result

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
    except ImportError as exc:
        result["logreg"] = None
        result["logreg_error"] = f"scikit-learn/numpy unavailable: {exc}"
        return result

    x = np.asarray(statuses, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(labels, dtype=int)

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(x, y_arr)
    pred = clf.predict(x)
    result["logreg"] = {
        "accuracy": float((pred == y_arr).mean()),
        "f1": float(f1_score(y_arr, pred, pos_label=1, zero_division=0)),
        "coef": float(clf.coef_[0][0]),
        "intercept": float(clf.intercept_[0]),
        "note": (
            "Trained and evaluated on the same records (in-sample); this is an "
            "optimistic measure of how leaky the status code is, not a "
            "generalization estimate."
        ),
    }
    return result
