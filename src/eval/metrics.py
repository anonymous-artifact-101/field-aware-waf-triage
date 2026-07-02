
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):

    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import math
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "confusion_counts",
    "precision_recall_f1",
    "per_class_f1",
    "macro_f1",
    "micro_f1",
    "accuracy",
    "binary_detection_prf1",
    "roc_auc",
    "pr_auc",
    "mean_ci95",
    "bootstrap_ci",
    "make_cells",
    "assemble_results_json",
]

def _to_int_list(values: Sequence) -> List[int]:
    return [int(v) for v in values]

def confusion_counts(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int,
) -> List[List[int]]:
    preds = _to_int_list(preds)
    labels = _to_int_list(labels)
    if len(preds) != len(labels):
        raise ValueError(
            f"preds ({len(preds)}) and labels ({len(labels)}) length mismatch"
        )
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(labels, preds):
        if not (0 <= t < num_classes) or not (0 <= p < num_classes):
            raise ValueError(
                f"class index out of range for num_classes={num_classes}: "
                f"true={t}, pred={p}"
            )
        matrix[t][p] += 1
    return matrix

def precision_recall_f1(
    tp: int, fp: int, fn: int, *, as_percent: bool = True
) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    scale = 100.0 if as_percent else 1.0
    return precision * scale, recall * scale, f1 * scale

def per_class_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int,
    *,
    class_names: Optional[Sequence[str]] = None,
    as_percent: bool = True,
) -> Dict[str, Dict[str, float]]:
    names = (
        list(class_names)
        if class_names is not None
        else [str(c) for c in range(num_classes)]
    )
    if len(names) != num_classes:
        raise ValueError(
            f"class_names length {len(names)} must equal num_classes {num_classes}"
        )
    preds = _to_int_list(preds)
    labels = _to_int_list(labels)
    out: Dict[str, Dict[str, float]] = {}
    for c in range(num_classes):
        tp = sum(1 for p, t in zip(preds, labels) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, labels) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, labels) if p != c and t == c)
        support = sum(1 for t in labels if t == c)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn, as_percent=as_percent)
        out[names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return out

def macro_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int,
    *,
    as_percent: bool = True,
) -> float:
    per = per_class_f1(preds, labels, num_classes, as_percent=as_percent)
    if not per:
        return float("nan")
    return sum(d["f1"] for d in per.values()) / len(per)

def accuracy(preds: Sequence[int], labels: Sequence[int], *, as_percent: bool = True) -> float:
    preds = _to_int_list(preds)
    labels = _to_int_list(labels)
    if not labels:
        return float("nan")
    correct = sum(1 for p, t in zip(preds, labels) if p == t)
    frac = correct / len(labels)
    return frac * (100.0 if as_percent else 1.0)

_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
}

def mean_ci95(values: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        nan = float("nan")
        return nan, (nan, nan)
    mean = sum(vals) / n
    if n == 1:
        return mean, (mean, mean)
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    df = n - 1
    t = _T_95.get(df, 1.96)
    margin = t * sem
    return mean, (mean - margin, mean + margin)

def micro_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int,
    *,
    as_percent: bool = True,
) -> float:
    per = per_class_f1(preds, labels, num_classes, as_percent=False)

    cm = confusion_counts(preds, labels, num_classes)
    tp = sum(cm[c][c] for c in range(num_classes))
    fp = sum(sum(cm[r][c] for r in range(num_classes)) - cm[c][c] for c in range(num_classes))
    fn = sum(sum(cm[c]) - cm[c][c] for c in range(num_classes))
    _, _, f1 = precision_recall_f1(tp, fp, fn, as_percent=as_percent)
    return f1

def binary_detection_prf1(
    preds: Sequence[int],
    labels: Sequence[int],
    *,
    pos_label: int = 1,
    as_percent: bool = True,
) -> Dict[str, float]:
    preds = _to_int_list(preds)
    labels = _to_int_list(labels)
    if len(preds) != len(labels):
        raise ValueError(f"preds ({len(preds)}) and labels ({len(labels)}) length mismatch")
    tp = sum(1 for p, t in zip(preds, labels) if p == pos_label and t == pos_label)
    fp = sum(1 for p, t in zip(preds, labels) if p == pos_label and t != pos_label)
    fn = sum(1 for p, t in zip(preds, labels) if p != pos_label and t == pos_label)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn, as_percent=as_percent)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "support": float(tp + fn),
    }

def _binary_labels(labels: Sequence[int], pos_label: int) -> List[int]:
    return [1 if int(v) == pos_label else 0 for v in labels]

def roc_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    pos_label: int = 1,
) -> float:
    y = _binary_labels(labels, pos_label)
    s = [float(v) for v in scores]
    if len(y) != len(s):
        raise ValueError(f"labels ({len(y)}) and scores ({len(s)}) length mismatch")
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_ranks_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def pr_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    pos_label: int = 1,
) -> float:
    y = _binary_labels(labels, pos_label)
    s = [float(v) for v in scores]
    if len(y) != len(s):
        raise ValueError(f"labels ({len(y)}) and scores ({len(s)}) length mismatch")
    if sum(y) == 0:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))

def bootstrap_ci(
    values: Sequence[float],
    *,
    n: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    import numpy as np

    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n), arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boot_means, 100.0 * alpha))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return (mean, lo, hi)

def make_cells(
    triples: Sequence[Tuple[str, str, Sequence[float]]],
    *,
    seeds: Sequence[int],
    n_bootstrap: int = 1000,
    ci_seed: int = 0,
    round_to: Optional[int] = 1,
) -> List[Dict[str, Any]]:

    def _fmt(x: float) -> float:
        if round_to is not None and not (isinstance(x, float) and math.isnan(x)):
            return round(float(x), round_to)
        return float(x)

    cells: List[Dict[str, Any]] = []
    for row, col, vals in triples:
        mean, lo, hi = bootstrap_ci(vals, n=n_bootstrap, seed=ci_seed)
        cells.append(
            {
                "row": row,
                "col": col,
                "value": _fmt(mean),
                "ci_95": [_fmt(lo), _fmt(hi)],
                "seeds": list(seeds),
            }
        )
    return cells

def assemble_results_json(
    table: str,
    cells: Sequence[Mapping[str, Any]],
    *,
    commit: str = "UNKNOWN",
    run_date: Optional[str] = None,
    gpu: str = "CPU",
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "commit": commit,
        "date": run_date if run_date is not None else date.today().isoformat(),
        "gpu": gpu,
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            metadata.setdefault(k, v)
    return {
        "table": table,
        "cells": [dict(c) for c in cells],
        "metadata": metadata,
    }

def _self_test() -> None:
    from src.data.labels import SUBTYPES

    labels = [1, 1, 1, 0, 0, 0, 1, 0]
    preds = [1, 1, 0, 0, 0, 1, 1, 0]
    b = binary_detection_prf1(preds, labels, as_percent=False)
    assert b["tp"] == 3 and b["fp"] == 1 and b["fn"] == 1, b
    exp_f1 = 2 * (3 / 4) * (3 / 4) / (3 / 4 + 3 / 4)
    assert abs(b["f1"] - exp_f1) < 1e-9, b
    print(f"[metrics] binary attack F1={b['f1']:.3f} (P={b['precision']:.3f} R={b['recall']:.3f})")

    n_cls = len(SUBTYPES)
    mt = [0, 0, 1, 1, 2, 2, 0, 1]
    mp = [0, 1, 1, 1, 2, 0, 0, 1]
    cm = confusion_counts(mp, mt, n_cls)
    assert len(cm) == n_cls and sum(sum(r) for r in cm) == len(mt)
    macro = macro_f1(mp, mt, n_cls)
    micro = micro_f1(mp, mt, n_cls)
    acc = accuracy(mp, mt)
    assert abs(micro - acc) < 1e-6, (micro, acc)
    print(f"[metrics] subtype macro_f1={macro:.2f} micro_f1={micro:.2f} acc={acc:.2f} (%)")

    yl = [0, 0, 0, 1, 1, 1]
    sc = [0.1, 0.2, 0.35, 0.4, 0.8, 0.9]
    auc = roc_auc(yl, sc)
    ap = pr_auc(yl, sc)
    assert abs(auc - 1.0) < 1e-9, auc
    print(f"[metrics] anomaly ROC-AUC={auc:.3f} PR-AUC={ap:.3f}")

    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    yy = rng.integers(0, 2, size=200).tolist()
    ss = (rng.normal(size=200) + np.asarray(yy) * 0.7).tolist()
    assert abs(roc_auc(yy, ss) - float(roc_auc_score(yy, ss))) < 1e-9
    print("[metrics] ROC-AUC matches sklearn on a 200-sample case")

    per_seed = [91.2, 92.4, 90.8, 93.1, 92.0]
    mean, lo, hi = bootstrap_ci(per_seed, n=2000, seed=42)
    assert lo <= mean <= hi
    assert bootstrap_ci(per_seed, n=2000, seed=42) == (mean, lo, hi)
    print(f"[metrics] bootstrap_ci(5 seeds) mean={mean:.3f} 95% CI=[{lo:.3f}, {hi:.3f}]")

    cells = make_cells(
        [("Proposed (Low-label)", "5%", per_seed)],
        seeds=[42, 43, 44, 45, 46],
    )
    doc = assemble_results_json("table_03_rq1_baselines", cells, commit="abc1234")
    cell = doc["cells"][0]
    assert set(cell) == {"row", "col", "value", "ci_95", "seeds"}, cell
    assert cell["seeds"] == [42, 43, 44, 45, 46] and len(cell["ci_95"]) == 2
    print(f"[metrics] results.json cell: {cell}")
    print("[metrics] self-test passed.")

if __name__ == "__main__":
    _self_test()
