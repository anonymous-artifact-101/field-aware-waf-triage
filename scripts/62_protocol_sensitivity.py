
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit
from src.eval.metrics import mean_ci95
from src.eval.predictions_io import (
    PREDICTIONS_DIRNAME,
    load_predictions,
    load_shared_labels,
    prediction_file_path,
)
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

_SOURCE_TABLE = "table_03_rq1_baselines"
_OUT_TABLE = "table_15_protocol_sensitivity"
_DEFAULT_ROWS: Tuple[str, ...] = (
    "Status-only (shortcut)",
    "TF-IDF + LogReg",
    "TF-IDF + SVD + HGBDT",
    "TF-IDF typed + LinearSVC",
    "TF-IDF flat + LinearSVC",
    "Char-CNN",
    "Proposed (FastText field-aware)",
    "ModSec-AdvLearn",
    "ModSec-Learn",
)

def _available_seeds(table_dir: Path, row: str, col: str) -> List[int]:
    return [
        seed
        for seed in range(0, 100)
        if prediction_file_path(table_dir, row, col, seed).is_file()
    ]

def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)

def macro_f1_excluding_true_class(
    preds: Sequence[int],
    labels: Sequence[int],
    *,
    exclude_idx: int,
    num_classes: int,
) -> float:
    retained = [(int(p), int(y)) for p, y in zip(preds, labels) if int(y) != int(exclude_idx)]
    classes = [c for c in range(int(num_classes)) if c != int(exclude_idx)]
    f1s: List[float] = []
    for c in classes:
        tp = sum(1 for p, y in retained if p == c and y == c)
        fp = sum(1 for p, y in retained if p == c and y != c)
        fn = sum(1 for p, y in retained if p != c and y == c)
        f1s.append(_f1(tp, fp, fn))
    return 100.0 * float(sum(f1s) / len(f1s))

def _mean_cell(values: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
    mean, ci = mean_ci95([float(v) for v in values])
    return round(float(mean), 4), (round(float(ci[0]), 4), round(float(ci[1]), 4))

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="7-class sensitivity excluding protocol.")
    parser.add_argument("--source-table", default=_SOURCE_TABLE)
    parser.add_argument("--out-table", default=_OUT_TABLE)
    parser.add_argument("--cols", nargs="+", default=["5%", "10%"], help="Label-budget columns.")
    parser.add_argument("--split", default="owasp_test")
    parser.add_argument("--exclude", default="protocol")
    parser.add_argument("--rows", nargs="*", default=list(_DEFAULT_ROWS))
    args = parser.parse_args(argv)

    source_dir = RESULTS_DIR / args.source_table
    out_dir = RESULTS_DIR / args.out_table
    if not (source_dir / PREDICTIONS_DIRNAME).is_dir():
        raise SystemExit(f"[62] missing predictions dir: {source_dir / PREDICTIONS_DIRNAME}")

    shared = load_shared_labels(source_dir, args.split)
    labels = [int(y) for y in shared["labels"]]
    num_classes = int(shared.get("num_classes", len(SUBTYPES)))
    subtypes = list(shared.get("subtypes") or SUBTYPES)
    if args.exclude not in subtypes:
        raise SystemExit(f"[62] exclude class {args.exclude!r} not in {subtypes}")
    exclude_idx = int(subtypes.index(args.exclude))
    sha = str(shared["labels_sha256"])
    retained_n = sum(1 for y in labels if y != exclude_idx)
    excluded_n = len(labels) - retained_n

    cells: List[Dict[str, Any]] = [
        {
            "row": "Retained non-protocol test records",
            "col": "n",
            "value": retained_n,
            "ci_95": None,
            "seeds": [],
        },
        {
            "row": "Excluded protocol test records",
            "col": "n",
            "value": excluded_n,
            "ci_95": None,
            "seeds": [],
        },
    ]
    per_row: Dict[str, Any] = {}

    for col in args.cols:
        for row in args.rows:
            seeds = _available_seeds(source_dir, row, col)
            if not seeds:
                continue
            all8_values: List[float] = []
            seven_values: List[float] = []
            by_seed: Dict[str, Any] = {}
            for seed in seeds:
                doc = load_predictions(
                    source_dir,
                    row=row,
                    col=col,
                    seed=seed,
                    expected_labels_sha256=sha,
                )
                preds = [int(p) for p in doc["preds"]]
                all8 = float(doc["macro_f1"])
                seven = macro_f1_excluding_true_class(
                    preds, labels, exclude_idx=exclude_idx, num_classes=num_classes
                )
                all8_values.append(all8)
                seven_values.append(seven)
                by_seed[str(seed)] = {
                    "macro_f1_8_class": round(all8, 4),
                    "macro_f1_7_class_excluding_protocol": round(seven, 4),
                }

            all8_mean, all8_ci = _mean_cell(all8_values)
            seven_mean, seven_ci = _mean_cell(seven_values)
            cells.append({
                "row": row,
                "col": f"{col} 8-class",
                "value": all8_mean,
                "ci_95": list(all8_ci),
                "seeds": seeds,
            })
            cells.append({
                "row": row,
                "col": f"{col} 7-class excl. protocol",
                "value": seven_mean,
                "ci_95": list(seven_ci),
                "seeds": seeds,
            })
            per_row.setdefault(row, {})[col] = {
                "seeds": seeds,
                "by_seed": by_seed,
                "macro_f1_8_class": {"mean": all8_mean, "ci_95": list(all8_ci)},
                "macro_f1_7_class_excluding_protocol": {
                    "mean": seven_mean,
                    "ci_95": list(seven_ci),
                },
            }
            print(
                f"[62] {row:30s} {col:>4s}: "
                f"8-class={all8_mean:.2f}  7-class={seven_mean:.2f}"
            )

    out = {
        "table": args.out_table,
        "cells": cells,
        "analysis": {
            "source_table": args.source_table,
            "split": args.split,
            "excluded_class": args.exclude,
            "excluded_class_index": exclude_idx,
            "n_total": len(labels),
            "n_retained_non_protocol": retained_n,
            "n_excluded_protocol": excluded_n,
            "definition": (
                "Filter out records whose true label is protocol; compute macro-F1 "
                "over the remaining seven subtype classes. Predictions to protocol "
                "on retained records still count as misses for the true class."
            ),
            "rows": per_row,
        },
        "metadata": {
            "commit": git_commit(),
            "dataset_version": dataset_version(),
            "labels_sha256": sha,
            "metric": "macro_f1_protocol_sensitivity",
            "source": "results/table_03_rq1_baselines/predictions/*.json",
        },
    }
    dump_json(out_dir / "results.json", out)
    print(f"[62] wrote {out_dir / 'results.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
