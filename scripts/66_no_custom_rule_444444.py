
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit, load_split_records, macro_f1
from src.eval.metrics import mean_ci95
from src.eval.predictions_io import load_predictions, load_shared_labels, prediction_file_path
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

_TABLE = "table_17_no_custom_444444"
_ROWS = [
    "Proposed (FastText field-aware)",
    "TF-IDF flat + LinearSVC",
    "TF-IDF typed + LinearSVC",
    "Field-prefixed FastText + LinearSVC",
    "Hashing char-ngram + SGD",
    "TF-IDF + SVD + HGBDT",
    "Char-CNN",
    "ModSec-Learn",
    "ModSec-AdvLearn",
    "Status-only (shortcut)",
]

def _available_seeds(table_dir: Path, row: str, col: str) -> List[int]:
    return [s for s in range(0, 100) if prediction_file_path(table_dir, row, col, s).is_file()]

def _has_rule_444444(rec: dict) -> bool:
    return any(str(rule) == "444444" for rule in rec.get("crs_rule_ids", []) or [])

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Report macro-F1 excluding rule 444444 records.")
    parser.add_argument("--source-table", default="table_03_rq1_baselines")
    parser.add_argument("--cols", nargs="+", default=["5%", "10%"])
    parser.add_argument("--rows", nargs="*", default=_ROWS)
    parser.add_argument("--split", default="owasp_test")
    args = parser.parse_args(argv)

    table_dir = RESULTS_DIR / args.source_table
    shared = load_shared_labels(table_dir, args.split)
    labels = [int(v) for v in shared["labels"]]
    sha = str(shared["labels_sha256"])
    records = load_split_records(args.split)
    if len(records) != len(labels):
        raise SystemExit(f"[66] split/prediction length mismatch: records={len(records)} labels={len(labels)}")

    keep_idx = [i for i, rec in enumerate(records) if not _has_rule_444444(rec)]
    drop_idx = [i for i, rec in enumerate(records) if _has_rule_444444(rec)]
    kept_labels = [labels[i] for i in keep_idx]
    kept_support = Counter(kept_labels)
    dropped_support = Counter(labels[i] for i in drop_idx)

    cells = [
        {
            "row": "filtered records",
            "col": "kept",
            "value": len(keep_idx),
            "ci_95": None,
            "seeds": [],
        },
        {
            "row": "filtered records",
            "col": "excluded rule 444444",
            "value": len(drop_idx),
            "ci_95": None,
            "seeds": [],
        },
    ]
    details: Dict[str, dict] = {}
    for col in args.cols:
        for row in args.rows:
            seeds = _available_seeds(table_dir, row, col)
            if not seeds:
                continue
            vals = []
            for seed in seeds:
                doc = load_predictions(
                    table_dir, row=row, col=col, seed=seed,
                    expected_labels_sha256=sha,
                )
                preds = [int(v) for v in doc["preds"]]
                kept_preds = [preds[i] for i in keep_idx]
                vals.append(100.0 * macro_f1(kept_preds, kept_labels, len(SUBTYPES)))
            mean, (lo, hi) = mean_ci95(vals)
            cell = {
                "row": row,
                "col": f"{col} macro_f1_no_444444",
                "value": round(float(mean), 4),
                "ci_95": [round(float(lo), 4), round(float(hi), 4)],
                "seeds": [int(s) for s in seeds],
            }
            cells.append(cell)
            details[f"{row} | {col}"] = dict(cell)

    out = {
        "table": _TABLE,
        "cells": cells,
        "metadata": {
            "commit": git_commit(),
            "date": date.today().isoformat(),
            "gpu": "cpu",
            "source_table": args.source_table,
            "split": args.split,
            "dataset_version": dataset_version(),
            "labels_sha256": sha,
            "sensitivity_type": "evaluation_only_filter_no_refit",
            "filter": "exclude any test record whose crs_rule_ids contains 444444",
            "n_total": len(records),
            "n_kept": len(keep_idx),
            "n_excluded_444444": len(drop_idx),
            "kept_support": {SUBTYPES[i]: int(kept_support[i]) for i in range(len(SUBTYPES))},
            "excluded_support": {SUBTYPES[i]: int(dropped_support[i]) for i in range(len(SUBTYPES))},
            "details": details,
        },
    }
    out_path = RESULTS_DIR / _TABLE / "results.json"
    dump_json(out_path, out)
    print(f"[66] wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
