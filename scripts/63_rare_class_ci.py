
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit, load_split_records, per_class_f1
from src.eval.predictions_io import load_predictions, load_shared_labels
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

_PROPOSED_ROW = "Proposed (FastText field-aware)"
_TABLE = "table_16_rare_class_ci"

def _temporal_blocks(records: Sequence[dict], *, block_size: int, cluster_key: str) -> List[List[int]]:
    blocks: List[List[int]] = []
    cur: List[int] = []
    cur_key = None
    max_len = max(1, int(block_size))
    for idx, rec in enumerate(records):
        key = rec.get(cluster_key) if cluster_key else None
        if cur and (len(cur) >= max_len or key != cur_key):
            blocks.append(cur)
            cur = []
        if not cur:
            cur_key = key
        cur.append(idx)
    if cur:
        blocks.append(cur)
    return blocks

def _confusion_for_indices(preds: np.ndarray, labels: np.ndarray, indices: Sequence[int], C: int) -> np.ndarray:
    conf = np.zeros((C, C), dtype=np.int64)
    for idx in indices:
        y = int(labels[int(idx)])
        p = int(preds[int(idx)])
        if 0 <= y < C and 0 <= p < C:
            conf[y, p] += 1
    return conf

def _per_class_f1_from_conf(conf: np.ndarray) -> np.ndarray:
    tp = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    support = conf.sum(axis=2).astype(np.float64)
    predicted = conf.sum(axis=1).astype(np.float64)
    fp = predicted - tp
    fn = support - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=np.float64)
    np.divide(2.0 * tp, denom, out=f1, where=denom > 0.0)
    return f1 * 100.0

def _bootstrap_per_class(
    preds: Sequence[int],
    labels: Sequence[int],
    blocks: Sequence[Sequence[int]],
    *,
    B: int,
    seed: int,
    chunk: int,
) -> np.ndarray:
    C = len(SUBTYPES)
    p = np.asarray([int(v) for v in preds], dtype=np.int64)
    y = np.asarray([int(v) for v in labels], dtype=np.int64)
    block_conf = np.stack([_confusion_for_indices(p, y, block, C) for block in blocks])
    n_blocks = block_conf.shape[0]
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(B), C), dtype=np.float64)
    done = 0
    while done < int(B):
        b = min(int(chunk), int(B) - done)
        draws = rng.integers(0, n_blocks, size=(b, n_blocks))
        conf = block_conf[draws].sum(axis=1)
        out[done:done + b, :] = _per_class_f1_from_conf(conf)
        done += b
    return out

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Rare-class temporal-block F1 CIs.")
    parser.add_argument("--source-table", default="table_03_rq1_baselines")
    parser.add_argument("--cols", nargs="+", default=["5%", "10%"])
    parser.add_argument("--row", default=_PROPOSED_ROW)
    parser.add_argument("--seed", type=int, default=42, help="Prediction sidecar seed.")
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--split", default="owasp_test")
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--cluster-key", default="day")
    parser.add_argument("--rare-support-threshold", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=1000)
    args = parser.parse_args(argv)

    table_dir = RESULTS_DIR / args.source_table
    shared = load_shared_labels(table_dir, args.split)
    labels = [int(v) for v in shared["labels"]]
    sha = str(shared["labels_sha256"])
    records = load_split_records(args.split)
    if len(records) != len(labels):
        raise SystemExit(f"[63] split/prediction length mismatch: records={len(records)} labels={len(labels)}")
    blocks = _temporal_blocks(records, block_size=args.block_size, cluster_key=args.cluster_key)
    block_sizes = [len(b) for b in blocks]
    support = Counter(labels)

    cells = []
    details: Dict[str, dict] = {}
    for col in args.cols:
        doc = load_predictions(
            table_dir, row=args.row, col=col, seed=args.seed,
            expected_labels_sha256=sha,
        )
        preds = [int(v) for v in doc["preds"]]
        point = [100.0 * float(v) for v in per_class_f1(preds, labels, len(SUBTYPES))]
        boot = _bootstrap_per_class(
            preds, labels, blocks, B=args.B, seed=args.bootstrap_seed, chunk=args.chunk
        )
        col_details: Dict[str, dict] = {}
        for i, subtype in enumerate(SUBTYPES):
            lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
            is_rare = int(support[i]) < int(args.rare_support_threshold)
            cells.extend([
                {
                    "row": subtype,
                    "col": f"{col} f1",
                    "value": round(float(point[i]), 4),
                    "ci_95": [round(float(lo), 4), round(float(hi), 4)],
                    "seeds": [int(args.seed)],
                },
                {
                    "row": subtype,
                    "col": f"{col} support",
                    "value": int(support[i]),
                    "ci_95": None,
                    "seeds": [],
                },
            ])
            col_details[subtype] = {
                "f1": round(float(point[i]), 4),
                "ci_95": [round(float(lo), 4), round(float(hi), 4)],
                "support": int(support[i]),
                "rare": bool(is_rare),
            }
        details[col] = col_details

    out = {
        "table": _TABLE,
        "cells": cells,
        "metadata": {
            "commit": git_commit(),
            "date": date.today().isoformat(),
            "gpu": "cpu",
            "source_table": args.source_table,
            "row": args.row,
            "split": args.split,
            "dataset_version": dataset_version(),
            "labels_sha256": sha,
            "B": int(args.B),
            "rng_seed": int(args.bootstrap_seed),
            "bootstrap_unit": "temporal_cluster_block",
            "block_size": int(args.block_size),
            "cluster_key": str(args.cluster_key),
            "n_blocks": len(blocks),
            "block_size_summary": {
                "min": min(block_sizes),
                "median": statistics.median(block_sizes),
                "max": max(block_sizes),
            },
            "rare_support_threshold": int(args.rare_support_threshold),
            "details": details,
        },
    }
    out_path = RESULTS_DIR / _TABLE / "results.json"
    dump_json(out_path, out)
    print(f"[63] wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
