
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.eval.predictions_io import (
    PREDICTIONS_DIRNAME,
    load_predictions,
    load_shared_labels,
    prediction_file_path,
)
from src.baselines._common import load_split_records
from src.eval.significance import paired_block_bootstrap_macro_f1
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

_PROPOSED_ROW = "Proposed (FastText field-aware)"

_BASELINE_ROWS = [
    "Char-CNN",
    "Status-only (shortcut)",
    "TF-IDF + LogReg",
    "TF-IDF + SVD + HGBDT",

    "TF-IDF typed + LinearSVC",
    "TF-IDF flat + LinearSVC",
    "Field-prefixed FastText + LinearSVC",
    "Hashing char-ngram + SGD",
    "FastText random-init (no pre-train)",
    "ModSec-Learn",
    "ModSec-AdvLearn",
]

def _available_seeds(table_dir: Path, row: str, col: str) -> List[int]:
    return [s for s in range(0, 100)
            if prediction_file_path(table_dir, row, col, s).is_file()]

def _temporal_blocks(records: List[dict], *, block_size: int, cluster_key: str) -> List[List[int]]:
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

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Paired temporal-block bootstrap (Table 3).")
    parser.add_argument("--table", default="table_03_rq1_baselines", help="Results table dir name.")
    parser.add_argument("--col", default="5%", help="Label-budget column to compare on.")
    parser.add_argument("--proposed-seed", type=int, default=42,
                        help="Seed of the (deterministic) proposed sidecar to use.")
    parser.add_argument("--B", type=int, default=10000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed (reproducibility).")
    parser.add_argument("--split", default="owasp_test", help="Test split name.")
    parser.add_argument("--block-size", type=int, default=500,
                        help="Maximum records per contiguous block within each cluster.")
    parser.add_argument("--cluster-key", default="day",
                        help="Record field that blocks may not cross (default: day).")
    args = parser.parse_args(argv)

    table_dir = RESULTS_DIR / args.table
    pred_dir = table_dir / PREDICTIONS_DIRNAME
    if not pred_dir.is_dir():
        raise SystemExit(
            f"[55] no predictions dir at {pred_dir}. Run scripts/12_evaluate.py "
            f"(proposed) and scripts/20_run_baseline.py (baselines) first."
        )

    shared = load_shared_labels(table_dir, args.split)
    labels = [int(y) for y in shared["labels"]]
    sha = str(shared["labels_sha256"])
    num_classes = int(shared.get("num_classes", 8))
    print(f"[55] shared labels: split={args.split} n={len(labels)} sha={sha[:12]}")

    records = load_split_records(args.split)
    if len(records) != len(labels):
        raise SystemExit(
            f"[55] split/prediction length mismatch: records={len(records)} labels={len(labels)}"
        )
    blocks = _temporal_blocks(records, block_size=args.block_size, cluster_key=args.cluster_key)
    cluster_counts = Counter(str(r.get(args.cluster_key, "")) for r in records)
    block_sizes = [len(b) for b in blocks]
    block_description = (
        f"contiguous temporal blocks of up to {args.block_size} records, "
        f"not crossing {args.cluster_key} clusters"
    )
    print(
        f"[55] bootstrap blocks: {len(blocks)} blocks; "
        f"size min/median/max={min(block_sizes)}/{statistics.median(block_sizes)}/{max(block_sizes)}; "
        f"{args.cluster_key} clusters={dict(cluster_counts)}"
    )

    proposed_doc = load_predictions(
        table_dir, row=_PROPOSED_ROW, col=args.col, seed=args.proposed_seed,
        expected_labels_sha256=sha,
    )
    proposed_preds = [int(p) for p in proposed_doc["preds"]]
    proposed_mf1 = float(proposed_doc["macro_f1"])

    proposed_ci = paired_block_bootstrap_macro_f1(
        proposed_preds, proposed_preds, labels, blocks, num_classes,
        B=args.B, seed=args.seed, name_a=_PROPOSED_ROW, name_b=_PROPOSED_ROW,
        block_description=block_description,
    )["ci_a"]
    print(f"[55] proposed: macro_f1={proposed_mf1} CI={proposed_ci} "
          f"(deterministic, seed {args.proposed_seed})")

    comparisons: Dict[str, dict] = {}
    for row in _BASELINE_ROWS:
        seeds = _available_seeds(table_dir, row, args.col)
        if not seeds:
            print(f"[55]   skip {row!r}: no prediction sidecar at col {args.col!r}")
            continue

        by_seed: Dict[str, dict] = {}
        seed_means: Dict[int, float] = {}
        for s in seeds:
            doc = load_predictions(
                table_dir, row=row, col=args.col, seed=s, expected_labels_sha256=sha,
            )
            other_preds = [int(p) for p in doc["preds"]]
            res = paired_block_bootstrap_macro_f1(
                proposed_preds, other_preds, labels, blocks, num_classes,
                B=args.B, seed=args.seed, name_a=_PROPOSED_ROW, name_b=row,
                block_description=block_description,
            )
            by_seed[str(s)] = res
            seed_means[s] = res["macro_f1_b"]

        ordered = sorted(seed_means, key=lambda s: seed_means[s])
        median_seed = ordered[len(ordered) // 2]
        headline = dict(by_seed[str(median_seed)])
        headline["headline_seed"] = int(median_seed)
        deltas = [v["delta"] for v in by_seed.values()]
        comparisons[row] = {
            "deterministic": len(seeds) == 1,
            "seeds": seeds,
            "by_seed": by_seed,
            "headline": headline,
            "delta_range_over_seeds": [round(min(deltas), 4), round(max(deltas), 4)],
            "baseline_macro_f1_range": [round(min(seed_means.values()), 4),
                                        round(max(seed_means.values()), 4)],
        }
        sig = "SIGNIFICANT" if headline["significant"] else "n.s."
        print(f"[55]   vs {row:24s} delta={headline['delta']:+6.2f} "
              f"CI={headline['delta_ci']}  p={headline['p_value']}  {sig}")

    out = {
        "table": args.table,
        "col": args.col,
        "method": (
            f"paired temporal-block bootstrap, B={args.B} resamples with replacement "
            f"of {len(blocks)} contiguous blocks from the shared n={len(labels)} "
            f"{args.split} records; blocks are capped at {args.block_size} records "
            f"and do not cross {args.cluster_key} clusters; identical sampled "
            f"blocks applied to both systems"
        ),
        "bootstrap_unit": "temporal_cluster_block",
        "block_size": int(args.block_size),
        "cluster_key": str(args.cluster_key),
        "cluster_counts": dict(cluster_counts),
        "n_blocks": len(blocks),
        "block_size_summary": {
            "min": min(block_sizes),
            "median": statistics.median(block_sizes),
            "max": max(block_sizes),
        },
        "rng_seed": args.seed,
        "zero_support_convention": (
            "count absent/zero-TP class as F1=0 over a fixed denominator of "
            f"{num_classes} classes (matches src.baselines._common.macro_f1)"
        ),
        "n": len(labels),
        "labels_sha256": sha,
        "proposed": {
            "row": _PROPOSED_ROW, "seed_used": args.proposed_seed, "deterministic": True,
            "macro_f1": proposed_mf1,
            "ci_95": proposed_ci,
        },
        "comparisons": comparisons,
        "metadata": {
            "commit": git_commit(),
            "dataset_version": dataset_version(),
            "B": args.B,
        },
    }

    col_slug = str(args.col).replace("%", "pct").replace("/", "_")
    per_col_path = table_dir / f"bootstrap_significance_{col_slug}.json"
    dump_json(per_col_path, out)
    out_path = table_dir / "bootstrap_significance.json"
    dump_json(out_path, out)
    print(f"\n[55] wrote {per_col_path} (+ canonical {out_path.name} for col {args.col})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
