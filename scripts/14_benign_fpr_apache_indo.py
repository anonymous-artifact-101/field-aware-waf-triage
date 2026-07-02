
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit, load_split_records
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.eval.aggregate import load_cell_files, write_cell_file
from src.eval.metrics import mean_ci95
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR, ROOT
from src.utils.runlog import append_run, dataset_version, embedding_version
from src.utils.seeds import set_seed

_TABLE = "table_11_benign_fpr_apache_indo"
_APACHE_INDO = ROOT / "data" / "processed" / "apache_indo_parsed" / "apache_indo.jsonl"
_PERCENTILES = (95.0, 99.0, 99.9)
_HEADLINE_PCT = 99.0

def _load_benign_apache_indo(max_records: Optional[int] = None
                             ) -> "tuple[List[Dict[str, Any]], int]":
    import json
    if not _APACHE_INDO.is_file():
        raise SystemExit(
            f"[14_benign_fpr] parsed Apache-Indo not found at {_APACHE_INDO}; "
            f"parse it first (src/data/parsers/apache_indo.py)."
        )
    recs: List[Dict[str, Any]] = []
    n_flagged = 0
    with _APACHE_INDO.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_records is not None and len(recs) >= max_records:
                break
            d = json.loads(line)
            if str(d.get("label", "")).lower() == "benign":
                recs.append(d)
            else:
                n_flagged += 1
    return recs, n_flagged

def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[lo]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac

def _aggregate(commit: str) -> Path:
    table_dir = RESULTS_DIR / _TABLE
    cells = load_cell_files(table_dir)
    if not cells:
        raise SystemExit(f"[14_benign_fpr] no cells under {table_dir/'cells'}; "
                         f"run seeds 42..46 first.")
    series: Dict[str, List[float]] = {}
    seeds: List[int] = []
    embs = set()
    last_md: Dict[str, Any] = {}
    for c in cells:
        col = str(c["col"])
        series.setdefault(col, []).append(float(c["value"]))
        seeds.append(int(c["seed"]))
        md = c.get("metadata", {})
        embs.add(md.get("embedding_version"))
        last_md = md
    if len(embs) > 1:
        raise SystemExit(f"[14_benign_fpr] cells span multiple embeddings {embs}; "
                         f"re-run all seeds on the current embedding before aggregating.")
    seeds = sorted(set(seeds))
    out_cells = []
    for col, vals in series.items():
        m, (lo, hi) = mean_ci95(vals)
        out_cells.append({"row": "Proposed (zero-shot benign FPR)", "col": col,
                          "value": round(m, 4), "ci_95": [round(lo, 4), round(hi, 4)],
                          "seeds": seeds})
    doc = {
        "table": _TABLE,
        "cells": out_cells,
        "metadata": {
            "commit": commit, "date": "", "gpu": "cpu",
            "ci_method": "student_t", "n_cell_files": len(cells),
            "dataset": "apache_indo_2020", "single_run_measured": False,
            "dataset_version": last_md.get("dataset_version"),
            "embedding_version": embs.pop() if embs else None,
            "n_benign": last_md.get("n_benign"),
            "n_flagged_excluded": last_md.get("n_flagged_excluded"),
            "headline_percentile": _HEADLINE_PCT,
            "note": "Zero-shot benign FPR on Apache-Indo AMAN subset. Scorer AND "
                    "operating threshold fit on the benign Kaggle weblog (no "
                    "Apache-Indo fit); FPR = fraction of AMAN records above the "
                    "weblog Pth-percentile threshold. Labels are heuristic, so only "
                    "AMAN (benign) is used; BAHAYA/DICURIGAI are excluded, never an "
                    "attack class. Deterministic in seed -> zero-width CI.",
        },
    }
    out = table_dir / "results.json"
    dump_json(out, doc)
    return out

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-shot benign FPR on Apache-Indo.")
    parser.add_argument("--config", default="configs/finetune/label_0pct.yaml",
                        help="Unsupervised detector config (0% label, centroid scorer).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None, help="Cap (smoke only).")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--aggregate", action="store_true",
                        help="Collect per-seed cells -> results.json and exit.")
    args = parser.parse_args(argv)

    if args.aggregate:
        out = _aggregate(git_commit())
        print(f"[14_benign_fpr] aggregated results -> {out}")
        return 0

    cfg = dict(load_config(args.config))
    seed = int(args.seed)
    set_seed(seed)

    ft_model, train_recs, _ = load_detector_inputs(cfg, max_records=args.max_records)
    detector = fit_detector(cfg, ft_model, train_recs, seed)
    if detector.mode != "unsupervised":
        raise SystemExit("[14_benign_fpr] use the 0%-label (unsupervised) config.")

    weblog_scores = sorted(float(s) for s in detector.anomaly_score(train_recs))
    thresholds = {pct: _percentile(weblog_scores, pct) for pct in _PERCENTILES}

    benign_recs, n_flagged = _load_benign_apache_indo(max_records=args.max_records)
    indo_scores = [float(s) for s in detector.anomaly_score(benign_recs)]
    n = len(indo_scores)

    fpr_at: Dict[str, float] = {}
    for pct, thr in thresholds.items():
        n_flag = sum(1 for s in indo_scores if s > thr)
        fpr_at[f"FPR@p{pct:g}"] = (n_flag / n) if n else float("nan")

    print(f"[14_benign_fpr] Apache-Indo AMAN benign n={n} "
          f"(excluded {n_flagged} heuristic-flagged)")
    for pct in _PERCENTILES:
        col = f"FPR@p{pct:g}"
        print(f"  weblog threshold p{pct:g}={thresholds[pct]:.4f} -> {col}={fpr_at[col]:.4f}")

    if args.no_write:
        return 0

    table_dir = RESULTS_DIR / _TABLE
    base_md = {
        "commit": git_commit(), "gpu": "cpu",
        "dataset_version": dataset_version(), "embedding_version": embedding_version(),
        "metric": "false_positive_rate", "mode": "unsupervised_zero_shot_benign_fpr",
        "checkpoint_loaded": True, "exploratory_single_seed": True,
        "eval_capped": args.max_records is not None,
        "n_benign": n, "n_flagged_excluded": n_flagged,
        "weblog_thresholds": {f"p{p:g}": round(thresholds[p], 4) for p in _PERCENTILES},
        "note": "Zero-shot benign FPR; scorer+threshold from benign weblog, AMAN-only.",
    }
    for pct in _PERCENTILES:
        col = f"FPR@p{pct:g}"
        write_cell_file(table_dir, table=_TABLE, row="Proposed (zero-shot benign FPR)",
                        col=col, seed=seed, value=round(fpr_at[col], 4), metadata=base_md)
    print(f"[14_benign_fpr] wrote {len(_PERCENTILES)} cells under {table_dir/'cells'}")

    try:
        append_run(stage="benign_fpr_apache_indo", config_path=args.config, seed=seed,
                   artifact=str(table_dir),
                   notes=f"FPR@p{_HEADLINE_PCT:g}={round(fpr_at[f'FPR@p{_HEADLINE_PCT:g}'],4)}|n={n}",
                   extra={"eval_capped": args.max_records is not None})
    except Exception as exc:
        print(f"[14_benign_fpr] WARN ledger: {exc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
