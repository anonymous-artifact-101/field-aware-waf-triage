
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import (
    SUBTYPES,
    git_commit,
    load_split_records,
    macro_f1,
    subtype_labels,
    weighted_f1,
)
from src.detector.classifier import build_detector
from src.detector.evaluate import load_detector_inputs
from src.eval.metrics import mean_ci95
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR
from src.utils.provenance import today_iso
from src.utils.runlog import append_run
from src.utils.seeds import set_seed

_LOGGER = logging.getLogger("18_prefix_resample_ci")
TABLE_DIR = RESULTS_DIR / "table_14_budget_composition_ci"

def _window_offsets(n_train: int, k: int, n_windows: int) -> List[int]:
    if k >= n_train:
        return [0]
    last_start = n_train - k
    if n_windows <= 1:
        return [0]
    step = last_start / (n_windows - 1)
    offsets = sorted({int(round(i * step)) for i in range(n_windows)})
    return offsets

def _fit_eval_window(cfg, ft_model, train_recs, test_recs, start, k, seed) -> Dict[str, float]:
    set_seed(int(seed))
    cfg = dict(cfg)
    cfg["seed"] = int(seed)
    detector = build_detector(cfg, ft_model)
    window = train_recs[start:start + k]
    detector.fit(window, subtype_labels(window))
    preds = detector.predict(test_recs)
    gold = subtype_labels(test_recs)
    return {
        "macro_f1": round(100.0 * macro_f1(preds, gold, len(SUBTYPES)), 4),
        "weighted_f1": round(100.0 * weighted_f1(preds, gold, len(SUBTYPES)), 4),
    }

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Budget-composition CI via sliding labeled windows.")
    parser.add_argument("--config", default="configs/finetune/label_5pct.yaml",
                        help="Supervised detector config (the budget is read from it / --budget).")
    parser.add_argument("--budget", type=float, default=None,
                        help="Labeled window fraction (default: the config's data.label_budget).")
    parser.add_argument("--n-windows", type=int, default=8, help="How many window offsets to slide.")
    parser.add_argument("--seed", type=int, default=42, help="Estimator seed (held fixed across windows).")
    parser.add_argument("--max-records", type=int, default=None, help="Cap (smoke only).")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config)
    budget = float(args.budget if args.budget is not None
                   else cfg.get("data", {}).get("label_budget", 0.05))
    col = f"{round(budget * 100)}%"

    ft_model, train_recs, test_recs = load_detector_inputs(cfg, max_records=args.max_records)
    n_train = len(train_recs)
    k = max(1, int(round(budget * n_train)))
    offsets = _window_offsets(n_train, k, args.n_windows)
    _LOGGER.info(
        "budget=%.3f window_size=%d/%d train, test n=%d, sliding %d windows at offsets %s",
        budget, k, n_train, len(test_recs), len(offsets), offsets,
    )

    per_window: List[float] = []
    details: List[Dict[str, Any]] = []
    for start in offsets:
        res = _fit_eval_window(cfg, ft_model, train_recs, test_recs, start, k, args.seed)
        per_window.append(res["macro_f1"])
        frac = round(start / n_train, 4)
        details.append({"start_index": start, "start_frac": frac, **res})
        _LOGGER.info("  window start=%5d (%.1f%% into train) -> macro_f1=%.2f",
                     start, frac * 100, res["macro_f1"])

    ci_mean, (ci_lo, ci_hi) = mean_ci95(per_window)
    if len(per_window) > 1:
        m = sum(per_window) / len(per_window)
        ci_std = (sum((v - m) ** 2 for v in per_window) / (len(per_window) - 1)) ** 0.5
    else:
        ci_std = 0.0
    earliest = per_window[0]
    _LOGGER.info(
        "budget-composition over %d windows: mean=%.2f 95%%CI=[%.2f, %.2f] (std=%.2f); "
        "canonical earliest-prefix point=%.2f",
        len(per_window), ci_mean, ci_lo, ci_hi, ci_std, earliest,
    )

    if args.no_write:
        print("[18_prefix_resample_ci] (nothing written: --no-write)")
        return 0

    table_name = TABLE_DIR.name
    meta = {
        "commit": git_commit(),
        "date": today_iso(),
        "gpu": "cpu",
        "budget": budget,
        "n_windows": len(per_window),
        "window_size": k,
        "n_train": n_train,
        "n_test": len(test_recs),
        "window_macro_f1": per_window,
        "window_details": details,
        "earliest_prefix_macro_f1": earliest,
        "ci_kind": "budget_composition_over_sliding_windows",
        "note": ("CI is over sliding contiguous LABELED windows within the "
                 "time-ordered train split (every window precedes the test set, "
                 "so no leakage); it reports budget-composition variance, which "
                 "the seed-only CI (deterministic prefix) cannot capture. The "
                 "earliest-prefix point is the value reported in Table 3."),
    }

    new_cell = {
        "row": "Proposed (FastText field-aware)",
        "col": col,
        "value": round(ci_mean, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "seeds": [int(args.seed)],
        "metadata": {**meta, "std": round(ci_std, 4),
                     "earliest_prefix": earliest, "single_run_measured": False},
    }
    results_path = TABLE_DIR / "results.json"
    if results_path.exists():
        import json as _json
        doc = _json.loads(results_path.read_text(encoding="utf-8"))
        cells = [c for c in doc.get("cells", [])
                 if not (c["row"] == new_cell["row"] and c["col"] == new_cell["col"])]
    else:
        cells = []
    cells.append(new_cell)
    cells.sort(key=lambda c: (c["row"], c["col"]))
    dump_json(results_path, {
        "table": table_name,
        "cells": cells,
        "metadata": {"commit": git_commit(), "date": today_iso(), "gpu": "cpu",
                     "ci_kind": "budget_composition_over_sliding_windows"},
    })
    append_run(
        stage="budget_composition_ci_table14",
        config_path=args.config,
        seed=int(args.seed),
        command_line=f"scripts/18_prefix_resample_ci.py --budget {budget} --n-windows {args.n_windows} --seed {args.seed}",
        artifact=str(TABLE_DIR),
        notes=f"mean={round(ci_mean,4)} ci={[round(ci_lo,4), round(ci_hi,4)]} earliest={earliest} windows={len(per_window)}",
    )
    dump_json(TABLE_DIR / "budget_composition_ci.json", {
        "budget": budget, "col": col, "windows": details,
        "mean": round(ci_mean, 4), "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "std": round(ci_std, 4), "earliest_prefix": earliest, "metadata": meta,
    })
    print(f"[18_prefix_resample_ci] wrote budget-composition CI under {TABLE_DIR}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
