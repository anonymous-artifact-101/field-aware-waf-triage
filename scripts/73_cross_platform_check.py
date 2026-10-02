
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import platform
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.detector.evaluate import budget_to_col, evaluate_detector
from src.detector.fasttext_embed import load_fasttext
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR

_BUDGETS = (0.01, 0.05, 0.10, 0.20, 0.50)
_REF_DIR = RESULTS_DIR / "table_21_budget_selection_val" / "cells"

def _reference() -> dict:
    ref = {}
    for f in _REF_DIR.glob("*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        if int(c["seed"]) == 42:
            ref[(c["row"], c["col"])] = float(c["value"])
    return ref

def main() -> int:
    ref = _reference()
    ft = load_fasttext("models/detector/fasttext/weblog_fasttext.model")
    rows = []
    for b in _BUDGETS:
        base = load_config(f"configs/finetune/label_{int(round(b * 100))}pct.yaml")
        col = budget_to_col(b)
        for tag, split in (("val", "owasp_val"), ("test", "owasp_test")):
            cfg = copy.deepcopy(base)
            cfg["data"]["test_split"] = split
            here = round(float(evaluate_detector(cfg, seed=42, fasttext_model=ft)["value"]), 4)
            there = ref.get((f"Proposed ({tag})", col))
            rows.append({"budget": col, "split": tag, "this_platform": here, "windows_reference": there,
                         "abs_diff": None if there is None else round(abs(here - there), 4),
                         "rounded_1dp_changes": None if there is None else
                         f"{here:.1f}" != f"{there:.1f}"})
            print(rows[-1], flush=True)
    diffs = [r["abs_diff"] for r in rows if r["abs_diff"] is not None]
    system = platform.system().lower()
    out = {"platform": platform.platform(), "python": platform.python_version(),
           "cells": rows, "max_abs_diff": max(diffs) if diffs else None,
           "n_rounding_changes": sum(bool(r["rounded_1dp_changes"]) for r in rows),
           "metadata": {"commit": git_commit(), "seed": 42,
                        "reference": "results/table_21_budget_selection_val/cells (Windows 11)"}}
    path = RESULTS_DIR / "diagnostics" / f"cross_platform_{system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({k: out[k] for k in ("platform", "max_abs_diff", "n_rounding_changes")})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
