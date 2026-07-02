
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import dump_json, load_json
from src.utils.paths import RESULTS_DIR, ensure_dir

_FIG_DIR = "figure_04_learning_curve"
_ACC_TABLE = "table_03_rq1_baselines"
_PROPOSED = "Proposed (FastText field-aware)"

_BUDGETS = ["1%", "5%", "10%", "20%", "50%"]
_SEED = 42

def _cell_path(budget: str) -> Path:
    b = budget.rstrip("%")
    return RESULTS_DIR / _ACC_TABLE / "cells" / f"proposed-fasttext-field-aware__{b}__seed{_SEED}.json"

def _collect() -> List[Dict[str, Optional[float]]]:
    points: List[Dict[str, Optional[float]]] = []
    for budget in _BUDGETS:
        path = _cell_path(budget)
        if not path.is_file():
            print(f"[51_make_learning_curve_figure] missing cell for {budget}: {path}")
            continue
        doc = load_json(path)
        meta = doc.get("metadata", {})
        test = doc.get("value")
        train = meta.get("train_macro_f1")
        if test is None or train is None:
            print(f"[51_make_learning_curve_figure] {budget}: missing test/train "
                  f"(test={test}, train={train}); re-run scripts/12_evaluate.py")
            continue
        points.append({
            "budget": budget,
            "budget_frac": float(budget.rstrip("%")) / 100.0,
            "n_train_labeled": meta.get("n_train_labeled"),
            "train_macro_f1": round(float(train), 4),
            "test_macro_f1": round(float(test), 4),
            "gap": round(float(train) - float(test), 4),
        })
    return points

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="Override the figure output dir.")
    args = parser.parse_args()

    points = _collect()
    if not points:
        print("[51_make_learning_curve_figure] no usable cells; nothing written.")
        return 1

    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (RESULTS_DIR / _FIG_DIR))
    fig_data = {
        "figure": _FIG_DIR,
        "x_col": "label budget (% of time-ordered train prefix)",
        "y_col": "macro_f1 (%)",
        "series": {
            "train": "re-substitution macro-F1 on the labeled fit prefix",
            "test": "macro-F1 on the full OWASP test split",
        },
        "points": points,
        "note": ("Single seed (42); the supervised detector is deterministic in the "
                 "seed (earliest-prefix budget + linear head), so the curve is exact, "
                 "not seed-averaged -- consistent with the deterministic flag on the "
                 "Table 3 cells. Train is re-substitution macro-F1 on the labeled fit "
                 "prefix; the train-test gap is the Low-label generalization gap and "
                 "narrows as the budget grows."),
        "source": {"per_budget_cells": f"results/{_ACC_TABLE}/cells/"
                                        f"proposed-fasttext-field-aware__<budget>__seed{_SEED}.json"},
    }
    data_path = out_dir / "figure_data.json"
    dump_json(data_path, fig_data)
    print(f"[51_make_learning_curve_figure] wrote {data_path} ({len(points)} budgets)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [p["budget_frac"] * 100 for p in points]
        tr = [p["train_macro_f1"] for p in points]
        te = [p["test_macro_f1"] for p in points]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, tr, "o-", label="Training (fit prefix)", color="#1f77b4")
        ax.plot(xs, te, "s-", label="Test (held-out)", color="#d62728")
        ax.fill_between(xs, te, tr, alpha=0.08, color="gray")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([p["budget"] for p in points])
        ax.set_xlabel("Label budget (% of train prefix, log scale)")
        ax.set_ylabel("macro-F1 (%)")
        ax.set_title("Learning curve: training vs. test macro-F1")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, which="both", alpha=0.2)
        png_path = out_dir / "learning_curve.png"
        fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)
        print(f"[51_make_learning_curve_figure] wrote {png_path}")
    except ImportError:
        print("[51_make_learning_curve_figure] matplotlib not installed; wrote figure_data.json only.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
