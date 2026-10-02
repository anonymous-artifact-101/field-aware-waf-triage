
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit, load_split_records, select_label_budget
from src.detector.evaluate import budget_to_col, evaluate_detector
from src.detector.fasttext_embed import load_fasttext
from src.eval.aggregate import aggregate_table, write_cell_file
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

_TABLE = "table_21_budget_selection_val"
_BUDGETS = (0.01, 0.05, 0.10, 0.20, 0.50)
KNEE_DELTA = 1.0

def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-based budget selection (Table 21).")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--per-class-only", action="store_true",
                        help="only write per_class_breakdown_val_test_10pct.json (seed 42; deterministic)")
    args = parser.parse_args(argv)

    table_dir = RESULTS_DIR / _TABLE
    table_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    meta_common = {"commit": commit, "dataset_version": dataset_version()}

    if args.per_class_only:
        ft = load_fasttext("models/detector/fasttext/weblog_fasttext.model")
        base = load_config("configs/finetune/label_10pct.yaml")
        out = {}
        for tag, split in (("val", "owasp_val"), ("test", "owasp_test")):
            cfg = copy.deepcopy(base)
            cfg["data"]["test_split"] = split
            res = evaluate_detector(cfg, seed=42, fasttext_model=ft)
            out[tag] = {"macro_f1": res["value"],
                        "per_class_f1": {k: round(100 * v, 2) for k, v in res["per_class_f1"].items()},
                        "support": res.get("class_support")}
        _dump(table_dir / "per_class_breakdown_val_test_10pct.json",
              {**out, "metadata": {**meta_common, "label_budget": 0.10, "seed": 42}})
        for c in SUBTYPES:
            print(f"[69] {c:15} val={out['val']['per_class_f1'].get(c)} test={out['test']['per_class_f1'].get(c)}")
        return 0

    train = load_split_records("owasp_train")
    counts = {}
    for b in _BUDGETS:
        idx = select_label_budget(len(train), b)
        c = Counter(train[i]["attack_subtype"] for i in idx)
        counts[budget_to_col(b)] = {"n_labeled": len(idx), **{s: int(c.get(s, 0)) for s in SUBTYPES}}
    _dump(table_dir / "budget_class_counts.json",
          {"counts": counts, "metadata": {**meta_common, "source": "owasp_train earliest prefix",
                                          "selector": "src.baselines._common.select_label_budget"}})

    ft = load_fasttext("models/detector/fasttext/weblog_fasttext.model")
    val_by_budget = {}
    for b in _BUDGETS:
        base = load_config(f"configs/finetune/label_{int(round(b * 100))}pct.yaml")
        col = budget_to_col(b)
        for s in args.seeds:
            for tag, split in (("val", "owasp_val"), ("test", "owasp_test")):
                cfg = copy.deepcopy(base)
                cfg["data"]["test_split"] = split
                v = round(float(evaluate_detector(cfg, seed=int(s), fasttext_model=ft)["value"]), 4)
                print(f"[69] {col} {tag} seed {s}: macro_f1={v}", flush=True)
                if tag == "val":
                    val_by_budget.setdefault(col, []).append(v)
                write_cell_file(table_dir, table=_TABLE, row=f"Proposed ({tag})", col=col,
                                seed=int(s), value=v,
                                metadata={**meta_common, "date": None, "gpu": "cpu",
                                          "metric": "macro_f1", "label_budget": b, "eval_split": split})

    cols = [budget_to_col(b) for b in _BUDGETS]
    mean = {c: sum(val_by_budget[c]) / len(val_by_budget[c]) for c in cols}
    gains = {f"{a}->{b}": round(mean[b] - mean[a], 4) for a, b in zip(cols, cols[1:])}
    knee = next((a for a, b in zip(cols, cols[1:]) if mean[b] - mean[a] < KNEE_DELTA), cols[-1])
    _dump(table_dir / "selection.json", {
        "rule": f"smallest budget whose next budget adds < {KNEE_DELTA} val macro-F1",
        "val_mean_macro_f1": {c: round(mean[c], 4) for c in cols},
        "val_gain_to_next_budget": gains,
        "knee_budget": knee,
        "argmax_val_budget": max(cols, key=lambda c: mean[c]),
        "status": "rule stated after test results were seen; test significance at the "
                  "knee budget is exploratory / post-selection",
        "metadata": meta_common,
    })
    print(f"[69] knee={knee} argmax={max(cols, key=lambda c: mean[c])} gains={gains}")
    if args.aggregate:
        aggregate_table(table_dir, commit=commit, gpu="cpu")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
