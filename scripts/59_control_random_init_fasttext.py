
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit
from src.detector.evaluate import budget_to_col, evaluate_detector
from src.detector.fasttext_embed import load_fasttext
from src.eval.aggregate import aggregate_table, write_cell_file
from src.eval.predictions_io import dump_predictions, dump_shared_labels
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version
from src.utils.seeds import set_seed

_TABLE = "table_03_rq1_baselines"
_ROW = "FastText random-init (no pre-train)"

def _randomize_fasttext(model, seed: int):
    wv = model.wv
    rng = np.random.default_rng(int(seed))
    dim = int(wv.vector_size)
    for name in ("vectors_vocab", "vectors_ngrams"):
        if hasattr(wv, name):
            arr = getattr(wv, name)
            arr[:] = ((rng.random(arr.shape, dtype=np.float32) - np.float32(0.5))
                      / np.float32(dim)).astype(np.float32)
    if hasattr(wv, "adjust_vectors"):
        wv.adjust_vectors()
    return model

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Random-init FastText control -> Table 3 cells (N3).")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--config", default="configs/finetune/label_10pct.yaml")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config))
    data = dict(cfg.get("data", {})); data["label_budget"] = float(args.budget); cfg["data"] = data
    col = budget_to_col(float(args.budget))
    model_path = str(cfg.get("fasttext", {}).get(
        "model_path", "models/detector/fasttext/weblog_fasttext.model"))
    table_dir = RESULTS_DIR / _TABLE
    commit = git_commit()

    last = None
    for s in args.seeds:
        set_seed(int(s))
        model = load_fasttext(model_path)
        _randomize_fasttext(model, int(s))
        res = evaluate_detector(dict(cfg), seed=int(s), max_records=args.max_records,
                                fasttext_model=model)
        val = float(res["value"]); last = val
        print(f"[59] random-init FastText @{col} seed {s}: macro_f1={val}")
        if args.no_write:
            continue
        meta = {
            "commit": commit, "date": None, "gpu": "cpu",
            "dataset_version": dataset_version(),
            "metric": "macro_f1", "label_budget": float(args.budget),
            "n_train_labeled": res.get("n_train_labeled"), "n_test": res.get("n_test"),
            "control_role": "self-supervised-pre-training control: proposed pipeline with the "
                            "FastText embedding re-randomized in place (vocab+buckets kept); "
                            "(trained proposed) - (this) = pre-training accuracy gain.",
            "weighted_f1": res.get("weighted_f1"),
            "eval_capped": args.max_records is not None,
            "exploratory_single_seed": False,
        }
        write_cell_file(table_dir, table=_TABLE, row=_ROW, col=col, seed=int(s),
                        value=round(val, 4), metadata=meta)
        if res.get("preds") is not None and res.get("test_labels") is not None and args.max_records is None:
            dump_shared_labels(table_dir, split="owasp_test", labels=res["test_labels"],
                               subtypes=list(SUBTYPES), num_classes=len(SUBTYPES),
                               metadata={"commit": commit})
            dump_predictions(table_dir, row=_ROW, col=col, seed=int(s), split="owasp_test",
                             preds=res["preds"], labels=res["test_labels"], num_classes=len(SUBTYPES),
                             metric="macro_f1", value=round(val, 4),
                             metadata={"commit": commit, "control": "random_init_fasttext",
                                       "deterministic": True})

    if args.aggregate and not args.no_write:
        aggregate_table(table_dir, commit=commit, gpu="cpu")
        print(f"[59] aggregated {table_dir / 'results.json'}")
    print(f"[59] done (last value {last}).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
