
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit, load_split_records, select_label_budget
from src.detector.evaluate import budget_to_col, evaluate_detector
from src.detector.fasttext_embed import load_fasttext, train_fasttext
from src.eval.aggregate import aggregate_table, write_cell_file
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version
from src.utils.seeds import set_seed

_TABLE = "table_20_pretrain_corpus_control"
_ENCODINGS = (("flat", 1), ("6-field", 6))
_SPLITS = (("test", "owasp_test"), ("val", "owasp_val"))

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Embedding-corpus x granularity control (Table 20).")
    parser.add_argument("--budgets", nargs="*", type=float, default=[0.05, 0.10])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed-summary", action="store_true",
                        help="only write seed_summary.json (mean/std/min/max over seeds) from existing cells")
    args = parser.parse_args(argv)

    table_dir = RESULTS_DIR / _TABLE
    commit = git_commit()

    if args.seed_summary:
        import json
        import statistics
        groups = {}
        for f in sorted((table_dir / "cells").glob("*.json")):
            c = json.loads(f.read_text(encoding="utf-8"))
            groups.setdefault(f"{c['row']} | {c['col']}", []).append(float(c["value"]))
        summary = {k: {"n_seeds": len(v), "mean": round(statistics.mean(v), 4),
                       "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                       "min": round(min(v), 4), "max": round(max(v), 4)}
                   for k, v in sorted(groups.items())}
        (table_dir / "seed_summary.json").write_text(json.dumps(
            {"cells": summary, "metadata": {
                "commit": commit, "note": "Benign-weblog rows use the single deployed embedding "
                "(std 0); labeled-prefix rows re-fit FastText per seed (seed controls the "
                "FastText fit; the LinearSVC head and prefix are deterministic)."}},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1))
        return 0
    train = load_split_records("owasp_train", max_records=args.max_records)
    benign = load_fasttext("models/detector/fasttext/weblog_fasttext.model")

    for budget in args.budgets:
        base = load_config(f"configs/finetune/label_{int(round(budget * 100))}pct.yaml")
        base["data"]["label_budget"] = float(budget)
        pct = budget_to_col(budget)
        prefix = [train[i] for i in select_label_budget(len(train), budget)]
        for s in args.seeds:
            set_seed(int(s))
            prefix_ft = train_fasttext(prefix, base, seed=int(s))
            for corpus, model in (("Benign weblog", benign), ("Labeled prefix", prefix_ft)):
                for enc_name, gran in _ENCODINGS:
                    for split_name, split in _SPLITS:
                        cfg = copy.deepcopy(base)
                        cfg["granularity"] = gran
                        cfg.setdefault("model", {})["num_fields"] = gran
                        cfg["data"]["test_split"] = split
                        res = evaluate_detector(cfg, seed=int(s), max_records=args.max_records,
                                                fasttext_model=model)
                        row, col = f"{corpus} / {enc_name}", f"{pct} {split_name}"
                        val = round(float(res["value"]), 4)
                        print(f"[68] {row} | {col} | seed {s}: macro_f1={val}", flush=True)
                        if args.no_write:
                            continue
                        write_cell_file(
                            table_dir, table=_TABLE, row=row, col=col, seed=int(s), value=val,
                            metadata={
                                "commit": commit, "date": None, "gpu": "cpu",
                                "dataset_version": dataset_version(), "metric": "macro_f1",
                                "label_budget": float(budget), "eval_split": split,
                                "embedding_corpus": ("weblog_pretrain (fit once, seed 42)"
                                                     if corpus == "Benign weblog"
                                                     else "owasp_train labeled prefix text (no labels), seeded"),
                                "granularity": gran,
                                "labeled_prefix_vocab": len(prefix_ft.wv.key_to_index),
                                "eval_capped": args.max_records is not None,
                            })

    if args.aggregate and not args.no_write:
        aggregate_table(table_dir, commit=commit, gpu="cpu")
        print(f"[68] aggregated {table_dir / 'results.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
