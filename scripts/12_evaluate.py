
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit
from src.detector.evaluate import evaluate_detector
from src.eval.aggregate import aggregate_table, write_cell_file
from src.eval.predictions_io import dump_predictions, dump_shared_labels
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR, ROOT
from src.utils.runlog import append_run, dataset_version, embedding_version
from src.utils.seeds import set_seed

_TABLE_DIRS = {
    "3": "table_03_rq1_baselines",
    "5": "table_05_field_granularity",
    "6": "table_06_per_class_owasp",
}

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the FastText detector; write a results cell.")
    parser.add_argument("--config", required=True, help="Label-budget or ablation YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed (paper set: 42..46).")
    parser.add_argument("--table", default="3", choices=sorted(_TABLE_DIRS), help="Paper table (default: 3).")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap records per split (earliest contiguous; for fast smoke runs only).",
    )
    parser.add_argument("--no-write", action="store_true", help="Evaluate but do not write a cell (dry run).")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="After this run, (re)build results/<table>/results.json from ALL "
        "accumulated cell files (mean + 95%% CI).",
    )
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    table = _TABLE_DIRS[args.table]
    table_dir = RESULTS_DIR / table

    result = evaluate_detector(cfg, seed, max_records=args.max_records)
    row, col, value = result["row"], result["col"], result["value"]
    metric, mode = result["metric"], result["mode"]

    print(f"[12_evaluate] table={table} row={row!r} col={col!r} seed={seed} {metric}={value}")

    if args.no_write:
        print("[12_evaluate] (nothing written: --no-write)")
        return 0

    metadata = {
        "commit": git_commit(),

        "gpu": "cpu",

        "dataset_version": dataset_version(),
        "embedding_version": embedding_version(),
        "metric": metric,
        "mode": mode,

        "checkpoint_loaded": True,
        "exploratory_single_seed": True,
        "detector": "fasttext_field_aware",
        "granularity": cfg.get("granularity"),
        "n_test": result.get("n_test"),

        "eval_capped": args.max_records is not None,
        "max_records": args.max_records,
    }
    if mode == "supervised":
        metadata["n_train_labeled"] = result.get("n_train_labeled")
        metadata["per_class_f1"] = {k: round(float(v), 6) for k, v in result["per_class_f1"].items()}

        metadata["deterministic"] = True

        if result.get("weighted_f1") is not None:
            metadata["weighted_f1"] = result["weighted_f1"]
        if result.get("class_support") is not None:
            metadata["class_support"] = {k: int(v) for k, v in result["class_support"].items()}

        for key in ("macro_precision", "macro_recall", "accuracy"):
            if result.get(key) is not None:
                metadata[key] = result[key]

        if result.get("train_macro_f1") is not None:
            metadata["train_macro_f1"] = result["train_macro_f1"]
        if result.get("per_class_recall") is not None:
            metadata["per_class_recall"] = {k: round(float(v), 4)
                                            for k, v in result["per_class_recall"].items()}
        if result.get("per_class_precision") is not None:
            metadata["per_class_precision"] = {k: round(float(v), 4)
                                               for k, v in result["per_class_precision"].items()}
        if result.get("confusion_matrix") is not None:
            metadata["confusion_matrix"] = result["confusion_matrix"]
    else:
        metadata["score_summary"] = {k: round(float(v), 6) if isinstance(v, float) else v
                                     for k, v in result["score_summary"].items()}

    cell_path = write_cell_file(
        table_dir, table=table, row=row, col=col, seed=int(seed), value=value, metadata=metadata
    )
    print(f"[12_evaluate] wrote {cell_path}")

    try:
        append_run(
            stage=f"evaluate_table{args.table}", config_path=args.config, seed=int(seed),
            artifact=str(cell_path),
            notes=f"row={row}|col={col}|{metric}={value}",
            extra={"eval_capped": args.max_records is not None},
        )
    except Exception as exc:
        print(f"[12_evaluate] WARN: ledger append failed: {exc}")

    if mode == "supervised":
        dump_json(table_dir / "per_class_breakdown.json", {
            "row": row, "col": col, "seed": int(seed),
            "per_class_f1": result["per_class_f1"],
        })

    if mode == "supervised" and args.max_records is None:
        test_labels = result["test_labels"]
        dump_shared_labels(
            table_dir, split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
            labels=test_labels, subtypes=list(SUBTYPES), num_classes=len(SUBTYPES),
            metadata={"commit": git_commit(), "dataset_version": dataset_version()},
        )
        dump_predictions(
            table_dir, row=row, col=col, seed=int(seed),
            split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
            preds=result["preds"], labels=test_labels, num_classes=len(SUBTYPES),
            metric=metric, value=value,
            metadata={"commit": git_commit(), "deterministic": True,
                      "detector": "fasttext_field_aware"},
        )

    if args.aggregate:
        agg = aggregate_table(table_dir, commit=git_commit(), gpu="cpu")
        print(f"[12_evaluate] aggregated -> {agg}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
