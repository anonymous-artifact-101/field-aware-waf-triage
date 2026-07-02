
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines import get_runner
from src.baselines._common import git_commit, results_path
from src.eval.aggregate import aggregate_table
from src.utils.config import load_config
from src.utils.runlog import append_run

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Table 3 baseline; write results.json.")
    parser.add_argument("--config", required=True, help="configs/baselines/<x>.yaml path.")
    parser.add_argument("--seed", type=int, default=42, help="Seed (paper set: 42..46).")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap records per split (earliest contiguous; for fast smoke runs only).",
    )
    parser.add_argument("--device", default=None, help="Force device for torch baselines ('cpu'/'cuda').")
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override the config's data.label_budget (e.g. 0.10 for the 10%% column). "
             "The runner derives the cell column from the budget, so the same config can "
             "produce the 5%% and 10%% Table-3 columns without per-budget config clones.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Fit and report but do not write the cell file (dry run).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="After this run, (re)build results/table_03_rq1_baselines/results.json from "
        "ALL accumulated cell files (mean + 95%% CI). Normally run once at sweep end.",
    )
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config))
    baseline = str(cfg.get("baseline", ""))
    if not baseline:
        raise SystemExit(f"config {args.config} has no 'baseline' key; cannot dispatch a runner.")

    if args.budget is not None:
        data = dict(cfg.get("data", {}))
        data["label_budget"] = float(args.budget)
        cfg["data"] = data

    if args.max_records is not None:
        cfg["eval_max_records"] = int(args.max_records)

    runner = get_runner(baseline)

    kwargs = {"max_records": args.max_records, "write": not args.no_write}
    if baseline in {"char_cnn", "deeplog", "logbert"}:
        kwargs["device"] = args.device

    result = runner(cfg, args.seed, **kwargs)

    value = result.get("value")
    rp = result.get("results_path")
    print(f"[20_run_baseline] baseline={baseline} seed={args.seed} value={value}")
    if rp:
        print(f"[20_run_baseline] wrote {rp} (+ per-seed cell file under cells/)")
    else:
        print("[20_run_baseline] (nothing written: --no-write)")

    if not args.no_write:
        try:
            append_run(
                stage=f"baseline_{baseline}", config_path=args.config, seed=int(args.seed),
                artifact=str(rp or ""),
                notes=f"value={value}",
                extra={"eval_capped": args.max_records is not None},
            )
        except Exception as exc:
            print(f"[20_run_baseline] WARN: ledger append failed: {exc}")

    if args.aggregate:
        if args.no_write:
            print("[20_run_baseline] WARN: --aggregate ignored under --no-write (no cell files).")
        else:

            table_dir = results_path(cfg).parent
            agg_path = aggregate_table(table_dir, commit=git_commit(), gpu="cpu")
            print(f"[20_run_baseline] aggregated -> {agg_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
