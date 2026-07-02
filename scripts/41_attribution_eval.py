
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.data.labels import SUBTYPES
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.eval.aggregate import aggregate_table, write_cell_file
from src.eval.attribution_eval import evaluate_attribution
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import append_run, dataset_version, embedding_version
from src.utils.seeds import set_seed

_TABLE = "table_07_attribution"
_MACRO_ROW = "Macro (mean)"

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory field-saliency agreement (paper Table 7).")
    parser.add_argument("--config", default="configs/ablation/six_field.yaml",
                        help="6-field detector config (attribution needs 6-field granularity).")
    parser.add_argument("--seed", type=int, default=None, help="Seed (paper set: 42..46).")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per split (fast SMOKE only; never paper numbers).")
    parser.add_argument("--truth", default=None, help="Override attribution_truth.yaml path.")
    parser.add_argument("--no-write", action="store_true", help="Evaluate but write nothing.")
    parser.add_argument("--aggregate", action="store_true",
                        help="After this run, rebuild results.json from all cell files (mean+95% CI).")
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    if int(cfg.get("granularity", 6)) != 6:
        raise SystemExit(
            f"[41_attribution_eval] config {args.config} is not 6-field "
            f"(granularity={cfg.get('granularity')}); attribution needs 6-field.")

    ft_model, train_recs, test_recs = load_detector_inputs(cfg, max_records=args.max_records)
    detector = fit_detector(cfg, ft_model, train_recs, seed)

    out = evaluate_attribution(detector, test_recs, truth_path=args.truth)
    macro = round(100.0 * float(out["macro_agreement"]), 4)
    per_sub = {s: out["per_subtype"][s] for s in SUBTYPES}
    populated = out["per_subtype_populated"]
    support = out["support"]
    macro_excl = round(100.0 * float(out["macro_excl_protocol"]), 4)
    macro_obs = round(100.0 * float(out["macro_field_observable"]), 4)
    macro_qry = round(100.0 * float(out["macro_query_borne"]), 4)

    print(f"[41_attribution_eval] seed={seed} macro(all8)={macro} "
          f"macro(excl protocol)={macro_excl} field-observable={macro_obs} "
          f"query-borne={macro_qry} query_empty_rate={round(100.0*out['query_empty_rate'],1)}% "
          f"n={out['n']} (6-field)")

    if args.no_write:
        print("[41_attribution_eval] (nothing written: --no-write)")
        return 0

    table_dir = RESULTS_DIR / _TABLE
    commit = git_commit()
    data_ver, emb_ver = dataset_version(), embedding_version()
    base_meta = {
        "commit": commit,
        "gpu": "cpu",
        "dataset_version": data_ver,
        "embedding_version": emb_ver,
        "metric": "attribution_agreement_pct",
        "mode": "attribution",
        "checkpoint_loaded": True,
        "exploratory_single_seed": True,
        "granularity": 6,
        "eval_capped": args.max_records is not None,
        "max_records": args.max_records,
        "n_test": out["n"],
    }

    n_written = 0

    def _write(row, col, value, *, extra=None):
        nonlocal n_written
        meta = dict(base_meta)
        if extra:
            meta.update(extra)
        write_cell_file(table_dir, table=_TABLE, row=row, col=col,
                        seed=int(seed), value=value, metadata=meta)
        n_written += 1

    strata = out["strata"]
    obs_set, qry_set = set(strata["field_observable"]), set(strata["query_borne"])

    for s in SUBTYPES:
        v = per_sub[s]
        if v != v:
            continue
        group = ("protocol" if s == "protocol"
                 else "field_observable" if s in obs_set else "query_borne")
        _write(s, "agreement", round(100.0 * float(v), 4),
               extra={"support": int(support.get(s, 0)), "stratum": group})
        _write(s, "support", int(support.get(s, 0)),
               extra={"metric": "count"})
        pop = populated.get(s)
        if pop == pop:
            _write(s, "populated", round(100.0 * float(pop), 4),
                   extra={"metric": "populated_pct"})

    _write(_MACRO_ROW, "agreement", macro, extra={"support": int(out["n"])})
    _write("Macro (excl. protocol)", "agreement", macro_excl,
           extra={"note": "7-class macro; protocol catch-all dropped from denominator"})
    _write("Macro (field-observable)", "agreement", macro_obs,
           extra={"note": "payload field recorded by the log (path/ua): method recovery"})
    _write("Macro (query-borne)", "agreement", macro_qry,
           extra={"note": "payload rides in query, truncated in 94% of OWASP records: corpus ceiling"})

    _write("Random-field floor", "agreement", round(float(out["random_field_floor"]) * 100.0, 4),
           extra={"note": "1/6 argmax chance level", "single_run_measured": True})
    _write("Always-path baseline", "agreement",
           round(100.0 * float(out["always_path_macro_excl_protocol"]), 4),
           extra={"note": "degenerate always-predict-path, macro excl. protocol",
                  "single_run_measured": True})

    _write("Query-empty rate", "agreement", round(100.0 * float(out["query_empty_rate"]), 4),
           extra={"note": "fraction of test records with an empty query string",
                  "single_run_measured": True})

    print(f"[41_attribution_eval] wrote {n_written} cell(s) under {table_dir / 'cells'}")

    try:
        append_run(
            stage="attribution_table7", config_path=args.config, seed=int(seed),
            artifact=str(table_dir / "results.json"),
            notes=f"macro_agreement={macro}|n={out['n']}",
            extra={"eval_capped": args.max_records is not None},
        )
    except Exception as exc:
        print(f"[41_attribution_eval] WARN: ledger append failed: {exc}")

    if args.aggregate:
        agg = aggregate_table(table_dir, commit=commit, gpu="cpu")
        print(f"[41_attribution_eval] aggregated -> {agg}")
        print("[41_attribution_eval] render with: python scripts/50_make_paper_tables.py --table 7")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
