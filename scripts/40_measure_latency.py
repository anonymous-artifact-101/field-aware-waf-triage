
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.eval.aggregate import aggregate_table, write_cell_file
from src.eval.latency import measure_latency
from src.eval.efficiency import (
    measure_load_time,
    measure_memory,
    measure_throughput,
    psutil_available,
    repeat_latency,
)
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR, ROOT
from src.utils.runlog import append_run, dataset_version, embedding_version
from src.utils.seeds import set_seed

_TABLE = "table_08_latency"

_STAGES = ("tokenize", "encode", "score", "end_to_end")
_STATS = ("mean", "p50", "p95", "p99")

_FOOTPRINT_ROW = "Footprint (MB)"

_E2E_CI_ROW = "Proposed end-to-end (10-run)"
_THROUGHPUT_ROW = "Throughput (req/s/core)"
_LOADTIME_ROW = "Model load time (ms)"
_MEMORY_ROW = "Resident memory (MB)"

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def _pin_threads(num_threads: int, cap_omp_mkl: bool) -> None:
    if cap_omp_mkl:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, str(int(num_threads)))

def _footprint_mb(cfg: Mapping[str, Any], detector) -> Dict[str, float]:
    ft_path = _resolve(str(cfg.get("fasttext", {}).get(
        "model_path", "models/detector/fasttext/weblog_fasttext.model")))
    ft_bytes = 0
    if ft_path.is_file():

        for sib in ft_path.parent.glob(ft_path.name + "*"):
            if sib.is_file():
                ft_bytes += sib.stat().st_size

    head_obj = getattr(detector, "clf", None)
    if head_obj is None:
        head_obj = getattr(detector, "anomaly", None)
    head_bytes = len(pickle.dumps(head_obj)) if head_obj is not None else 0

    mb = 1024.0 * 1024.0
    ft_mb = ft_bytes / mb
    head_mb = head_bytes / mb
    return {
        "FastText model": round(ft_mb, 4),
        "Classifier head": round(head_mb, 4),
        "Total": round(ft_mb + head_mb, 4),
    }

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure CPU per-request latency + footprint (paper Table 8).")
    parser.add_argument("--config", default="configs/finetune/label_10pct.yaml",
                        help="Detector config to time (default: the headline supervised 10%).")
    parser.add_argument("--efficiency", default="configs/eval/efficiency.yaml",
                        help="Efficiency protocol config (warmup/n_trials/threads/split).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the (cheap) detector fit (default: config seed).")
    parser.add_argument("--runs", type=int, default=None,
                        help="Independent timing runs for the latency CI "
                             "(default: efficiency.yaml latency.runs or 10).")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per split (fast SMOKE only; never paper numbers).")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip rebuilding results.json after writing cells.")
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config))
    eff = dict(load_config(args.efficiency)).get("efficiency", {})
    lat_cfg = dict(eff.get("latency", {}))

    _fit_budget = float(cfg.get("data", {}).get("label_budget", 0.05))

    num_threads = int(eff.get("num_threads", 1))
    _pin_threads(num_threads, bool(eff.get("cap_omp_mkl_threads", True)))

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    warmup = int(lat_cfg.get("warmup", 20))
    n_trials = int(lat_cfg.get("n_trials", 300))
    split = str(lat_cfg.get("split", "owasp_test"))
    runs = int(args.runs) if args.runs is not None else int(lat_cfg.get("runs", 10))
    tp_cfg = dict(eff.get("throughput", {}))
    tp_n = int(tp_cfg.get("n_records", 2000))
    tp_batch = int((tp_cfg.get("batch_sizes", [1, 32]) or [1, 32])[-1])

    cfg.setdefault("data", {})
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["test_split"] = split
    ft_model, train_recs, test_recs = load_detector_inputs(cfg, max_records=args.max_records)
    detector = fit_detector(cfg, ft_model, train_recs, seed)

    print(f"[40_measure_latency] config={args.config} mode={detector.mode} "
          f"split={split} warmup={warmup} n_trials={n_trials} runs={runs} threads={num_threads}")

    timing = measure_latency(detector, test_recs, warmup=warmup, n_trials=n_trials)
    stages = timing["stages"]

    def _one_run():
        return measure_latency(detector, test_recs, warmup=warmup, n_trials=n_trials)["stages"]

    e2e_ci = repeat_latency(_one_run, runs=runs)
    print(f"[40_measure_latency] {runs}-run end_to_end mean="
          f"{e2e_ci['end_to_end_mean']['mean']:.4f}ms "
          f"+/-{e2e_ci['end_to_end_mean']['ci95']:.4f} (95% CI, std "
          f"{e2e_ci['end_to_end_mean']['std']:.4f})")

    table_dir = RESULTS_DIR / _TABLE
    commit = git_commit()
    data_ver, emb_ver = dataset_version(), embedding_version()

    n_written = 0
    for stage in _STAGES:
        st = stages.get(stage, {})
        for stat in _STATS:
            val = st.get(stat)
            if val is None:
                continue
            metadata = {
                "commit": commit,
                "gpu": "cpu",
                "dataset_version": data_ver,
                "embedding_version": emb_ver,
                "metric": "latency_ms",
                "unit": "milliseconds",
                "single_run_measured": True,
                "eval_capped": args.max_records is not None,
                "max_records": args.max_records,
                "mode": detector.mode,
                "config": args.config,
                "fit_budget": _fit_budget,
                "stage": stage,
                "stat": stat,
                "n_trials": timing.get("n_trials"),
                "warmup": timing.get("warmup"),
                "batch_size": timing.get("batch_size", 1),
                "num_threads": num_threads,
                "split": split,

                "hardware": os.environ.get("PECTI_HARDWARE", "CPU (unspecified; set $PECTI_HARDWARE)"),
            }
            write_cell_file(table_dir, table=_TABLE, row=stage, col=stat,
                            seed=int(seed), value=round(float(val), 6), metadata=metadata)
            n_written += 1

    fp = _footprint_mb(cfg, detector)
    for col, mb in fp.items():
        metadata = {
            "commit": commit,
            "gpu": "cpu",
            "dataset_version": data_ver,
            "embedding_version": emb_ver,
            "metric": "footprint_mb",
            "unit": "megabytes",
            "single_run_measured": True,
            "eval_capped": False,
            "mode": detector.mode,
            "component": col,
            "num_threads": num_threads,
        }
        write_cell_file(table_dir, table=_TABLE, row=_FOOTPRINT_ROW, col=col,
                        seed=int(seed), value=mb, metadata=metadata)
        n_written += 1

    _CI_STAT = {"mean": "end_to_end_mean", "p95": "end_to_end_p95", "p99": "end_to_end_p99"}
    for col, key in _CI_STAT.items():
        agg = e2e_ci[key]
        write_cell_file(
            table_dir, table=_TABLE, row=_E2E_CI_ROW, col=col,
            seed=int(seed), value=round(float(agg["mean"]), 6),
            metadata={
                "commit": commit, "gpu": "cpu", "dataset_version": data_ver,
                "embedding_version": emb_ver, "metric": "latency_ms_multirun",
                "unit": "milliseconds", "single_run_measured": False,
                "ci95_halfwidth": round(float(agg["ci95"]), 6),
                "std": round(float(agg["std"]), 6), "n_runs": int(agg["n"]),
                "config": args.config, "fit_budget": _fit_budget,
                "eval_capped": args.max_records is not None,
                "n_trials": timing.get("n_trials"), "warmup": timing.get("warmup"),
                "num_threads": num_threads, "split": split,
                "hardware": os.environ.get("PECTI_HARDWARE", "CPU (unspecified; set $PECTI_HARDWARE)"),
            },
        )
        n_written += 1

    def _featscore_one(rec):
        feat = detector.encoder.encode_record(rec)
        x = feat.reshape(1, -1)
        if detector.mode == "supervised":
            return detector._proba_from_features(x)
        return detector._anomaly_from_features(x)

    def _batch_score(batch):
        feats = detector.encoder.encode_records(list(batch))
        if detector.mode == "supervised":
            return detector._proba_from_features(feats)
        return detector._anomaly_from_features(feats)

    tp = measure_throughput(_featscore_one, test_recs, warmup=warmup,
                            n_records=tp_n, batch_score=_batch_score, batch_size=tp_batch)
    _tp_cols = {"batch=1": "batch1_records_per_second", f"batch={tp_batch}": "batched_records_per_second"}
    for col, key in _tp_cols.items():
        if tp.get(key) is None:
            continue
        write_cell_file(
            table_dir, table=_TABLE, row=_THROUGHPUT_ROW, col=col,
            seed=int(seed), value=round(float(tp[key]), 2),
            metadata={
                "commit": commit, "gpu": "cpu", "dataset_version": data_ver,
                "embedding_version": emb_ver, "metric": "throughput_rps",
                "unit": "records_per_second", "single_run_measured": True,
                "eval_capped": args.max_records is not None,
                "n_records": tp.get("batch1_n_records"), "num_threads": num_threads,
                "split": split,
                "hardware": os.environ.get("PECTI_HARDWARE", "CPU (unspecified; set $PECTI_HARDWARE)"),
            },
        )
        n_written += 1

    ft_path = _resolve(str(cfg.get("fasttext", {}).get(
        "model_path", "models/detector/fasttext/weblog_fasttext.model")))

    def _load_artifact():
        from gensim.models import FastText as _FT
        m = _FT.load(str(ft_path))
        _ = pickle.dumps(getattr(detector, "clf", None) or getattr(detector, "anomaly", None))
        return m

    if ft_path.is_file():
        lt = measure_load_time(_load_artifact, runs=5)
        write_cell_file(
            table_dir, table=_TABLE, row=_LOADTIME_ROW, col="mean",
            seed=int(seed), value=round(float(lt["mean"]), 3),
            metadata={
                "commit": commit, "gpu": "cpu", "dataset_version": data_ver,
                "embedding_version": emb_ver, "metric": "load_time_ms",
                "unit": "milliseconds", "single_run_measured": False,
                "ci95_halfwidth": round(float(lt["ci95"]), 3), "n_runs": int(lt["n"]),
                "eval_capped": False, "num_threads": num_threads,
                "hardware": os.environ.get("PECTI_HARDWARE", "CPU (unspecified; set $PECTI_HARDWARE)"),
            },
        )
        n_written += 1

    if psutil_available():
        burst = list(test_recs[: min(len(test_recs), max(1, tp_n))])

        def _run_burst():
            for r in burst:
                _featscore_one(r)

        _load_snippet = (
            "from gensim.models import FastText\n"
            f"m=FastText.load(r'{ft_path}')\n"
        )
        mem = measure_memory(_run_burst, burst=len(burst), model_load_snippet=_load_snippet)

        _mem_cols = {
            "baseline": "rss_subprocess_baseline_mb",
            "loaded": "rss_subprocess_loaded_mb",
            "model delta": "rss_model_delta_mb",
        }
        for col, key in _mem_cols.items():
            if mem.get(key) is None:
                continue
            write_cell_file(
                table_dir, table=_TABLE, row=_MEMORY_ROW, col=col,
                seed=int(seed), value=round(float(mem[key]), 3),
                metadata={
                    "commit": commit, "gpu": "cpu", "dataset_version": data_ver,
                    "embedding_version": emb_ver, "metric": "rss_mb",
                    "unit": "megabytes", "single_run_measured": True, "eval_capped": False,
                    "measurement": "clean subprocess (fresh interpreter + shipped FastText "
                                   "artifact only); excludes the corpus held in memory for timing",
                    "inproc_steadystate_rss_mb": mem.get("rss_loaded_mb"),
                    "inproc_peak_rss_mb": mem.get("rss_peak_mb"),
                    "tool": mem["tool"],
                    "num_threads": num_threads,
                    "hardware": os.environ.get("PECTI_HARDWARE", "CPU (unspecified; set $PECTI_HARDWARE)"),
                },
            )
            n_written += 1
        print(f"[40_measure_latency] RSS (clean) baseline={mem.get('rss_subprocess_baseline_mb')}MB "
              f"loaded={mem.get('rss_subprocess_loaded_mb')}MB "
              f"model-delta={mem.get('rss_model_delta_mb','n/a')}MB peak={mem.get('rss_peak_mb')}MB ({mem['tool']})")
    else:
        print("[40_measure_latency] WARN: psutil unavailable -> RSS/peak memory "
              "recorded as UNMEASURED (install the 'efficiency' extra: pip install psutil)")

    print(f"[40_measure_latency] throughput batch1={tp.get('batch1_records_per_second',0):.1f} "
          f"req/s/core, batched={tp.get('batched_records_per_second',float('nan')):.1f} req/s")

    print(f"[40_measure_latency] end_to_end mean={stages['end_to_end'].get('mean'):.4f}ms "
          f"p95={stages['end_to_end'].get('p95'):.4f}ms "
          f"p99={stages['end_to_end'].get('p99'):.4f}ms")
    print(f"[40_measure_latency] footprint: FastText={fp['FastText model']}MB "
          f"head={fp['Classifier head']}MB total={fp['Total']}MB")
    print(f"[40_measure_latency] wrote {n_written} cell file(s) under {table_dir / 'cells'}")

    try:
        append_run(
            stage="latency_table8", config_path=args.config, seed=int(seed),
            artifact=str(table_dir / "results.json"),
            notes=f"end_to_end_mean_ms={stages['end_to_end'].get('mean'):.4f}|"
                  f"footprint_mb={fp['Total']}|single_run_measured",
            extra={"eval_capped": args.max_records is not None},
        )
    except Exception as exc:
        print(f"[40_measure_latency] WARN: ledger append failed: {exc}")

    if not args.no_aggregate:

        agg = aggregate_table(table_dir, commit=commit, gpu="cpu", round_to=None)
        print(f"[40_measure_latency] aggregated -> {agg}")
        print("[40_measure_latency] render with: "
              "python scripts/50_make_paper_tables.py --table 8 --allow-exploratory")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
