
from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit, load_split_records
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version, embedding_version
from src.utils.seeds import set_seed

def _pin_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(int(n)))

def _score_feature(detector, feat):
    x = feat.reshape(1, -1)
    if detector.mode == "supervised":
        return detector._proba_from_features(x)
    return detector._anomaly_from_features(x)

def _field_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field, "")
    return "" if value is None else str(value)

def _uri_query_len(record: Mapping[str, Any]) -> int:
    path = _field_text(record, "path")
    query = _field_text(record, "query")
    return len(path) + len(query)

def _query_len(record: Mapping[str, Any]) -> int:
    return len(_field_text(record, "query"))

def _bin_label(value: int, bins: Sequence[int]) -> str:
    prev = 0
    for upper in bins:
        if value < upper:
            return f"{prev}-{upper - 1}"
        prev = upper
    return f">={prev}"

def _deterministic_spread(records: Sequence[Mapping[str, Any]], n: int) -> List[Mapping[str, Any]]:
    if n >= len(records):
        return list(records)

    if n <= 1:
        return [records[0]]
    last = len(records) - 1
    return [records[round(i * last / (n - 1))] for i in range(n)]

def _summarize(samples: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in samples)
    if not vals:
        return {"n": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}

    def pct(p: float) -> float:
        rank = max(1, min(len(vals), int(math.ceil(p * len(vals)))))
        return vals[rank - 1]

    stats = {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "max": vals[-1],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "p999": pct(0.999),
    }
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()}

def _stratify(records: Sequence[Mapping[str, Any]], samples: Sequence[float]) -> Dict[str, Any]:
    uri_bins = (64, 128, 256, 512, 1024)
    query_bins = (1, 64, 128, 256, 512)
    out: Dict[str, Dict[str, List[float]]] = {"uri_query_length": {}, "query_length": {}}
    for rec, sample in zip(records, samples):
        uri_key = _bin_label(_uri_query_len(rec), uri_bins)
        query_key = _bin_label(_query_len(rec), query_bins)
        out["uri_query_length"].setdefault(uri_key, []).append(float(sample))
        out["query_length"].setdefault(query_key, []).append(float(sample))
    return {
        group: {
            key: _summarize(vals)
            for key, vals in sorted(buckets.items(), key=lambda item: item[0])
        }
        for group, buckets in out.items()
    }

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="50k length-stratified latency stress artifact.")
    parser.add_argument("--config", default="configs/finetune/label_10pct.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--n-records", type=int, default=50000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["owasp_train", "owasp_val", "owasp_test"])
    parser.add_argument("--out", default="results/table_08_latency/latency_length_stress_50k.json")
    args = parser.parse_args(argv)

    _pin_threads(args.threads)
    set_seed(args.seed)

    cfg = dict(load_config(args.config))
    cfg.setdefault("data", {})
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["test_split"] = "owasp_test"

    ft_model, train_recs, _ = load_detector_inputs(cfg)
    detector = fit_detector(cfg, ft_model, train_recs, args.seed)

    all_records: List[Mapping[str, Any]] = []
    split_counts: Dict[str, int] = {}
    for split in args.splits:
        recs = load_split_records(split)
        split_counts[split] = len(recs)
        all_records.extend(recs)

    need = int(args.warmup) + int(args.n_records)
    selected = _deterministic_spread(all_records, need)
    if len(selected) <= args.warmup:
        raise ValueError(f"not enough records for warmup={args.warmup}: {len(selected)}")
    warmup_records = selected[: args.warmup]
    timed_records = selected[args.warmup : args.warmup + args.n_records]

    for rec in warmup_records:
        feat = detector.encoder.encode_record(rec)
        _score_feature(detector, feat)

    samples: List[float] = []
    t_start = time.perf_counter()
    for rec in timed_records:
        t0 = time.perf_counter()
        feat = detector.encoder.encode_record(rec)
        _score_feature(detector, feat)
        samples.append((time.perf_counter() - t0) * 1000.0)
    elapsed_s = time.perf_counter() - t_start

    artifact = {
        "artifact": "latency_length_stress_50k",
        "created_at_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "note": (
            "Large pinned-core microbenchmark over completed access-log records; "
            "tail percentiles are microbenchmark indicators, not production "
            "reverse-proxy tail-latency guarantees."
        ),
        "config": args.config,
        "seed": args.seed,
        "commit": git_commit(),
        "dataset_version": dataset_version(),
        "embedding_version": embedding_version(),
        "splits": split_counts,
        "selection": "deterministic_even_spread_over_concatenated_splits",
        "warmup_records": int(args.warmup),
        "timed_records": len(timed_records),
        "threads": int(args.threads),
        "elapsed_seconds": round(elapsed_s, 6),
        "throughput_records_per_second": round(len(timed_records) / elapsed_s, 3),
        "latency_ms": _summarize(samples),
        "length_strata_latency_ms": _stratify(timed_records, samples),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    dump_json(out_path, artifact)
    e2e = artifact["latency_ms"]
    print(
        "[56_latency_length_stress] "
        f"n={artifact['timed_records']} mean={e2e['mean']:.6f}ms "
        f"p95={e2e['p95']:.6f}ms p99={e2e['p99']:.6f}ms "
        f"p999={e2e['p999']:.6f}ms "
        f"out={out_path}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
