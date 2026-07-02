
from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.eval.latency import percentiles

__all__ = [
    "mean_ci95_runs",
    "repeat_latency",
    "measure_throughput",
    "measure_load_time",
    "measure_memory",
    "psutil_available",
]

def psutil_available() -> bool:
    try:
        import psutil
        return True
    except Exception:
        return False

def mean_ci95_runs(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "ci95": 0.0, "std": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)

    try:
        from scipy.stats import t as _t
        crit = float(_t.ppf(0.975, n - 1))
    except Exception:
        _TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
                  15: 2.131, 20: 2.086, 30: 2.042}
        crit = _TABLE.get(n - 1, 1.96)
    ci = crit * std / math.sqrt(n)
    return {"mean": mean, "ci95": ci, "std": std, "n": n}

def repeat_latency(
    time_one_run: Callable[[], Dict[str, Any]],
    *,
    runs: int = 10,
) -> Dict[str, Any]:
    per_run_e2e_mean: List[float] = []
    per_run_e2e_p95: List[float] = []
    per_run_e2e_p99: List[float] = []
    last_stages: Dict[str, Any] = {}
    for _ in range(max(1, int(runs))):
        gc.collect()
        stages = time_one_run()
        last_stages = stages
        e2e = stages.get("end_to_end", {})
        if e2e.get("mean") is not None:
            per_run_e2e_mean.append(float(e2e["mean"]))
        if e2e.get("p95") is not None:
            per_run_e2e_p95.append(float(e2e["p95"]))
        if e2e.get("p99") is not None:
            per_run_e2e_p99.append(float(e2e["p99"]))
    return {
        "runs": int(runs),
        "end_to_end_mean": mean_ci95_runs(per_run_e2e_mean),
        "end_to_end_p95": mean_ci95_runs(per_run_e2e_p95),
        "end_to_end_p99": mean_ci95_runs(per_run_e2e_p99),

        "last_run_stages": last_stages,
    }

def measure_throughput(
    featurize_score_one: Callable[[Mapping[str, Any]], Any],
    records: Sequence[Mapping[str, Any]],
    *,
    warmup: int = 20,
    n_records: int = 2000,
    batch_score: Optional[Callable[[Sequence[Mapping[str, Any]]], Any]] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    recs = list(records)
    timed = recs[warmup : warmup + max(1, int(n_records))]
    if not timed:
        timed = recs[: max(1, int(n_records))]

    for r in recs[:warmup]:
        featurize_score_one(r)
    t0 = time.perf_counter()
    for r in timed:
        featurize_score_one(r)
    elapsed = time.perf_counter() - t0
    out: Dict[str, Any] = {
        "batch1_records_per_second": (len(timed) / elapsed) if elapsed > 0 else float("nan"),
        "batch1_n_records": len(timed),
        "batch1_elapsed_s": elapsed,
    }
    if batch_score is not None:
        bs = max(1, int(batch_size))
        batches = [timed[i : i + bs] for i in range(0, len(timed), bs)]
        for b in batches[: max(1, warmup // bs or 1)]:
            batch_score(b)
        t0 = time.perf_counter()
        n_done = 0
        for b in batches:
            batch_score(b)
            n_done += len(b)
        el = time.perf_counter() - t0
        out.update({
            "batched_records_per_second": (n_done / el) if el > 0 else float("nan"),
            "batch_size": bs,
            "batched_n_records": n_done,
        })
    return out

def measure_load_time(load_fn: Callable[[], Any], *, runs: int = 5) -> Dict[str, Any]:
    samples: List[float] = []
    for _ in range(max(1, int(runs))):
        gc.collect()
        t0 = time.perf_counter()
        obj = load_fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
        del obj
    agg = mean_ci95_runs(samples)
    agg["unit"] = "milliseconds"
    agg["samples_ms"] = [round(s, 3) for s in samples]
    return agg

def measure_memory(
    run_burst: Callable[[], None],
    *,
    burst: int = 2000,
    model_load_snippet: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        import psutil
    except Exception as exc:
        raise RuntimeError(
            "measure_memory needs psutil (optional 'efficiency' extra: "
            "pip install psutil); install it or record memory as unmeasured."
        ) from exc

    proc = psutil.Process()
    mb = 1024.0 * 1024.0

    gc.collect()
    rss_loaded = proc.memory_info().rss / mb
    peak = rss_loaded
    for _ in range(8):
        run_burst()
        cur = proc.memory_info().rss / mb
        if cur > peak:
            peak = cur

    out: Dict[str, Any] = {
        "rss_loaded_mb": round(rss_loaded, 3),
        "rss_peak_mb": round(peak, 3),
        "burst_records": int(burst),
        "tool": f"psutil {getattr(psutil, '__version__', '?')}",
    }

    if model_load_snippet:
        import subprocess
        import sys
        driver = (
            "import psutil,gc\n"
            "p=psutil.Process();mb=1024.0*1024.0\n"
            "gc.collect();base=p.memory_info().rss/mb\n"
            f"{model_load_snippet}\n"
            "gc.collect();loaded=p.memory_info().rss/mb\n"
            "print('RSSDELTA',base,loaded,loaded-base)\n"
        )
        try:
            res = subprocess.run([sys.executable, "-c", driver],
                                 capture_output=True, text=True, timeout=300)
            line = next((l for l in res.stdout.splitlines() if l.startswith("RSSDELTA")), None)
            if line:
                _, base, loaded, delta = line.split()
                out["rss_subprocess_baseline_mb"] = round(float(base), 3)
                out["rss_subprocess_loaded_mb"] = round(float(loaded), 3)
                out["rss_model_delta_mb"] = round(float(delta), 3)
        except Exception as exc:
            out["rss_model_delta_mb_error"] = str(exc)

    return out
