
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["percentiles", "measure_latency"]

def percentiles(samples: List[float], probs=(0.95, 0.99)) -> Dict[str, float]:
    if not samples:
        nan = float("nan")
        base = {"mean": nan, "min": nan, "max": nan, "p50": nan, "n": 0}
        base.update({f"p{int(p * 100)}": nan for p in probs})
        return base
    ordered = sorted(samples)
    n = len(ordered)

    def _pct(p: float) -> float:

        rank = max(1, min(n, math.ceil(p * n)))
        return ordered[rank - 1]

    out = {
        "mean": sum(ordered) / n,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": _pct(0.50),
        "n": n,
    }
    for p in probs:
        out[f"p{int(p * 100)}"] = _pct(p)
    return out

def measure_latency(
    detector,
    records: Sequence[Mapping[str, Any]],
    *,
    warmup: int = 20,
    n_trials: int = 300,
    probs=(0.95, 0.99),
) -> Dict[str, Any]:
    encoder = detector.encoder
    groups = encoder.groups

    def _score_feature(feat):
        x = feat.reshape(1, -1)
        if detector.mode == "supervised":
            return detector._proba_from_features(x)
        return detector._anomaly_from_features(x)

    recs = list(records)
    if len(recs) <= warmup:
        raise ValueError(
            f"need more than warmup={warmup} records to time; got {len(recs)}"
        )
    timed = recs[warmup : warmup + max(1, int(n_trials))]

    samples: Dict[str, List[float]] = {"tokenize": [], "encode": [], "score": [], "end_to_end": []}

    from src.detector.fasttext_embed import tokenize_field
    from src.baselines._common import _field_text

    for rec in timed:

        t0 = time.perf_counter()
        feat_e2e = encoder.encode_record(rec)
        _ = _score_feature(feat_e2e)
        samples["end_to_end"].append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        for g in groups:
            for field in g:
                tokenize_field(_field_text(rec, field))
        samples["tokenize"].append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        feat = encoder.encode_record(rec)
        samples["encode"].append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = _score_feature(feat)
        samples["score"].append((time.perf_counter() - t0) * 1000.0)

    stages = {name: percentiles(vals, probs=probs) for name, vals in samples.items()}
    return {
        "unit": "milliseconds",
        "warmup": int(warmup),
        "n_trials": len(timed),
        "batch_size": 1,
        "mode": detector.mode,
        "stages": stages,
    }

def _smoke() -> None:
    from src.baselines._common import synthetic_records, subtype_labels
    from src.detector.classifier import FieldAwareDetector
    from src.detector.field_encoder import FieldEncoder
    from src.detector.fasttext_embed import train_fasttext

    recs = synthetic_records(80, seed=6)
    model = train_fasttext(
        recs, {"vector_size": 16, "epochs": 2, "min_count": 1, "min_n": 2, "max_n": 4}, seed=42
    )
    det = FieldAwareDetector(FieldEncoder(model, granularity=6), mode="supervised",
                             estimator="logreg", seed=42)
    det.fit(recs, subtype_labels(recs))
    out = measure_latency(det, recs, warmup=5, n_trials=30)
    e2e = out["stages"]["end_to_end"]
    print(f"[latency] smoke: end_to_end mean={e2e['mean']:.4f}ms p95={e2e['p95']:.4f}ms "
          f"n={out['n_trials']} stages={list(out['stages'])}")

if __name__ == "__main__":
    _smoke()
