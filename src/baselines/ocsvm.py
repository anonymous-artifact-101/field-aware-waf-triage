
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM

from src.baselines._common import (
    detection_summary,
    load_split_records,
    render_records,
    write_results_json,
)
from src.baselines.isolation_forest import _resolve_normal_records
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "fit_ocsvm"]

_LOGGER = get_logger("baseline.ocsvm")

def _build_vectorizer(features: Mapping[str, Any]) -> TfidfVectorizer:
    ngram = features.get("ngram_range", [3, 5])
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(int(ngram[0]), int(ngram[1])),
        max_features=int(features.get("max_features", 20000)),
        lowercase=False,
    )

def fit_ocsvm(
    train_texts: Sequence[str],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
):
    vec = _build_vectorizer(features)
    x_train = vec.fit_transform(train_texts)
    svm = OneClassSVM(
        kernel=str(model_cfg.get("kernel", "rbf")),
        nu=float(model_cfg.get("nu", 0.1)),
        gamma=model_cfg.get("gamma", "scale"),
    )
    svm.fit(x_train)
    return vec, svm

def _anomaly_scores(vec, svm, texts: Sequence[str]) -> np.ndarray:
    x = vec.transform(texts)

    return -svm.decision_function(x)

def run(
    cfg: Mapping[str, Any],
    seed: int = 42,
    *,
    max_records: Optional[int] = None,
    write: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    data = cfg.get("data", {})
    features = cfg.get("features", {})
    model_cfg = cfg.get("model", {})

    normal_recs, ref_name, limitation = _resolve_normal_records(data, max_records)

    max_fit = model_cfg.get("max_fit_samples", 8000 if max_records is None else None)
    if max_fit is not None and len(normal_recs) > int(max_fit):
        normal_recs = normal_recs[: int(max_fit)]
        ref_name = f"{ref_name}[:{int(max_fit)}]"

    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    vec, svm = fit_ocsvm(render_records(normal_recs), model_cfg, features)
    scores = _anomaly_scores(vec, svm, render_records(test_recs))
    summary = detection_summary(scores.tolist(), [1] * len(scores))
    value = round(float(summary["mean_score"]), 6)

    _LOGGER.info(
        "ocsvm: fit on %d (%s), scored %d test; mean_anomaly=%.6f",
        len(normal_recs), ref_name, len(test_recs), value,
    )

    extra = {
        "metric": "mean_anomaly_score",
        "normal_reference": ref_name,
        "n_fit": len(normal_recs),
        "n_test": len(test_recs),
        "score_summary": {k: round(float(v), 6) for k, v in summary.items()},
    }
    if limitation:
        extra["limitation"] = limitation

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="One-Class SVM", col="0%", value=value, seed=seed, extra_metadata=extra
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "summary": summary, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records, render_records as _rr

    recs = synthetic_records(64, seed=2)
    vec, svm = fit_ocsvm(_rr(recs), {"nu": 0.1, "kernel": "rbf"}, {"ngram_range": [3, 4], "max_features": 2000})
    scores = _anomaly_scores(vec, svm, _rr(recs[:8]))
    print(f"[ocsvm] smoke: fit ok, 8 scores e.g. {[round(s, 3) for s in scores.tolist()[:4]]}")

if __name__ == "__main__":
    _smoke()
