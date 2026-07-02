
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.baselines._common import (
    detection_summary,
    load_split_records,
    render_records,
    write_results_json,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "fit_unsupervised_text_detector"]

_LOGGER = get_logger("baseline.isolation_forest")

def _build_vectorizer(features: Mapping[str, Any]) -> TfidfVectorizer:
    ngram = features.get("ngram_range", [3, 5])
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(int(ngram[0]), int(ngram[1])),
        max_features=int(features.get("max_features", 20000)),
        lowercase=False,
    )

def _resolve_normal_records(
    cfg_data: Mapping[str, Any], max_records: Optional[int]
) -> "tuple[List[dict], str, Optional[str]]":
    normal_split = cfg_data.get("normal_train_split")
    normal_jsonl = cfg_data.get("normal_jsonl")
    if normal_split or normal_jsonl:
        try:
            if normal_jsonl:
                from src.utils.io import read_jsonl

                recs = list(read_jsonl(normal_jsonl))
                if max_records:
                    recs = recs[: int(max_records)]
            else:
                recs = load_split_records(str(normal_split), max_records=max_records)
            return recs, str(normal_split or normal_jsonl), None
        except FileNotFoundError:
            _LOGGER.warning("configured benign reference missing; falling back to OWASP proxy")
    train_split = str(cfg_data.get("train_split", "owasp_train"))
    recs = load_split_records(train_split, max_records=max_records)
    return (
        recs,
        f"{train_split}_proxy",
        "No parsed weblog benign corpus available; fit on the OWASP train split as a "
        "proxy normal reference. OWASP is a honeypot (all-attack), so the absolute "
        "anomaly level is not a clean novelty signal -- documented limitation.",
    )

def fit_unsupervised_text_detector(
    train_texts: Sequence[str],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    seed: int,
):
    vec = _build_vectorizer(features)
    x_train = vec.fit_transform(train_texts)
    contamination = model_cfg.get("contamination", "auto")
    forest = IsolationForest(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        max_samples=model_cfg.get("max_samples", "auto"),
        contamination=contamination,
        max_features=float(model_cfg.get("max_features", 1.0)),
        bootstrap=bool(model_cfg.get("bootstrap", False)),
        random_state=int(seed),
        n_jobs=1,
    )
    forest.fit(x_train)
    return vec, forest

def _anomaly_scores(vec, forest, texts: Sequence[str]) -> np.ndarray:
    x = vec.transform(texts)

    return -forest.score_samples(x)

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
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    vec, forest = fit_unsupervised_text_detector(
        render_records(normal_recs), model_cfg, features, seed
    )
    scores = _anomaly_scores(vec, forest, render_records(test_recs))
    summary = detection_summary(scores.tolist(), [1] * len(scores))
    value = round(float(summary["mean_score"]), 6)

    _LOGGER.info(
        "isolation_forest: fit on %d (%s), scored %d test; mean_anomaly=%.6f",
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
            cfg, row="Isolation Forest", col="0%", value=value, seed=seed, extra_metadata=extra
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "summary": summary, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records, render_records as _rr

    recs = synthetic_records(64, seed=1)
    vec, forest = fit_unsupervised_text_detector(
        _rr(recs), {"n_estimators": 50, "contamination": 0.1}, {"ngram_range": [3, 4], "max_features": 2000}, 42
    )
    scores = _anomaly_scores(vec, forest, _rr(recs[:8]))
    print(f"[isolation_forest] smoke: fit ok, 8 scores e.g. {[round(s, 3) for s in scores.tolist()[:4]]}")

if __name__ == "__main__":
    _smoke()
