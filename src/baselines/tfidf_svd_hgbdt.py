
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, Mapping, Optional, Sequence

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.baselines._common import (
    SUBTYPES,
    load_split_records,
    macro_f1,
    per_class_f1,
    render_records,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "fit_tfidf_svd_hgbdt"]

_LOGGER = get_logger("baseline.tfidf_svd_hgbdt")

def _build_vectorizer(features: Mapping[str, Any]) -> TfidfVectorizer:
    ngram = features.get("ngram_range", [1, 2])
    return TfidfVectorizer(
        analyzer=str(features.get("analyzer", "char_wb")),
        ngram_range=(int(ngram[0]), int(ngram[1])),
        max_features=int(features.get("max_features", 50000)),
        lowercase=False,
    )

def fit_tfidf_svd_hgbdt(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    n_components = int(features.get("svd_components", 256))
    clf = HistGradientBoostingClassifier(
        learning_rate=float(model_cfg.get("learning_rate", 0.08)),
        max_iter=int(model_cfg.get("max_iter", 300)),
        max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 20)),
        l2_regularization=float(model_cfg.get("l2_regularization", 0.0)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        early_stopping=bool(model_cfg.get("early_stopping", True)),
        validation_fraction=float(model_cfg.get("validation_fraction", 0.1)),
        n_iter_no_change=int(model_cfg.get("n_iter_no_change", 20)),
        random_state=int(seed),
    )
    pipe = Pipeline(
        steps=[
            ("tfidf", _build_vectorizer(features)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=int(seed))),
            ("hgbdt", clf),
        ]
    )
    pipe.fit(list(train_texts), list(train_labels))
    return pipe

def _predict(model: Pipeline, texts: Sequence[str]) -> list[int]:
    return [int(p) for p in model.predict(list(texts))]

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
    budget = float(data.get("label_budget", 0.05))

    train_recs = load_split_records(
        str(data.get("train_split", "owasp_train")), max_records=max_records
    )
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    idx = select_label_budget(len(train_recs), budget)
    if not idx:
        raise ValueError(f"label_budget={budget} selected 0 of {len(train_recs)} train records")
    labeled = [train_recs[i] for i in idx]

    model = fit_tfidf_svd_hgbdt(
        render_records(labeled), subtype_labels(labeled), model_cfg, features, seed
    )

    test_labels = subtype_labels(test_recs)
    preds = _predict(model, render_records(test_recs))
    value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    _LOGGER.info(
        "tfidf_svd_hgbdt: budget=%.3f (%d/%d labeled), test n=%d, macro_f1=%.4f",
        budget, len(labeled), len(train_recs), len(test_recs), value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "tree_baseline": True,
        "svd_components": int(features.get("svd_components", 256)),
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg,
            row="TF-IDF + SVD + HGBDT",
            col=col,
            value=value,
            seed=seed,
            extra_metadata=extra,
            per_class=pcf1,
            preds=preds,
            labels=test_labels,
            split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import render_records as _rr, subtype_labels as _sl, synthetic_records

    recs = synthetic_records(96, seed=5)
    model = fit_tfidf_svd_hgbdt(
        _rr(recs),
        _sl(recs),
        {"max_iter": 10, "early_stopping": False, "class_weight": "balanced"},
        {"ngram_range": [1, 2], "max_features": 2000, "svd_components": 16},
        42,
    )
    preds = _predict(model, _rr(recs[:16]))
    print(
        "[tfidf_svd_hgbdt] smoke: fit ok, 16 preds, train macro_f1="
        f"{round(macro_f1(preds, _sl(recs[:16])), 3)}"
    )

if __name__ == "__main__":
    _smoke()
