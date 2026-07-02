
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional

from src.baselines._common import (
    SUBTYPES,
    class_support,
    detection_summary,
    load_split_records,
    macro_f1,
    per_class_f1,
    select_label_budget,
    subtype_labels,
    weighted_f1,
)
from src.detector.classifier import build_detector
from src.detector.fasttext_embed import load_fasttext
from src.eval.metrics import accuracy as _accuracy
from src.eval.metrics import confusion_counts as _confusion_counts
from src.eval.metrics import per_class_f1 as _per_class_prf1
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = [
    "load_detector_inputs",
    "fit_detector",
    "evaluate_detector",
    "budget_to_col",
]

_LOGGER = get_logger("detector.evaluate")

def budget_to_col(budget: float) -> str:
    return f"{int(round(float(budget) * 100))}%"

def _fasttext_path(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("fasttext", {}).get(
        "model_path", "models/detector/fasttext/weblog_fasttext.model"
    ))

def _train_split_for_mode(cfg: Mapping[str, Any]) -> str:
    data = cfg.get("data", {})
    mode = str(cfg.get("detector", {}).get("mode", "supervised")).lower()
    if mode == "unsupervised":
        return str(data.get("normal_train_split", "weblog_pretrain"))
    return str(data.get("train_split", "owasp_train"))

def load_detector_inputs(
    cfg: Mapping[str, Any],
    *,
    max_records: Optional[int] = None,
    fasttext_model=None,
):
    data = cfg.get("data", {})
    if fasttext_model is None:
        fasttext_model = load_fasttext(_fasttext_path(cfg))
    train_recs = load_split_records(
        _train_split_for_mode(cfg), max_records=max_records
    )
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )
    return fasttext_model, train_recs, test_recs

def fit_detector(
    cfg: Mapping[str, Any],
    fasttext_model,
    train_records,
    seed: int,
):
    set_seed(int(seed))
    cfg = dict(cfg)
    cfg["seed"] = int(seed)
    detector = build_detector(cfg, fasttext_model)
    budget = float(cfg.get("data", {}).get("label_budget", 0.05))

    if detector.mode == "supervised":
        idx = select_label_budget(len(train_records), budget)
        if not idx:
            raise ValueError(
                f"label_budget={budget} selected 0 of {len(train_records)} train "
                f"records; supervised fit needs a positive budget."
            )
        labeled = [train_records[i] for i in idx]
        labeled_labels = subtype_labels(labeled)
        detector.fit(labeled, labeled_labels)
        detector._n_labeled = len(labeled)

        detector._fit_recs = labeled
        detector._fit_labels = labeled_labels
    else:
        detector.fit(train_records)
        detector._n_labeled = 0
    return detector

def evaluate_detector(
    cfg: Mapping[str, Any],
    seed: int = 42,
    *,
    max_records: Optional[int] = None,
    fasttext_model=None,
) -> Dict[str, Any]:
    fasttext_model, train_recs, test_recs = load_detector_inputs(
        cfg, max_records=max_records, fasttext_model=fasttext_model
    )
    detector = fit_detector(cfg, fasttext_model, train_recs, seed)
    budget = float(cfg.get("data", {}).get("label_budget", 0.05))
    col = budget_to_col(budget)
    row = str(cfg.get("row", "Proposed (FastText field-aware)"))

    if detector.mode == "supervised":
        test_labels = subtype_labels(test_recs)
        preds = detector.predict(test_recs)
        value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)

        train_preds = detector.predict(detector._fit_recs)
        train_macro_f1 = round(
            100.0 * macro_f1(train_preds, detector._fit_labels, len(SUBTYPES)), 4
        )
        wf1 = round(100.0 * weighted_f1(preds, test_labels, len(SUBTYPES)), 4)
        pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))
        support = dict(zip(SUBTYPES, class_support(test_labels, len(SUBTYPES))))

        prf1 = _per_class_prf1(preds, test_labels, len(SUBTYPES),
                               class_names=list(SUBTYPES), as_percent=True)
        per_class_recall = {c: round(d["recall"], 4) for c, d in prf1.items()}
        per_class_precision = {c: round(d["precision"], 4) for c, d in prf1.items()}
        macro_precision = round(sum(d["precision"] for d in prf1.values()) / len(prf1), 4)
        macro_recall = round(sum(d["recall"] for d in prf1.values()) / len(prf1), 4)
        acc = round(_accuracy(preds, test_labels, as_percent=True), 4)
        confusion = _confusion_counts(preds, test_labels, len(SUBTYPES))
        _LOGGER.info(
            "detector(supervised %s): budget=%.3f (%d labeled), test n=%d, "
            "macro_f1=%.4f weighted_f1=%.4f acc=%.2f macro_p=%.2f macro_r=%.2f",
            detector.estimator_name, budget, detector._n_labeled, len(test_recs),
            value, wf1, acc, macro_precision, macro_recall,
        )
        return {
            "value": value,
            "row": row,
            "col": col,
            "metric": "macro_f1",
            "mode": "supervised",

            "train_macro_f1": train_macro_f1,
            "per_class_f1": pcf1,

            "weighted_f1": wf1,
            "class_support": support,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "accuracy": acc,
            "per_class_recall": per_class_recall,
            "per_class_precision": per_class_precision,
            "confusion_matrix": confusion,
            "preds": preds,

            "test_labels": test_labels,
            "n_train_labeled": detector._n_labeled,
            "n_test": len(test_recs),
            "detector": detector,
        }

    scores = detector.anomaly_score(test_recs)
    summary = detection_summary([float(s) for s in scores], subtype_labels(test_recs))
    value = round(float(summary["mean_score"]), 4)
    _LOGGER.info(
        "detector(unsupervised %s): test n=%d, mean_anomaly_score=%.4f",
        detector.scorer_name, len(test_recs), value,
    )
    return {
        "value": value,
        "row": row,
        "col": col,
        "metric": "mean_anomaly_score",
        "mode": "unsupervised",
        "score_summary": summary,
        "scores": [float(s) for s in scores],
        "n_test": len(test_recs),
        "detector": detector,
    }

def _smoke() -> None:
    from src.baselines._common import synthetic_records
    from src.detector.fasttext_embed import train_fasttext

    recs = synthetic_records(120, seed=5)
    model = train_fasttext(
        recs, {"vector_size": 16, "epochs": 2, "min_count": 1, "min_n": 2, "max_n": 4}, seed=42
    )
    cfg = {
        "granularity": 6,
        "encoder": {"pooling": "mean"},
        "detector": {"mode": "supervised", "estimator": "logreg"},
        "data": {"label_budget": 0.5},
        "row": "Proposed (FastText field-aware)",
    }

    det = fit_detector(cfg, model, recs, seed=42)
    preds = det.predict(recs[:24])
    mf1 = macro_f1(preds, subtype_labels(recs[:24]), len(SUBTYPES))
    print(f"[evaluate] smoke(supervised): fit+predict ok, train macro_f1={mf1:.3f}")

    cfg_uns = dict(cfg)
    cfg_uns["detector"] = {"mode": "unsupervised", "scorer": "centroid"}
    cfg_uns["data"] = {"label_budget": 0.0}
    det2 = fit_detector(cfg_uns, model, recs, seed=42)
    s = det2.anomaly_score(recs[:24])
    print(f"[evaluate] smoke(unsupervised): scores n={len(s)} mean={float(s.mean()):.4f}")

if __name__ == "__main__":
    _smoke()
