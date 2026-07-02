
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, Mapping, Optional, Sequence

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.utils.murmurhash import murmurhash3_32

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

__all__ = ["HashingCharLinearPredictor", "fit_hashing_char_sgd", "run"]

_LOGGER = get_logger("baseline.hashing_char_sgd")

def _ngram_range(features: Mapping[str, Any]) -> tuple[int, int]:
    ngram = features.get("ngram_range", [2, 5])
    return (int(ngram[0]), int(ngram[1]))

def fit_hashing_char_sgd(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    seed: int,
):
    vec = HashingVectorizer(
        analyzer=str(features.get("analyzer", "char_wb")),
        ngram_range=_ngram_range(features),
        n_features=int(features.get("n_features", 2 ** 18)),
        alternate_sign=bool(features.get("alternate_sign", False)),
        norm=features.get("norm", "l2"),
        lowercase=bool(features.get("lowercase", False)),
    )
    clf = SGDClassifier(
        loss=str(model_cfg.get("loss", "hinge")),
        alpha=float(model_cfg.get("alpha", 1e-5)),
        max_iter=int(model_cfg.get("max_iter", 1000)),
        tol=float(model_cfg.get("tol", 1e-3)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=int(seed),
    )
    pipe = make_pipeline(vec, clf)
    pipe.fit(list(train_texts), list(train_labels))
    return pipe

class HashingCharLinearPredictor:

    def __init__(self, pipe) -> None:
        self.vectorizer = pipe.named_steps["hashingvectorizer"]
        self.classifier = pipe.named_steps["sgdclassifier"]
        self.analyzer = self.vectorizer.build_analyzer()
        self.n_features = int(self.vectorizer.n_features)
        self.coef = self.classifier.coef_
        self.intercept = self.classifier.intercept_
        self.classes = self.classifier.classes_
        if bool(getattr(self.vectorizer, "alternate_sign", False)):
            raise ValueError("HashingCharLinearPredictor expects alternate_sign=False")

    def featurize_text(self, text: str) -> Dict[int, float]:
        feats: Dict[int, float] = {}
        for token in self.analyzer(text):
            idx = abs(murmurhash3_32(token, 0)) % self.n_features
            feats[idx] = feats.get(idx, 0.0) + 1.0
        norm = getattr(self.vectorizer, "norm", "l2")
        if norm == "l2":
            denom = sum(v * v for v in feats.values()) ** 0.5
            if denom:
                inv = 1.0 / denom
                for idx in list(feats):
                    feats[idx] *= inv
        elif norm == "l1":
            denom = sum(abs(v) for v in feats.values())
            if denom:
                inv = 1.0 / denom
                for idx in list(feats):
                    feats[idx] *= inv
        return feats

    def predict_from_features(self, feats: Mapping[int, float]) -> int:
        import numpy as np

        scores = self.intercept.copy()
        for idx, value in feats.items():
            scores += self.coef[:, int(idx)] * float(value)
        return int(self.classes[int(np.argmax(scores))])

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

    train_texts = render_records(labeled)
    labels_train = subtype_labels(labeled)
    pipe = fit_hashing_char_sgd(train_texts, labels_train, model_cfg, features, seed)
    test_labels = subtype_labels(test_recs)
    preds = [int(p) for p in pipe.predict(render_records(test_recs))]
    value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    _LOGGER.info(
        "hashing_char_sgd: budget=%.3f (%d/%d labeled), n_features=%d, test n=%d, macro_f1=%.4f",
        budget, len(labeled), len(train_recs), int(features.get("n_features", 2 ** 18)),
        len(test_recs), value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "feature_dim": int(features.get("n_features", 2 ** 18)),
        "head": f"SGDClassifier(loss={model_cfg.get('loss', 'hinge')})",
        "control_role": (
            "Stateless char-ngram feature hashing baseline; no learned vocabulary, "
            "memory dominated by the linear head."
        ),
    }
    row_label = str(cfg.get("row", "Hashing char-ngram + SGD"))
    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row=row_label, col=col, value=value, seed=seed,
            extra_metadata=extra, per_class=pcf1,
            preds=preds, labels=test_labels,
            split=str(data.get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)
    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records

    recs = synthetic_records(96, seed=13)
    pipe = fit_hashing_char_sgd(
        render_records(recs),
        subtype_labels(recs),
        {"loss": "hinge", "alpha": 1e-5, "max_iter": 50, "tol": 1e-3, "class_weight": None},
        {"ngram_range": [2, 4], "analyzer": "char_wb", "n_features": 2 ** 12},
        42,
    )
    preds = [int(p) for p in pipe.predict(render_records(recs[:24]))]
    print(
        "[hashing_char_sgd] smoke: "
        f"macro_f1={round(macro_f1(preds, subtype_labels(recs[:24])), 3)}"
    )

if __name__ == "__main__":
    _smoke()
