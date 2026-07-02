
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.baselines._common import (
    FIELD_ORDER,
    SUBTYPES,
    _field_text,
    load_split_records,
    macro_f1,
    per_class_f1,
    render_record,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "fit_typed_tfidf_linearsvc", "TypedTfidfEncoder"]

_LOGGER = get_logger("baseline.tfidf_typed_linearsvc")

class TypedTfidfEncoder:

    def __init__(self, features: Mapping[str, Any]) -> None:
        ngram = features.get("ngram_range", [2, 5])
        self.analyzer = str(features.get("analyzer", "char_wb"))
        self.ngram_range = (int(ngram[0]), int(ngram[1]))

        self.max_features_per_field = int(features.get("max_features_per_field", 20000))
        self.fields: List[str] = list(features.get("fields", FIELD_ORDER))

        self.flat = bool(features.get("flat", False))
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.block_dims: Dict[str, int] = {}

    def _field_texts(self, records: Sequence[Mapping[str, Any]], field: str) -> List[str]:
        return [_field_text(r, field) for r in records]

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "TypedTfidfEncoder":
        recs = list(records)
        if self.flat:

            vec = TfidfVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                max_features=self.max_features_per_field * len(self.fields),
                lowercase=False,
            )
            mat = vec.fit_transform([render_record(r) for r in recs])
            self.vectorizers["__flat__"] = vec
            self.block_dims["__flat__"] = mat.shape[1]
            return self
        for field in self.fields:
            vec = TfidfVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                max_features=self.max_features_per_field,
                lowercase=False,
            )
            texts = self._field_texts(recs, field)

            try:
                mat = vec.fit_transform(texts)
                self.vectorizers[field] = vec
                self.block_dims[field] = mat.shape[1]
            except ValueError:
                self.vectorizers[field] = None
                self.block_dims[field] = 1
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> csr_matrix:
        recs = list(records)
        if self.flat:
            return self.vectorizers["__flat__"].transform(
                [render_record(r) for r in recs]
            ).tocsr()
        blocks = []
        for field in self.fields:
            vec = self.vectorizers.get(field)
            if vec is None:
                blocks.append(csr_matrix((len(recs), 1), dtype=np.float64))
            else:
                blocks.append(vec.transform(self._field_texts(recs, field)))
        return hstack(blocks).tocsr()

    @property
    def feature_dim(self) -> int:
        return int(sum(self.block_dims.values()))

def fit_typed_tfidf_linearsvc(
    train_records: Sequence[Mapping[str, Any]],
    train_labels: Sequence[int],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    seed: int,
):
    enc = TypedTfidfEncoder(features).fit(train_records)
    x_train = enc.transform(train_records)
    clf = LinearSVC(
        C=float(model_cfg.get("C", 1.0)),
        class_weight=model_cfg.get("class_weight", None),
        max_iter=int(model_cfg.get("max_iter", 2000)),
        random_state=int(seed),
    )
    clf.fit(x_train, list(train_labels))
    return enc, clf

def _predict(enc: TypedTfidfEncoder, clf, records: Sequence[Mapping[str, Any]]) -> List[int]:
    return [int(p) for p in clf.predict(enc.transform(records))]

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

    enc, clf = fit_typed_tfidf_linearsvc(
        labeled, subtype_labels(labeled), model_cfg, features, seed
    )

    test_labels = subtype_labels(test_recs)
    preds = _predict(enc, clf, test_recs)
    value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    _LOGGER.info(
        "tfidf_typed_linearsvc: budget=%.3f (%d/%d labeled), feat_dim=%d, test n=%d, macro_f1=%.4f",
        budget, len(labeled), len(train_recs), enc.feature_dim, len(test_recs), value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "feature_dim": int(enc.feature_dim),
        "per_field_block_dims": {k: int(v) for k, v in enc.block_dims.items()},
        "head": "LinearSVC(class_weight=None)",
        "ablation_role": "typed-block char-TF-IDF replaces per-field FastText embed "
                         "(isolates FastText's contribution; same skeleton + head as proposed).",
    }

    default_row = "TF-IDF flat + LinearSVC" if features.get("flat") else "TF-IDF typed + LinearSVC"
    row_label = str(cfg.get("row", default_row))
    extra["ablation_role"] = (
        "FLAT char-TF-IDF (one global vectorizer over the rendered record) + un-weighted "
        "LinearSVC; field-awareness control vs the typed row (same features + head)."
        if features.get("flat") else extra["ablation_role"]
    )

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row=row_label, col=col, value=value, seed=seed,
            extra_metadata=extra, per_class=pcf1,
            preds=preds, labels=test_labels,
            split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records, subtype_labels as _sl

    recs = synthetic_records(96, seed=7)
    enc, clf = fit_typed_tfidf_linearsvc(
        recs, _sl(recs),
        {"C": 1.0, "class_weight": None, "max_iter": 500},
        {"ngram_range": [2, 4], "analyzer": "char_wb", "max_features_per_field": 2000},
        42,
    )
    preds = _predict(enc, clf, recs[:24])
    print(f"[tfidf_typed_linearsvc] smoke: fit ok, feat_dim={enc.feature_dim}, "
          f"block_dims={enc.block_dims}, 24 preds, train macro_f1="
          f"{round(macro_f1(preds, _sl(recs[:24])), 3)}")

if __name__ == "__main__":
    _smoke()
