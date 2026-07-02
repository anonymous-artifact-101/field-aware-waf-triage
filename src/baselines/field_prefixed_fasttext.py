
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.svm import LinearSVC

from src.baselines._common import (
    FIELD_ORDER,
    SUBTYPES,
    _field_text,
    load_split_records,
    macro_f1,
    per_class_f1,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.detector.fasttext_embed import default_fasttext_params, tokenize_field
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = [
    "FieldPrefixedFastTextEncoder",
    "field_prefixed_tokens",
    "fit_field_prefixed_fasttext",
    "run",
]

_LOGGER = get_logger("baseline.field_prefixed_fasttext")

def field_prefixed_tokens(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str] = FIELD_ORDER,
) -> List[str]:
    out: List[str] = []
    for field in fields:
        for tok in tokenize_field(_field_text(record, field)):
            out.append(f"{field}:{tok}")
    return out

def _resolve_fasttext_params(features: Mapping[str, Any]) -> Dict[str, Any]:
    params = default_fasttext_params()
    for key in (
        "vector_size",
        "window",
        "min_count",
        "min_n",
        "max_n",
        "epochs",
        "sg",
        "negative",
        "bucket",
    ):
        if key in features and features[key] is not None:
            params[key] = features[key]
    params["workers"] = 1
    return params

class FieldPrefixedFastTextEncoder:

    def __init__(self, features: Mapping[str, Any], seed: int) -> None:
        self.fields = list(features.get("fields", FIELD_ORDER))
        self.params = _resolve_fasttext_params(features)
        self.seed = int(seed)
        self.model = None

    @property
    def vector_size(self) -> int:
        return int(self.params["vector_size"])

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "FieldPrefixedFastTextEncoder":
        from gensim.models import FastText

        corpus = [
            toks for toks in (
                field_prefixed_tokens(rec, fields=self.fields) for rec in records
            )
            if toks
        ]
        if not corpus:
            raise ValueError("field-prefixed FastText received an empty corpus")

        self.model = FastText(
            vector_size=int(self.params["vector_size"]),
            window=int(self.params["window"]),
            min_count=int(self.params["min_count"]),
            min_n=int(self.params["min_n"]),
            max_n=int(self.params["max_n"]),
            sg=int(self.params["sg"]),
            negative=int(self.params["negative"]),
            bucket=int(self.params["bucket"]),
            workers=1,
            seed=self.seed,
        )
        self.model.build_vocab(corpus_iterable=corpus)
        self.model.train(
            corpus_iterable=corpus,
            total_examples=len(corpus),
            epochs=int(self.params["epochs"]),
        )
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FieldPrefixedFastTextEncoder.transform called before fit")
        x = np.zeros((len(records), self.vector_size), dtype=np.float32)
        for i, rec in enumerate(records):
            toks = field_prefixed_tokens(rec, fields=self.fields)
            if not toks:
                continue
            vecs = [self.model.wv[tok] for tok in toks]
            if vecs:
                x[i] = np.mean(vecs, axis=0, dtype=np.float32)
        return x

    @property
    def feature_dim(self) -> int:
        return self.vector_size

def fit_field_prefixed_fasttext(
    train_records: Sequence[Mapping[str, Any]],
    train_labels: Sequence[int],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    seed: int,
):
    enc = FieldPrefixedFastTextEncoder(features, seed).fit(train_records)
    x_train = enc.transform(train_records)
    clf = LinearSVC(
        C=float(model_cfg.get("C", 1.0)),
        class_weight=model_cfg.get("class_weight", None),
        max_iter=int(model_cfg.get("max_iter", 2000)),
        random_state=int(seed),
    )
    clf.fit(x_train, list(train_labels))
    return enc, clf

def _predict(enc: FieldPrefixedFastTextEncoder, clf, records: Sequence[Mapping[str, Any]]) -> List[int]:
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

    enc, clf = fit_field_prefixed_fasttext(
        labeled, subtype_labels(labeled), model_cfg, features, seed
    )
    labels = subtype_labels(test_recs)
    preds = _predict(enc, clf, test_recs)
    value = round(100.0 * macro_f1(preds, labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, labels, len(SUBTYPES))))

    _LOGGER.info(
        "field_prefixed_fasttext: budget=%.3f (%d/%d labeled), dim=%d, test n=%d, macro_f1=%.4f",
        budget, len(labeled), len(train_recs), enc.feature_dim, len(test_recs), value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "feature_dim": int(enc.feature_dim),
        "head": "LinearSVC(class_weight=None)",
        "fasttext_params": {k: int(v) for k, v in enc.params.items() if isinstance(v, int)},
        "control_role": (
            "Flat FastText control with field prefixes inside tokens; tests field-token "
            "markers without the proposed six-block concatenation."
        ),
    }
    row_label = str(cfg.get("row", "Field-prefixed FastText + LinearSVC"))
    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row=row_label, col=col, value=value, seed=seed,
            extra_metadata=extra, per_class=pcf1,
            preds=preds, labels=labels,
            split=str(data.get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)
    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records

    recs = synthetic_records(96, seed=11)
    enc, clf = fit_field_prefixed_fasttext(
        recs,
        subtype_labels(recs),
        {"C": 1.0, "class_weight": None, "max_iter": 500},
        {"vector_size": 16, "epochs": 2, "min_count": 1, "min_n": 2, "max_n": 4, "bucket": 5000},
        42,
    )
    preds = _predict(enc, clf, recs[:24])
    print(
        "[field_prefixed_fasttext] smoke: "
        f"dim={enc.feature_dim}, macro_f1={round(macro_f1(preds, subtype_labels(recs[:24])), 3)}"
    )

if __name__ == "__main__":
    _smoke()
