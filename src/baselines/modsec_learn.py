
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.baselines._common import (
    SUBTYPES,
    load_split_records,
    macro_f1,
    per_class_f1,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "CrsFiringVectorizer", "fit_modsec_learn"]

_LOGGER = get_logger("baseline.modsec_learn")

class CrsFiringVectorizer:

    def __init__(self, source_field: str = "crs_rule_ids", binary: bool = True) -> None:
        self.source_field = str(source_field)
        self.binary = bool(binary)
        self.vocab_: Dict[str, int] = {}

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "CrsFiringVectorizer":
        seen: Dict[str, int] = {}
        for rec in records:
            for rid in rec.get(self.source_field, []) or []:
                key = str(rid)
                if key not in seen:
                    seen[key] = len(seen)
        self.vocab_ = seen
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        n = len(records)
        d = max(len(self.vocab_), 1)
        x = np.zeros((n, d), dtype=np.float32)
        for i, rec in enumerate(records):
            for rid in rec.get(self.source_field, []) or []:
                j = self.vocab_.get(str(rid))
                if j is not None:
                    x[i, j] = 1.0 if self.binary else x[i, j] + 1.0
        return x

    def fit_transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return self.fit(records).transform(records)

def fit_modsec_learn(
    train_recs: Sequence[Mapping[str, Any]],
    train_labels: Sequence[int],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    seed: int,
):
    vec = CrsFiringVectorizer(
        source_field=str(features.get("rule_firing_source", "crs_rule_ids")),
        binary=bool(features.get("binary_firings", True)),
    )
    x_train = vec.fit_transform(train_recs)
    clf = LogisticRegression(
        C=float(model_cfg.get("C", 1.0)),
        penalty=str(model_cfg.get("penalty", "l2")),
        solver="lbfgs",
        max_iter=int(model_cfg.get("max_iter", 1000)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=int(seed),
    )
    clf.fit(x_train, list(train_labels))
    return vec, clf

def _predict(vec: CrsFiringVectorizer, clf, records: Sequence[Mapping[str, Any]]) -> List[int]:
    return [int(p) for p in clf.predict(vec.transform(records))]

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

    vec, clf = fit_modsec_learn(labeled, subtype_labels(labeled), model_cfg, features, seed)
    test_labels = subtype_labels(test_recs)
    preds = _predict(vec, clf, test_recs)
    value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    _LOGGER.info(
        "modsec_learn: budget=%.3f (%d/%d labeled), |crs_vocab|=%d, test n=%d, macro_f1=%.4f",
        budget, len(labeled), len(train_recs), len(vec.vocab_), len(test_recs), value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "crs_vocab_size": len(vec.vocab_),
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="ModSec-Learn", col=col, value=value, seed=seed,
            extra_metadata=extra, per_class=pcf1,
            preds=preds, labels=test_labels,
            split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records, subtype_labels as _sl

    recs = synthetic_records(80, seed=5)
    vec, clf = fit_modsec_learn(
        recs, _sl(recs), {"C": 1.0, "max_iter": 200}, {"rule_firing_source": "crs_rule_ids"}, 42
    )
    preds = _predict(vec, clf, recs)
    print(f"[modsec_learn] smoke: fit ok, |crs_vocab|={len(vec.vocab_)}, "
          f"train macro_f1={round(macro_f1(preds, _sl(recs)), 3)}")

if __name__ == "__main__":
    _smoke()
