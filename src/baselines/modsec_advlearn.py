
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
    SUBTYPES,
    load_split_records,
    macro_f1,
    per_class_f1,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.baselines.modsec_learn import CrsFiringVectorizer
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "AdversarialLinearClassifier", "fit_modsec_advlearn"]

_LOGGER = get_logger("baseline.modsec_advlearn")

class AdversarialLinearClassifier:

    def __init__(
        self,
        num_classes: int,
        *,
        C: float = 1.0,
        epsilon: float = 0.1,
        n_steps: int = 7,
        train_ratio: float = 0.5,
        max_iter: int = 2000,
        seed: int = 42,
    ) -> None:
        self.num_classes = int(num_classes)
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.n_steps = int(n_steps)
        self.train_ratio = float(train_ratio)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.clf: Optional[LinearSVC] = None

    def _new_svc(self) -> LinearSVC:
        return LinearSVC(
            C=self.C, class_weight="balanced", max_iter=self.max_iter,
            random_state=self.seed,
        )

    def _coef_for_class(self, cls: int) -> np.ndarray:
        assert self.clf is not None
        coef = self.clf.coef_
        classes = list(self.clf.classes_)
        if coef.shape[0] == 1:

            return coef[0] if cls == classes[1] else -coef[0]
        try:
            return coef[classes.index(cls)]
        except ValueError:
            return np.zeros(coef.shape[1], dtype=coef.dtype)

    def _adversarial_flip(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_adv = x.copy()

        for _ in range(self.n_steps):
            for i in range(x_adv.shape[0]):
                w_true = self._coef_for_class(int(y[i]))
                active = int(x_adv[i].sum())
                budget = max(1, int(np.ceil(self.epsilon * max(active, 1))))

                delta = (1.0 - 2.0 * x_adv[i]) * (-w_true)
                top = np.argsort(delta)[::-1][:budget]
                x_adv[i, top] = 1.0 - x_adv[i, top]
        return x_adv

    def fit(self, x: np.ndarray, y: Sequence[int]) -> "AdversarialLinearClassifier":
        y = np.asarray(list(y), dtype=np.int64)

        self.clf = self._new_svc()
        self.clf.fit(x, y)

        k = int(round(self.train_ratio * x.shape[0]))
        if k > 0:
            adv = self._adversarial_flip(x[:k], y[:k])
            x_aug = np.vstack([x, adv])
            y_aug = np.concatenate([y, y[:k]])
            self.clf = self._new_svc()
            self.clf.fit(x_aug, y_aug)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.clf is not None
        return np.asarray(self.clf.predict(x), dtype=np.int64)

def fit_modsec_advlearn(
    train_recs: Sequence[Mapping[str, Any]],
    train_labels: Sequence[int],
    model_cfg: Mapping[str, Any],
    features: Mapping[str, Any],
    adversarial: Mapping[str, Any],
    seed: int,
):
    vec = CrsFiringVectorizer(
        source_field=str(features.get("rule_firing_source", "crs_rule_ids")),
        binary=bool(features.get("binary_firings", True)),
    )
    x_train = vec.fit_transform(train_recs)
    clf = AdversarialLinearClassifier(
        num_classes=int(model_cfg.get("num_classes", len(SUBTYPES))),
        C=float(model_cfg.get("C", 1.0)),
        epsilon=float(adversarial.get("epsilon", 0.1)),
        n_steps=int(adversarial.get("n_steps", 7)),
        train_ratio=float(adversarial.get("train_ratio", 0.5)) if adversarial.get("enabled", True) else 0.0,
        max_iter=int(model_cfg.get("max_iter", 2000)),
        seed=seed,
    )
    clf.fit(x_train, train_labels)
    return vec, clf

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
    adversarial = cfg.get("adversarial", {})
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

    vec, clf = fit_modsec_advlearn(
        labeled, subtype_labels(labeled), model_cfg, features, adversarial, seed
    )
    test_labels = subtype_labels(test_recs)
    preds = [int(p) for p in clf.predict(vec.transform(test_recs))]
    value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    _LOGGER.info(
        "modsec_advlearn: budget=%.3f (%d/%d labeled), |crs_vocab|=%d, eps=%.2f, test n=%d, macro_f1=%.4f",
        budget, len(labeled), len(train_recs), len(vec.vocab_),
        float(adversarial.get("epsilon", 0.1)), len(test_recs), value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "crs_vocab_size": len(vec.vocab_),
        "adversarial": {
            "enabled": bool(adversarial.get("enabled", True)),
            "epsilon": float(adversarial.get("epsilon", 0.1)),
            "n_steps": int(adversarial.get("n_steps", 7)),
            "train_ratio": float(adversarial.get("train_ratio", 0.5)),
        },
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="ModSec-AdvLearn", col=col, value=value, seed=seed,
            extra_metadata=extra, per_class=pcf1,
            preds=preds, labels=test_labels,
            split=str(cfg.get("data", {}).get("test_split", "owasp_test")),
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records, subtype_labels as _sl

    recs = synthetic_records(80, seed=6)
    vec, clf = fit_modsec_advlearn(
        recs, _sl(recs), {"num_classes": len(SUBTYPES), "C": 1.0},
        {"rule_firing_source": "crs_rule_ids"},
        {"enabled": True, "epsilon": 0.1, "n_steps": 3, "train_ratio": 0.5}, 42,
    )
    preds = [int(p) for p in clf.predict(vec.transform(recs))]
    print(f"[modsec_advlearn] smoke: fit ok, |crs_vocab|={len(vec.vocab_)}, "
          f"train macro_f1={round(macro_f1(preds, _sl(recs)), 3)}")

if __name__ == "__main__":
    _smoke()
