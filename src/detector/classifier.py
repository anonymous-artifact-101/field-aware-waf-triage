
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.data.labels import SUBTYPES
from src.detector.field_encoder import FieldEncoder
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = [
    "SUPERVISED_ESTIMATORS",
    "ANOMALY_SCORERS",
    "FieldAwareDetector",
    "build_detector",
]

_LOGGER = get_logger("detector.classifier")

SUPERVISED_ESTIMATORS = ("hist_gbdt", "logreg", "linear_svc")
ANOMALY_SCORERS = ("centroid", "isolation_forest")

def _build_estimator(name: str, params: Mapping[str, Any], seed: int):
    name = str(name).lower()
    if name in ("hist_gbdt", "hist_gradient_boosting", "hgb"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_iter=int(params.get("max_iter", 200)),
            max_depth=params.get("max_depth", None),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=int(seed),
        )
    if name in ("logreg", "logistic_regression", "logistic"):
        from sklearn.linear_model import LogisticRegression

        kw: Dict[str, Any] = {
            "C": float(params.get("C", 1.0)),
            "solver": str(params.get("solver", "lbfgs")),
            "max_iter": int(params.get("max_iter", 1000)),
            "class_weight": params.get("class_weight", "balanced"),
            "random_state": int(seed),
        }
        penalty = params.get("penalty")
        if penalty is not None and str(penalty) != "l2":
            kw["penalty"] = str(penalty)
        return LogisticRegression(**kw)
    if name in ("linear_svc", "linearsvc", "svc"):
        from sklearn.svm import LinearSVC

        return LinearSVC(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=int(seed),
        )
    raise ValueError(
        f"unknown supervised estimator {name!r}; options are {SUPERVISED_ESTIMATORS}"
    )

class _CentroidAnomaly:

    def __init__(self) -> None:
        self.centroid_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_CentroidAnomaly":
        self.centroid_ = np.asarray(X, dtype=np.float64).mean(axis=0)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.centroid_ is None:
            raise RuntimeError("centroid anomaly scorer is not fitted")
        diff = np.asarray(X, dtype=np.float64) - self.centroid_
        return np.sqrt((diff * diff).sum(axis=1))

class FieldAwareDetector:

    def __init__(
        self,
        encoder: FieldEncoder,
        *,
        mode: str = "supervised",
        estimator: str = "hist_gbdt",
        scorer: str = "centroid",
        estimator_params: Optional[Mapping[str, Any]] = None,
        scorer_params: Optional[Mapping[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        if mode not in ("supervised", "unsupervised"):
            raise ValueError(f"mode must be 'supervised' or 'unsupervised', got {mode!r}")
        self.encoder = encoder
        self.mode = mode
        self.estimator_name = estimator
        self.scorer_name = scorer
        self.estimator_params = dict(estimator_params or {})
        self.scorer_params = dict(scorer_params or {})
        self.seed = int(seed)
        self.clf = None
        self.anomaly = None
        self._fitted = False

    def fit(
        self,
        train_records: Sequence[Mapping[str, Any]],
        labels: Optional[Sequence[int]] = None,
    ) -> "FieldAwareDetector":
        set_seed(self.seed)
        X = self.encoder.encode_records(list(train_records))

        if self.mode == "supervised":
            if labels is None:
                raise ValueError("supervised fit requires labels")
            self.clf = _build_estimator(self.estimator_name, self.estimator_params, self.seed)
            self.clf.fit(X, list(labels))
        else:
            if str(self.scorer_name).lower() in ("isolation_forest", "iforest", "if"):
                from sklearn.ensemble import IsolationForest

                self.anomaly = IsolationForest(
                    n_estimators=int(self.scorer_params.get("n_estimators", 200)),
                    contamination=self.scorer_params.get("contamination", "auto"),
                    random_state=self.seed,
                )
                self.anomaly.fit(X)
            elif str(self.scorer_name).lower() in ("centroid", "distance"):
                self.anomaly = _CentroidAnomaly().fit(X)
            else:
                raise ValueError(
                    f"unknown anomaly scorer {self.scorer_name!r}; options {ANOMALY_SCORERS}"
                )
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("detector is not fitted; call fit() first")

    def predict(self, records: Sequence[Mapping[str, Any]]) -> List[int]:
        self._require_fitted()
        if self.mode != "supervised":
            raise RuntimeError("predict() is only valid in supervised mode")
        X = self.encoder.encode_records(list(records))
        return [int(p) for p in self.clf.predict(X)]

    def predict_proba(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        self._require_fitted()
        if self.mode != "supervised":
            raise RuntimeError("predict_proba() is only valid in supervised mode")
        X = self.encoder.encode_records(list(records))
        return self._proba_from_features(X)

    def _proba_from_features(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return np.asarray(self.clf.predict_proba(X), dtype=np.float64)

        margins = np.asarray(self.clf.decision_function(X), dtype=np.float64)
        if margins.ndim == 1:
            margins = np.column_stack([-margins, margins])
        m = margins - margins.max(axis=1, keepdims=True)
        e = np.exp(m)
        return e / e.sum(axis=1, keepdims=True)

    def anomaly_score(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        self._require_fitted()
        if self.mode != "unsupervised":
            raise RuntimeError("anomaly_score() is only valid in unsupervised mode")
        X = self.encoder.encode_records(list(records))
        return self._anomaly_from_features(X)

    def _anomaly_from_features(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.anomaly, _CentroidAnomaly):
            return self.anomaly.score_samples(X)

        return -np.asarray(self.anomaly.score_samples(X), dtype=np.float64)

    def _record_score_vector(self, blocks: List[np.ndarray]) -> float:
        feat = np.concatenate(blocks).reshape(1, -1)
        if self.mode == "supervised":
            return float(self._proba_from_features(feat)[0].max())
        return float(self._anomaly_from_features(feat)[0])

    def field_attribution(
        self, records: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        self._require_fitted()
        labels = self.encoder.block_labels
        out: List[Dict[str, Any]] = []
        for rec in records:
            blocks = self.encoder.encode_record_blocks(rec)
            ref = self._record_score_vector(blocks)
            deltas: Dict[str, float] = {}
            for i, _ in enumerate(blocks):
                perturbed = [b if j != i else np.zeros_like(b) for j, b in enumerate(blocks)]
                deltas[labels[i]] = ref - self._record_score_vector(perturbed)

            best_i = max(range(len(blocks)), key=lambda i: (deltas[labels[i]], -i))
            out.append(
                {
                    "field": labels[best_i],
                    "block_index": best_i,
                    "deltas": {k: float(v) for k, v in deltas.items()},
                    "reference_score": float(ref),
                }
            )
        return out

def build_detector(cfg: Mapping[str, Any], fasttext_model) -> FieldAwareDetector:
    enc_cfg = dict(cfg.get("encoder", {}))
    det_cfg = dict(cfg.get("detector", {}))

    granularity = cfg.get("granularity")
    if granularity is None:
        granularity = cfg.get("model", {}).get("num_fields", 6)
    groups = enc_cfg.get("groups")

    encoder = FieldEncoder(
        fasttext_model,
        granularity=granularity,
        groups=groups,
        pooling=str(enc_cfg.get("pooling", "mean")),
    )
    seed = int(cfg.get("seed", 42))
    return FieldAwareDetector(
        encoder,
        mode=str(det_cfg.get("mode", "supervised")),
        estimator=str(det_cfg.get("estimator", "hist_gbdt")),
        scorer=str(det_cfg.get("scorer", "centroid")),
        estimator_params=det_cfg.get("estimator_params", {}),
        scorer_params=det_cfg.get("scorer_params", {}),
        seed=seed,
    )

def _smoke() -> None:
    from src.baselines._common import (
        macro_f1,
        subtype_labels,
        synthetic_records,
    )
    from src.detector.fasttext_embed import train_fasttext

    recs = synthetic_records(96, seed=4)
    model = train_fasttext(
        recs,
        {"vector_size": 16, "epochs": 3, "min_count": 1, "min_n": 2, "max_n": 4},
        seed=42,
    )
    enc = FieldEncoder(model, granularity=6)

    sup = FieldAwareDetector(enc, mode="supervised", estimator="logreg", seed=42)
    sup.fit(recs, subtype_labels(recs))
    preds = sup.predict(recs[:32])
    mf1 = macro_f1(preds, subtype_labels(recs[:32]), len(SUBTYPES))
    attr = sup.field_attribution(recs[:4])
    print(
        f"[classifier] smoke(supervised/logreg): train macro_f1={mf1:.3f} "
        f"proba.shape={sup.predict_proba(recs[:3]).shape} "
        f"attr_fields={[a['field'] for a in attr]}"
    )

    sup2 = FieldAwareDetector(enc, mode="supervised", estimator="hist_gbdt", seed=42)
    sup2.fit(recs, subtype_labels(recs))
    print(f"[classifier] smoke(supervised/hist_gbdt): n_preds={len(sup2.predict(recs[:8]))}")

    for scorer in ("centroid", "isolation_forest"):
        uns = FieldAwareDetector(enc, mode="unsupervised", scorer=scorer, seed=42)
        uns.fit(recs)
        s = uns.anomaly_score(recs[:16])
        print(
            f"[classifier] smoke(unsupervised/{scorer}): scores n={len(s)} "
            f"mean={float(s.mean()):.4f} finite={bool(np.isfinite(s).all())}"
        )

if __name__ == "__main__":
    _smoke()
