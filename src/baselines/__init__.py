
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

__all__ = ["BASELINE_RUNNERS", "get_runner"]

def _lazy(module_name: str) -> Callable[..., Dict[str, Any]]:

    def _runner(cfg: Mapping[str, Any], seed: int, **kwargs: Any) -> Dict[str, Any]:
        import importlib

        mod = importlib.import_module(f"src.baselines.{module_name}")
        return mod.run(cfg, seed, **kwargs)

    _runner.__name__ = f"run_{module_name}"
    return _runner

BASELINE_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "isolation_forest": _lazy("isolation_forest"),
    "ocsvm": _lazy("ocsvm"),
    "tfidf_logreg": _lazy("tfidf_logreg"),
    "tfidf_svd_hgbdt": _lazy("tfidf_svd_hgbdt"),

    "tfidf_typed_linearsvc": _lazy("tfidf_typed_linearsvc"),
    "field_prefixed_fasttext": _lazy("field_prefixed_fasttext"),
    "hashing_char_sgd": _lazy("hashing_char_sgd"),
    "char_cnn": _lazy("char_cnn"),
    "deeplog": _lazy("deeplog"),
    "logbert": _lazy("logbert"),
    "modsec_learn": _lazy("modsec_learn"),
    "modsec_advlearn": _lazy("modsec_advlearn"),

    "status_only": _lazy("status_only"),
}

def get_runner(baseline: str) -> Callable[..., Dict[str, Any]]:
    try:
        return BASELINE_RUNNERS[baseline]
    except KeyError as exc:
        known = ", ".join(sorted(BASELINE_RUNNERS))
        raise KeyError(f"unknown baseline {baseline!r}; known baselines: {known}") from exc
