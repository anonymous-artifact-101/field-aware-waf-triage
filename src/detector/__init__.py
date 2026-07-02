
from __future__ import annotations

__all__ = [
    "train_fasttext",
    "load_fasttext",
    "save_fasttext",
    "FieldEncoder",
    "FieldAwareDetector",
    "build_detector",
]

def __getattr__(name: str):

    if name in ("train_fasttext", "load_fasttext", "save_fasttext"):
        from src.detector import fasttext_embed as _m

        return getattr(_m, name)
    if name == "FieldEncoder":
        from src.detector.field_encoder import FieldEncoder as _c

        return _c
    if name in ("FieldAwareDetector", "build_detector"):
        from src.detector import classifier as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
