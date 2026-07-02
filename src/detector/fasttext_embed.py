
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from src.baselines._common import FIELD_ORDER, _field_text
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir
from src.utils.seeds import set_seed

__all__ = [
    "FIELD_ORDER",
    "tokenize_field",
    "record_tokens",
    "iter_record_tokens",
    "train_fasttext",
    "save_fasttext",
    "load_fasttext",
    "default_fasttext_params",
]

_LOGGER = get_logger("detector.fasttext_embed")

_DELIMS = re.compile(r"[\s/?&=;:,.+%#@!()\[\]{}<>\"'\\|^~`*-]+")

def tokenize_field(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    return [tok for tok in _DELIMS.split(text) if tok]

def record_tokens(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str] = FIELD_ORDER,
) -> List[str]:
    tokens: List[str] = []
    for field in fields:
        tokens.extend(tokenize_field(_field_text(record, field)))
    return tokens

def iter_record_tokens(
    records: Iterable[Mapping[str, Any]],
    *,
    fields: Sequence[str] = FIELD_ORDER,
):
    for rec in records:
        toks = record_tokens(rec, fields=fields)
        if toks:
            yield toks

def default_fasttext_params() -> dict:
    return {
        "vector_size": 64,
        "window": 5,
        "min_count": 1,
        "min_n": 3,
        "max_n": 6,
        "epochs": 5,
        "sg": 1,
        "negative": 5,
        "bucket": 200000,
        "workers": 1,
    }

def _resolve_params(cfg: Optional[Mapping[str, Any]]) -> dict:
    params = default_fasttext_params()
    if cfg:

        src = cfg.get("fasttext", cfg) if isinstance(cfg, Mapping) else {}
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
            if key in src and src[key] is not None:
                params[key] = src[key]

    params["workers"] = 1
    return params

def train_fasttext(
    corpus_iter: Iterable[Mapping[str, Any]],
    cfg: Optional[Mapping[str, Any]] = None,
    *,
    seed: int = 42,
    fields: Sequence[str] = FIELD_ORDER,
    max_records: Optional[int] = None,
):
    from gensim.models import FastText

    set_seed(int(seed))
    params = _resolve_params(cfg)

    corpus: List[List[str]] = []
    for i, toks in enumerate(iter_record_tokens(corpus_iter, fields=fields)):
        if max_records is not None and i >= int(max_records):
            break
        corpus.append(toks)

    if not corpus:
        raise ValueError(
            "train_fasttext received an empty corpus (every record tokenized to "
            "nothing). Check the split path and the field values."
        )

    _LOGGER.info(
        "training FastText: %d sentences, vector_size=%d, n-grams %d..%d, "
        "epochs=%d, min_count=%d (workers=1, seed=%d)",
        len(corpus),
        params["vector_size"],
        params["min_n"],
        params["max_n"],
        params["epochs"],
        params["min_count"],
        seed,
    )

    model = FastText(
        vector_size=int(params["vector_size"]),
        window=int(params["window"]),
        min_count=int(params["min_count"]),
        min_n=int(params["min_n"]),
        max_n=int(params["max_n"]),
        sg=int(params["sg"]),
        negative=int(params["negative"]),
        bucket=int(params["bucket"]),
        workers=1,
        seed=int(seed),
    )
    model.build_vocab(corpus_iterable=corpus)
    model.train(
        corpus_iterable=corpus,
        total_examples=len(corpus),
        epochs=int(params["epochs"]),
    )
    _LOGGER.info(
        "FastText trained: vocab=%d, vector_size=%d",
        len(model.wv),
        model.wv.vector_size,
    )
    return model

def save_fasttext(model, path: "str | Path") -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    model.save(str(out))
    _LOGGER.info("saved FastText model -> %s", out)
    return out

def load_fasttext(path: "str | Path"):
    from gensim.models import FastText

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"FastText model not found: {p}. Run scripts/10_train_fasttext.py first."
        )
    return FastText.load(str(p))

def _smoke() -> None:
    from src.baselines._common import synthetic_records

    recs = synthetic_records(120, seed=1)
    model = train_fasttext(
        recs,
        {"vector_size": 16, "epochs": 2, "min_count": 1, "min_n": 2, "max_n": 4},
        seed=42,
    )

    oov = "totally-unseen-payload-zzz"
    vec = model.wv[oov]
    in_vocab = "etc" in model.wv
    print(
        f"[fasttext_embed] smoke: vocab={len(model.wv)} dim={model.wv.vector_size} "
        f"OOV({oov!r}) vec_norm={float((vec ** 2).sum() ** 0.5):.4f} "
        f"in_vocab('etc')={in_vocab}"
    )

    model2 = train_fasttext(
        recs,
        {"vector_size": 16, "epochs": 2, "min_count": 1, "min_n": 2, "max_n": 4},
        seed=42,
    )
    same = bool((model2.wv[oov] == vec).all())
    print(f"[fasttext_embed] smoke: deterministic re-fit identical_OOV={same}")

if __name__ == "__main__":
    _smoke()
