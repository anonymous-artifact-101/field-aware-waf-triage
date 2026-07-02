
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from src.data.tokenizers.special_tokens import SPECIAL_TOKENS, UNK

__all__ = ["train_bpe", "load_bpe", "DEFAULT_VOCAB_SIZE", "DEFAULT_MIN_FREQUENCY"]

DEFAULT_VOCAB_SIZE = 32000
DEFAULT_MIN_FREQUENCY = 2

def _set_single_thread() -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def train_bpe(
    corpus_iter: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    out_path: "str | Path" = "configs/tokenizer/bpe_32k.json",
    *,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    add_prefix_space: bool = False,
    special_tokens: "list[str] | None" = None,
    single_thread: bool = True,
) -> "Tokenizer":

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    if single_thread:
        _set_single_thread()

    if special_tokens is None:
        special_tokens = list(SPECIAL_TOKENS)

    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=add_prefix_space)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=special_tokens,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(out_path), pretty=False)
    return tokenizer

def load_bpe(path: "str | Path") -> "Tokenizer":
    from tokenizers import Tokenizer

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(str(path))

def _stream_lines(paths: Iterable["str | Path"]) -> Iterator[str]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line
