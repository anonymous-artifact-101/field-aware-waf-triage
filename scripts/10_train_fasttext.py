
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import load_split_records
from src.detector.fasttext_embed import save_fasttext, train_fasttext
from src.utils.config import load_config
from src.utils.paths import ROOT
from src.utils.runlog import append_run
from src.utils.seeds import set_seed

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Train the field-aware FastText embedding.")
    parser.add_argument(
        "--config",
        default="configs/detector/fasttext_base.yaml",
        help="Detector YAML config (default: configs/detector/fasttext_base.yaml).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--out", default=None, help="Override the output model path.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap corpus records (earliest contiguous; for fast smoke runs only).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    ft_cfg = dict(cfg.get("fasttext", {}))
    train_split = str(ft_cfg.get("train_split", "weblog_pretrain"))
    out_path = _resolve(
        args.out or ft_cfg.get("model_path", "models/detector/fasttext/weblog_fasttext.model")
    )

    print(f"[10_train_fasttext] loading corpus split={train_split} (max_records={args.max_records})")
    records = load_split_records(train_split, max_records=args.max_records)
    print(f"[10_train_fasttext] {len(records)} records; fitting FastText (seed={seed}, workers=1)")

    model = train_fasttext(records, ft_cfg, seed=seed)
    saved = save_fasttext(model, out_path)
    print(
        f"[10_train_fasttext] wrote {saved} "
        f"(vocab={len(model.wv)}, vector_size={model.wv.vector_size})"
    )

    try:
        append_run(
            stage="train_fasttext", config_path=args.config, seed=int(seed),
            artifact=str(saved),
            notes=f"vocab={len(model.wv)}|vector_size={model.wv.vector_size}|"
                  f"split={train_split}",
            extra={"eval_capped": args.max_records is not None},
        )
    except Exception as exc:
        print(f"[10_train_fasttext] WARN: ledger append failed: {exc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
