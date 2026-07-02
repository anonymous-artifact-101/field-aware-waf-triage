
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.detector.evaluate import fit_detector, load_detector_inputs
from src.utils.config import load_config
from src.utils.paths import ROOT, ensure_dir
from src.utils.seeds import set_seed

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the FastText field-aware detector head.")
    parser.add_argument("--config", required=True, help="configs/finetune/<x>.yaml or detector config.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed (paper set: 42..46).")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap records per split (earliest contiguous; for fast smoke runs only).",
    )
    parser.add_argument("--out", default=None, help="Override the fitted-detector output path.")
    args = parser.parse_args(argv)

    cfg = dict(load_config(args.config))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    fasttext_model, train_recs, _ = load_detector_inputs(cfg, max_records=args.max_records)
    detector = fit_detector(cfg, fasttext_model, train_recs, seed)

    out_dir = _resolve(args.out or cfg.get("output_dir", "models/detector/finetuned/run/"))
    ensure_dir(out_dir)
    out_path = out_dir / "detector.pkl"

    payload = {
        "mode": detector.mode,
        "estimator_name": detector.estimator_name,
        "scorer_name": detector.scorer_name,
        "clf": detector.clf,
        "anomaly": detector.anomaly,
        "block_labels": detector.encoder.block_labels,
        "granularity": cfg.get("granularity"),
        "groups": cfg.get("encoder", {}).get("groups"),
        "pooling": cfg.get("encoder", {}).get("pooling", "mean"),
        "fasttext_path": cfg.get("fasttext", {}).get("model_path"),
        "seed": seed,
        "n_train_labeled": getattr(detector, "_n_labeled", None),
    }
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[11_finetune] fit detector mode={detector.mode} "
        f"({'estimator=' + detector.estimator_name if detector.mode == 'supervised' else 'scorer=' + detector.scorer_name}) "
        f"n_labeled={getattr(detector, '_n_labeled', 0)} -> {out_path}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
