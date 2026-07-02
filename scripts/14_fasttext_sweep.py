
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import (
    SUBTYPES, git_commit, load_split_records, macro_f1, select_label_budget,
    subtype_labels, weighted_f1,
)
from src.detector.classifier import build_detector
from src.detector.fasttext_embed import save_fasttext, train_fasttext
from src.eval.aggregate import write_cell_file
from src.eval.latency import measure_latency
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import MODELS_DIR, RESULTS_DIR
from src.utils.runlog import append_run
from src.utils.seeds import set_seed

_TABLE = "table_10_fasttext_sweep"
_MB = 1024.0 * 1024.0
_SWEEP_DIR = MODELS_DIR / "detector" / "fasttext_sweep"

def _parse_ngram(s: str) -> Tuple[int, int]:
    lo, hi = s.split("-")
    return int(lo), int(hi)

def main(argv: "Optional[List[str]]" = None) -> int:
    p = argparse.ArgumentParser(description="FastText vector_size x n-gram efficiency sweep.")
    p.add_argument("--config", default="configs/finetune/label_5pct.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vector-sizes", type=int, nargs="*", default=[64, 128, 256])
    p.add_argument("--ngrams", nargs="*", default=["3-6", "2-8"])
    p.add_argument("--max-records", type=int, default=None, help="Cap (smoke only).")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--n-trials", type=int, default=300)
    args = p.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    set_seed(args.seed)

    cfg = dict(load_config(args.config))
    base_ft = dict(cfg.get("fasttext", {}))
    budget = float(cfg.get("data", {}).get("label_budget", 0.05))

    pre = load_split_records(str(base_ft.get("train_split", "weblog_pretrain")),
                             max_records=args.max_records)
    train = load_split_records("owasp_train", max_records=args.max_records)
    test = load_split_records("owasp_test", max_records=args.max_records)
    idx = select_label_budget(len(train), budget)
    labeled = [train[i] for i in idx]
    labels = subtype_labels(labeled)
    test_labels = subtype_labels(test)
    _SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    table_dir = RESULTS_DIR / _TABLE
    commit = git_commit()
    hardware = os.environ.get("PECTI_HARDWARE", "CPU (set $PECTI_HARDWARE)")
    grid = [(v, _parse_ngram(g)) for v in args.vector_sizes for g in args.ngrams]
    print(f"[14_sweep] {len(grid)} configs; pre={len(pre)} labeled={len(labeled)} test={len(test)}")

    rows: List[Dict[str, Any]] = []
    for vsize, (min_n, max_n) in grid:
        ft_cfg = dict(base_ft)
        ft_cfg.update({"vector_size": int(vsize), "min_n": int(min_n), "max_n": int(max_n)})
        tag = f"v{vsize}_n{min_n}{max_n}"

        model = train_fasttext(pre, ft_cfg, seed=args.seed)
        mpath = _SWEEP_DIR / f"weblog_{tag}.model"
        save_fasttext(model, mpath)

        dcfg = dict(cfg)
        dcfg["fasttext"] = ft_cfg
        det = build_detector(dcfg, model)
        det.fit(labeled, labels)
        preds = det.predict(test)
        mf = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
        wf = round(100.0 * weighted_f1(preds, test_labels, len(SUBTYPES)), 4)

        timing = measure_latency(det, test, warmup=args.warmup, n_trials=args.n_trials)
        e2e = timing["stages"]["end_to_end"]
        ft_bytes = sum(s.stat().st_size for s in mpath.parent.glob(mpath.name + "*") if s.is_file())
        fp_mb = round(ft_bytes / _MB, 4)

        row = {"config": tag, "vector_size": vsize, "min_n": min_n, "max_n": max_n,
               "macro_f1": mf, "weighted_f1": wf,
               "latency_ms": round(float(e2e["mean"]), 6),
               "p95_ms": round(float(e2e["p95"]), 6), "footprint_mb": fp_mb}
        rows.append(row)
        print(f"[14_sweep] {tag:10s} macro_f1={mf:5.1f} wtd={wf:5.1f} "
              f"lat={row['latency_ms']:.3f}ms fp={fp_mb:.1f}MB")

        meta = {"commit": commit, "gpu": "cpu", "metric": "macro_f1",
                "single_run_measured": True, "eval_capped": args.max_records is not None,
                "vector_size": vsize, "min_n": min_n, "max_n": max_n,
                "latency_ms": row["latency_ms"], "footprint_mb": fp_mb,
                "weighted_f1": wf, "hardware": hardware,
                "note": "FastText efficiency sweep; base model (v64_n36) is the deployed config."}
        write_cell_file(table_dir, table=_TABLE, row=tag, col="macro_f1",
                        seed=int(args.seed), value=mf, metadata=meta)
        try:
            append_run(stage="fasttext_sweep", config_path=args.config, seed=int(args.seed),
                       artifact=str(mpath), notes=f"{tag}|f1={mf}|lat={row['latency_ms']}|fp={fp_mb}",
                       extra={"eval_capped": args.max_records is not None})
        except Exception as exc:
            print(f"[14_sweep] WARN ledger: {exc}")

    dump_json(table_dir / "sweep_summary.json", {
        "table": _TABLE, "seed": args.seed, "commit": commit, "rows": rows,
        "note": "vector_size x n-gram grid; macro_f1 vs latency_ms vs footprint_mb. "
                "Base deployed config is v64_n36.",
    })
    print(f"[14_sweep] wrote {len(rows)} configs -> {table_dir/'sweep_summary.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
