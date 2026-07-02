
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit, load_split_records
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.eval.aggregate import load_cell_files, write_cell_file
from src.eval.metrics import mean_ci95
from src.utils.config import load_config
from src.utils.io import dump_json, load_json
from src.utils.paths import RESULTS_DIR, ROOT
from src.utils.runlog import append_run, dataset_version, embedding_version
from src.utils.seeds import set_seed

_TABLE = "table_09_cross_dataset_csic"
_CSIC = ROOT / "data" / "processed" / "csic_parsed" / "csic.jsonl"

def _load_csic(max_records: Optional[int] = None) -> "tuple[List[Dict[str, Any]], List[int]]":
    if not _CSIC.is_file():
        raise SystemExit(f"[13_cross_dataset_csic] CSIC not found at {_CSIC}; "
                         f"run the CSIC parser first.")
    recs: List[Dict[str, Any]] = []
    labels: List[int] = []
    with _CSIC.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_records is not None and i >= max_records:
                break
            d = load_json_line(line)
            recs.append(d)
            labels.append(1 if str(d.get("label", "")).lower() == "attack" else 0)
    return recs, labels

def load_json_line(line: str) -> Dict[str, Any]:
    import json
    return json.loads(line)

def _roc_auc(scores: List[float], labels: List[int]) -> float:
    pairs = sorted(zip(scores, range(len(scores))), key=lambda t: t[0])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def _pr_auc(scores: List[float], labels: List[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap

def _threshold_metrics(scores: List[float], labels: List[int]) -> Dict[str, float]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    best = {"j": -1.0}
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 0)
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        j = tpr - fpr
        if j > best["j"]:
            best = {"j": j, "threshold": float(t), "tpr": tpr, "fpr": fpr,
                    "precision": prec, "tp": tp, "fp": fp}
    return best

def _aggregate(commit: str) -> Path:
    table_dir = RESULTS_DIR / _TABLE
    cells = load_cell_files(table_dir)
    if not cells:
        raise SystemExit(f"[13_cross_dataset_csic] no cells under {table_dir/'cells'}; "
                         f"run seeds 42..46 first.")

    series: Dict[str, List[float]] = {k: [] for k in
                                      ("roc_auc", "pr_auc", "tpr", "fpr", "precision")}
    embs, seeds = set(), []
    for c in cells:
        md = c.get("metadata", {})
        tm = md.get("threshold_metrics", {})
        series["roc_auc"].append(float(c["value"]))
        series["pr_auc"].append(float(md["pr_auc"]))
        series["tpr"].append(float(tm["tpr"]))
        series["fpr"].append(float(tm["fpr"]))
        series["precision"].append(float(tm["precision"]))
        embs.add(md.get("embedding_version"))
        seeds.append(int(c["seed"]))
    if len(embs) > 1:
        raise SystemExit(f"[13_cross_dataset_csic] cells span multiple embeddings "
                         f"{embs}; re-run all seeds on the current embedding before "
                         f"aggregating (stale-artifact guard).")
    seeds = sorted(set(seeds))
    out_cells = []
    for col, vals in series.items():
        m, (lo, hi) = mean_ci95(vals)
        out_cells.append({"row": "Proposed (zero-shot transfer)", "col": col,
                          "value": round(m, 4), "ci_95": [round(lo, 4), round(hi, 4)],
                          "seeds": seeds})
    last = cells[-1].get("metadata", {})
    doc = {
        "table": _TABLE,
        "cells": out_cells,
        "metadata": {
            "commit": commit, "date": last.get("date") or "", "gpu": "cpu",
            "ci_method": "student_t", "n_cell_files": len(cells),
            "dataset": "csic2010", "dataset_version": last.get("dataset_version"),
            "embedding_version": embs.pop() if embs else None,
            "n_total": last.get("n_csic"), "n_attack": last.get("n_attack"),
            "n_benign": last.get("n_benign"),
            "base_rate_pos": round(float(last.get("n_attack", 0))
                                   / max(1, int(last.get("n_csic", 1))), 4),
            "youden_j_threshold": last.get("threshold_metrics", {}).get("threshold"),
            "note": "Zero-shot OWASP->CSIC transfer, 5 seeds {42..46}; deterministic "
                    "in seed (centroid fit on benign weblog_pretrain, no CSIC fit), "
                    "hence zero-width CI. Aggregated by scripts/13 --aggregate.",
        },
    }
    out = table_dir / "results.json"
    dump_json(out, doc)
    return out

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="CSIC 2010 cross-dataset transfer (zero-shot).")
    parser.add_argument("--config", default="configs/finetune/label_0pct.yaml",
                        help="Unsupervised detector config (0% label, centroid scorer).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None, help="Cap (smoke only).")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--aggregate", action="store_true",
                        help="Collect per-seed cells -> 5-col results.json and exit.")
    args = parser.parse_args(argv)

    if args.aggregate:
        out = _aggregate(git_commit())
        print(f"[13_cross_dataset_csic] aggregated 5-col results -> {out}")
        return 0

    cfg = dict(load_config(args.config))
    seed = int(args.seed)
    set_seed(seed)

    ft_model, train_recs, _ = load_detector_inputs(cfg, max_records=args.max_records)
    detector = fit_detector(cfg, ft_model, train_recs, seed)
    if detector.mode != "unsupervised":
        raise SystemExit("[13_cross_dataset_csic] use the 0%-label (unsupervised) "
                         "config; subtype labels do not transfer to CSIC's binary task.")

    csic_recs, csic_labels = _load_csic(max_records=args.max_records)
    scores = [float(s) for s in detector.anomaly_score(csic_recs)]
    auc = _roc_auc(scores, csic_labels)
    pr_auc = _pr_auc(scores, csic_labels)
    thr = _threshold_metrics(scores, csic_labels)

    n_pos = sum(csic_labels)
    n_neg = len(csic_labels) - n_pos
    print(f"[13_cross_dataset_csic] zero-shot CSIC: n={len(csic_recs)} "
          f"(attack={n_pos}, benign={n_neg}) ROC-AUC={auc:.4f} PR-AUC={pr_auc:.4f}")
    print(f"[13_cross_dataset_csic] @best-J threshold: TPR={thr['tpr']:.3f} "
          f"FPR={thr['fpr']:.3f} precision={thr['precision']:.3f}")

    if args.no_write:
        return 0

    table_dir = RESULTS_DIR / _TABLE
    metadata = {
        "commit": git_commit(),
        "gpu": "cpu",
        "dataset_version": dataset_version(),
        "embedding_version": embedding_version(),
        "metric": "roc_auc",
        "mode": "unsupervised_zero_shot_transfer",
        "checkpoint_loaded": True,
        "exploratory_single_seed": True,
        "eval_capped": args.max_records is not None,
        "n_csic": len(csic_recs),
        "n_attack": n_pos,
        "n_benign": n_neg,
        "pr_auc": round(float(pr_auc), 4),
        "threshold_metrics": {k: (round(float(v), 4) if isinstance(v, float) else v)
                              for k, v in thr.items() if k != "j"},
        "note": "Zero-shot transfer: detector trained on OWASP/Kaggle, NO CSIC fit. "
                "CSIC is saturated, so ROC-AUC of the transferred anomaly score "
                "(not an F1 horse-race) is the external-validity claim.",
    }
    write_cell_file(table_dir, table=_TABLE, row="Proposed (zero-shot transfer)",
                    col="ROC-AUC", seed=seed, value=round(auc, 4), metadata=metadata)
    print(f"[13_cross_dataset_csic] wrote cell under {table_dir / 'cells'}")

    try:
        append_run(stage="cross_dataset_csic", config_path=args.config, seed=seed,
                   artifact=str(table_dir), notes=f"roc_auc={round(auc,4)}|n={len(csic_recs)}",
                   extra={"eval_capped": args.max_records is not None})
    except Exception as exc:
        print(f"[13_cross_dataset_csic] WARN ledger: {exc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
