
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.baselines._common import FIELD_ORDER, SUBTYPES, load_split_records, render_records, subtype_labels
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR, ensure_dir
from src.utils.runlog import append_run, dataset_version, embedding_version, git_commit

_TABLE = "table_12_label_noise_cleanlab"
_DEFAULT_OUT = RESULTS_DIR / _TABLE

def _load_records(split_names: Sequence[str], max_records: int | None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    remaining = max_records
    for split in split_names:
        if remaining is not None and remaining <= 0:
            break
        part = load_split_records(split, max_records=remaining)
        for rec in part:
            rec = dict(rec)
            rec["_split"] = split
            records.append(rec)
        if remaining is not None:
            remaining -= len(part)
    return records

def _build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=str(args.analyzer),
        ngram_range=(int(args.ngram_min), int(args.ngram_max)),
        max_features=int(args.max_features),
        lowercase=False,
    )

def _fit_fold(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    test_texts: Sequence[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vec = _build_vectorizer(args)
    x_train = vec.fit_transform(train_texts)
    x_test = vec.transform(test_texts)
    clf = LogisticRegression(
        C=float(args.C),
        solver=str(args.solver),
        max_iter=int(args.max_iter),
        class_weight=("balanced" if args.class_weight_balanced else None),
        random_state=int(seed),
    )
    clf.fit(x_train, list(train_labels))

    probs_fold = np.zeros((len(test_texts), len(SUBTYPES)), dtype=np.float64)
    raw = clf.predict_proba(x_test)
    for j, cls in enumerate(clf.classes_):
        probs_fold[:, int(cls)] = raw[:, j]
    preds_fold = probs_fold.argmax(axis=1).astype(np.int64)
    return probs_fold, preds_fold

def _oof_probabilities(
    texts: Sequence[str],
    labels: Sequence[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    probs = np.zeros((len(labels_arr), len(SUBTYPES)), dtype=np.float64)
    preds = np.zeros(len(labels_arr), dtype=np.int64)
    fold_reports: List[Dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels_arr)), labels_arr), start=1):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = [int(labels_arr[i]) for i in train_idx]
        probs_fold, preds_fold = _fit_fold(train_texts, train_labels, test_texts, args, int(args.seed) + fold_idx)
        probs[test_idx, :] = probs_fold
        preds[test_idx] = preds_fold
        y_fold = labels_arr[test_idx]
        fold_reports.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "macro_f1": round(float(f1_score(y_fold, preds_fold, average="macro", labels=list(range(len(SUBTYPES))))), 6),
                "accuracy": round(float(accuracy_score(y_fold, preds_fold)), 6),
            }
        )
        print(
            f"[44] fold {fold_idx}/{args.folds}: n_train={len(train_idx)} "
            f"n_test={len(test_idx)} macro_f1={fold_reports[-1]['macro_f1']:.4f} "
            f"acc={fold_reports[-1]['accuracy']:.4f}"
        )
    return probs, preds, fold_reports

def _cleanlab_issues(labels: Sequence[int], pred_probs: np.ndarray) -> tuple[np.ndarray, str]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    try:
        from cleanlab.filter import find_label_issues

        ranked = find_label_issues(
            labels=labels_arr,
            pred_probs=pred_probs,
            filter_by="prune_by_noise_rate",
            return_indices_ranked_by="self_confidence",
        )
        return np.asarray(ranked, dtype=np.int64), "cleanlab.filter.find_label_issues"
    except Exception as exc:
        print(f"[44] cleanlab unavailable/failed ({exc}); using conservative fallback.")
        self_conf = pred_probs[np.arange(len(labels_arr)), labels_arr]
        pred = pred_probs.argmax(axis=1)
        idx = np.where(pred != labels_arr)[0]
        ranked = idx[np.argsort(self_conf[idx])]
        return np.asarray(ranked, dtype=np.int64), "fallback: OOF disagreements ranked by self-confidence"

def _top_examples(
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    preds: Sequence[int],
    probs: np.ndarray,
    issue_indices: Sequence[int],
    limit: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    for rank, i in enumerate(list(issue_indices)[: int(limit)], start=1):
        rec = records[int(i)]
        given = int(labels_arr[int(i)])
        pred = int(preds_arr[int(i)])
        self_conf = float(probs[int(i), given])
        pred_conf = float(probs[int(i), pred])
        out.append(
            {
                "rank": rank,
                "index": int(i),
                "split": str(rec.get("_split", "")),
                "request_id": str(rec.get("request_id", rec.get("unique_id", ""))),
                "given_label": SUBTYPES[given],
                "suggested_label": SUBTYPES[pred],
                "self_confidence": round(self_conf, 6),
                "suggested_confidence": round(pred_conf, 6),
                "margin": round(pred_conf - self_conf, 6),
                "method": str(rec.get("method", "")),
                "path": str(rec.get("path", ""))[:160],
                "query": str(rec.get("query", ""))[:160],
                "ua": str(rec.get("ua", ""))[:160],
                "status": rec.get("status", ""),
                "crs_rule_ids": rec.get("crs_rule_ids", []),
            }
        )
    return out

def _write_top_csv(path: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    if not examples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(examples[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in examples:
            writer.writerow(dict(row))

def _rate(n: int, denom: int) -> float:
    return 100.0 * n / denom if denom else float("nan")

def _summaries(
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    preds: Sequence[int],
    probs: np.ndarray,
    issue_indices: Sequence[int],
) -> Dict[str, Any]:
    n = len(labels)
    issue_set = set(int(i) for i in issue_indices)
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    self_conf = probs[np.arange(n), y]
    class_counts = Counter(int(v) for v in y)
    issue_by_given: Dict[str, Dict[str, Any]] = {}
    for cls_idx, cls_name in enumerate(SUBTYPES):
        support = int(class_counts.get(cls_idx, 0))
        idxs = [i for i, yy in enumerate(y) if int(yy) == cls_idx]
        issue_count = sum(1 for i in idxs if i in issue_set)
        issue_by_given[cls_name] = {
            "support": support,
            "suspected_issues": int(issue_count),
            "issue_rate_pct": round(_rate(issue_count, support), 3),
            "mean_self_confidence": round(float(np.mean(self_conf[idxs])), 6) if idxs else None,
        }

    pair_counts: Counter[tuple[int, int]] = Counter()
    for i in issue_set:
        pair_counts[(int(y[i]), int(p[i]))] += 1
    top_pairs = [
        {
            "given_label": SUBTYPES[g],
            "suggested_label": SUBTYPES[s],
            "count": int(c),
            "rate_of_all_issues_pct": round(_rate(c, len(issue_set)), 3),
        }
        for (g, s), c in pair_counts.most_common(12)
    ]

    split_counts: Counter[str] = Counter(str(r.get("_split", "")) for r in records)
    issue_split_counts: Counter[str] = Counter(str(records[i].get("_split", "")) for i in issue_set)
    by_split = {
        split: {
            "support": int(split_counts[split]),
            "suspected_issues": int(issue_split_counts.get(split, 0)),
            "issue_rate_pct": round(_rate(issue_split_counts.get(split, 0), split_counts[split]), 3),
        }
        for split in sorted(split_counts)
    }

    return {
        "n_records": int(n),
        "n_suspected_issues": int(len(issue_set)),
        "issue_rate_pct": round(_rate(len(issue_set), n), 3),
        "oof_macro_f1": round(float(f1_score(y, p, average="macro", labels=list(range(len(SUBTYPES))))), 6),
        "oof_accuracy": round(float(accuracy_score(y, p)), 6),
        "mean_self_confidence": round(float(np.mean(self_conf)), 6),
        "median_self_confidence": round(float(np.median(self_conf)), 6),
        "per_given_label": issue_by_given,
        "top_confused_pairs_among_issues": top_pairs,
        "by_split": by_split,
    }

def _build_results_doc(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    cells = [
        {
            "row": "Cleanlab suspected CRS subtype issues",
            "col": "issue_rate",
            "value": float(summary["issue_rate_pct"]),
            "ci_95": None,
            "seeds": [int(metadata["seed"])],
        },
        {
            "row": "Cleanlab suspected CRS subtype issues",
            "col": "count",
            "value": int(summary["n_suspected_issues"]),
            "ci_95": None,
            "seeds": [int(metadata["seed"])],
        },
        {
            "row": "OOF TF-IDF+LogReg audit classifier",
            "col": "macro_f1",
            "value": round(100.0 * float(summary["oof_macro_f1"]), 3),
            "ci_95": None,
            "seeds": [int(metadata["seed"])],
        },
        {
            "row": "OOF TF-IDF+LogReg audit classifier",
            "col": "accuracy",
            "value": round(100.0 * float(summary["oof_accuracy"]), 3),
            "ci_95": None,
            "seeds": [int(metadata["seed"])],
        },
    ]
    return {"table": _TABLE, "cells": cells, "metadata": dict(metadata)}

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["owasp_train", "owasp_val", "owasp_test"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--analyzer", default="char_wb")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--class-weight-balanced", action="store_true", default=True)
    parser.add_argument("--no-class-weight-balanced", dest="class_weight_balanced", action="store_false")
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = ensure_dir(Path(args.out_dir))

    records = _load_records(args.splits, args.max_records)
    if not records:
        raise SystemExit("[44] no records loaded")
    labels = subtype_labels(records)
    counts = Counter(labels)
    if min(counts.values()) < int(args.folds):
        raise SystemExit(f"[44] folds={args.folds} exceeds smallest class count {min(counts.values())}")

    print(f"[44] loaded n={len(records)} records across splits={args.splits}")
    print("[44] class counts:", {SUBTYPES[k]: int(v) for k, v in sorted(counts.items())})
    texts = render_records(records, fields=FIELD_ORDER)
    probs, preds, fold_reports = _oof_probabilities(texts, labels, args)
    issue_indices, method = _cleanlab_issues(labels, probs)
    summary = _summaries(records, labels, preds, probs, issue_indices)
    examples = _top_examples(records, labels, preds, probs, issue_indices, args.top_n)

    metadata = {
        "commit": git_commit(),
        "date": "2026-06-02",
        "gpu": "cpu",
        "seed": int(args.seed),
        "folds": int(args.folds),
        "splits": list(args.splits),
        "max_records": args.max_records,
        "dataset_version": dataset_version(args.splits),
        "embedding_version": embedding_version(),
        "audit_method": method,
        "audit_classifier": "TF-IDF + LogisticRegression",

        "artifact_kind": "supporting_audit",
        "single_run_audit": True,
        "features": {
            "fields": list(FIELD_ORDER),
            "analyzer": str(args.analyzer),
            "ngram_range": [int(args.ngram_min), int(args.ngram_max)],
            "max_features": int(args.max_features),
        },
        "model": {
            "C": float(args.C),
            "solver": str(args.solver),
            "max_iter": int(args.max_iter),
            "class_weight": "balanced" if args.class_weight_balanced else None,
        },
        "fold_reports": fold_reports,
        "note": (
            "Statistical audit of suspected CRS-derived subtype label issues using "
            "out-of-fold probabilities; not human ground truth."
        ),
    }

    dump_json(out_dir / "results.json", _build_results_doc(summary, metadata))
    dump_json(out_dir / "summary.json", {"summary": summary, "metadata": metadata})
    dump_json(out_dir / "top_suspected_issues.json", {"examples": examples, "metadata": metadata})
    _write_top_csv(out_dir / "top_suspected_issues.csv", examples)
    append_run(
        stage="cleanlab_label_audit",
        config_path="scripts/44_cleanlab_label_audit.py",
        seed=int(args.seed),
        artifact=out_dir / "results.json",
        notes=f"n={len(records)} issue_rate={summary['issue_rate_pct']} method={method}",
        split_names=args.splits,
        extra={"max_records": args.max_records},
    )

    print(
        f"[44] suspected issues: {summary['n_suspected_issues']}/{summary['n_records']} "
        f"({summary['issue_rate_pct']}%)"
    )
    print(f"[44] OOF audit classifier: macro_f1={summary['oof_macro_f1']:.4f} acc={summary['oof_accuracy']:.4f}")
    print(f"[44] wrote {out_dir / 'results.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
