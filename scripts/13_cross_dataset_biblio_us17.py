
from __future__ import annotations

import argparse
import math
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.eval.aggregate import load_cell_files, write_cell_file
from src.eval.metrics import mean_ci95
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR, ROOT
from src.utils.runlog import append_run, dataset_version, embedding_version
from src.utils.seeds import set_seed

_TABLE = "table_09_cross_dataset_biblio_us17"
_UNPACKED = ROOT / "download" / "5.Biblio-US17" / "Biblio-US17"
_ARCHIVE = ROOT / "download" / "5.Biblio-US17.tar.gz"
_DEFAULT_SOURCE = _UNPACKED if _UNPACKED.is_dir() else _ARCHIVE
_ROW = "Proposed (zero-shot transfer)"

_ROW_MASKED = "Proposed (request-fields only, status-masked)"

def _to_int(token: str) -> int:
    try:
        return int(float(str(token).strip()))
    except ValueError:
        return 0

def _split_uri(uri: str) -> Tuple[str, str]:
    uri = str(uri).strip()
    try:
        parsed = urlsplit(uri)
        path = parsed.path or "/"
        query = parsed.query
    except ValueError:
        path, _, query = uri.partition("?")
        path = path or "/"
    return path, query

def _clip_field(value: str, max_chars: Optional[int]) -> str:
    if max_chars is None or int(max_chars) <= 0:
        return value
    return str(value)[: int(max_chars)]

def _day_from_identifier(identifier: str) -> str:

    s = identifier.strip().strip("[]")
    parts = s.split("-")
    if len(parts) >= 2:
        return f"2017-{parts[0]}-{parts[1]}"
    return "2017"

def _parse_biblio_line(
    line: bytes,
    *,
    label: str,
    source_set: str,
    max_field_chars: Optional[int] = 2048,
) -> Optional[Dict[str, Any]]:
    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
    if not text:
        return None
    parts = text.split("\t")
    if len(parts) < 6:
        return None
    identifier, method, uri, protocol, status, size = [p.strip() for p in parts[:6]]
    if not identifier or not method:
        return None
    path, query = _split_uri(uri)
    path = _clip_field(path, max_field_chars)
    query = _clip_field(query, max_field_chars)
    ident_slug = identifier.strip("[]")
    return {
        "unique_id": f"biblio-us17-{ident_slug}",
        "day": _day_from_identifier(identifier),
        "method": method,
        "path": path,
        "query": query,
        "ua": "",
        "status": _to_int(status),
        "timing": 0,
        "bytes": _to_int(size),
        "protocol": protocol.rstrip('"'),
        "label": label,
        "attack_subtype": "protocol" if label == "attack" else "",
        "biblio_id": identifier,
        "biblio_set": source_set,
    }

def _iter_biblio_records(
    source: Path,
    *,
    attack_stride: int = 1,
    benign_stride: int = 1,
    clean_per_day: Optional[int] = None,
    max_attack_records: Optional[int] = None,
    max_field_chars: Optional[int] = 2048,
) -> Iterator[Tuple[Dict[str, Any], int]]:
    if not source.exists():
        raise SystemExit(f"[13_cross_dataset_biblio_us17] Biblio-US17 source not found: {source}")
    attack_stride = max(1, int(attack_stride))
    benign_stride = max(1, int(benign_stride))
    attack_seen = 0
    benign_seen = 0
    attack_emitted = 0

    def consume_lines(
        lines: Iterable[bytes],
        *,
        is_clean: bool,
        is_attack: bool,
        source_set: str,
        label: str,
    ) -> Iterator[Tuple[Dict[str, Any], int]]:
        nonlocal attack_seen, benign_seen, attack_emitted
        binary = 0 if is_clean else 1
        parsed_in_file = 0
        for raw_line in lines:
            if is_clean and clean_per_day is not None and parsed_in_file >= clean_per_day:
                break
            if is_attack and max_attack_records is not None and attack_emitted >= max_attack_records:
                break
            rec = _parse_biblio_line(
                raw_line,
                label=label,
                source_set=source_set,
                max_field_chars=max_field_chars,
            )
            if rec is None:
                continue
            parsed_in_file += 1
            if is_attack:
                take = (attack_seen % attack_stride) == 0
                attack_seen += 1
                if take:
                    attack_emitted += 1
                    yield rec, binary
            else:
                take = (benign_seen % benign_stride) == 0
                benign_seen += 1
                if take:
                    yield rec, binary

    if source.is_dir():
        for source_set, subdir, suffix in (
            ("attack", "attack", ".att"),
            ("clean", "clean", ".cl"),
        ):
            is_attack = source_set == "attack"
            is_clean = source_set == "clean"
            if is_attack and max_attack_records is not None and attack_emitted >= max_attack_records:
                continue
            label = "attack" if is_attack else "benign"
            for path in sorted((source / subdir).glob(f"*{suffix}")):
                if is_attack and max_attack_records is not None and attack_emitted >= max_attack_records:
                    break
                with path.open("rb") as fh:
                    yield from consume_lines(
                        fh,
                        is_clean=is_clean,
                        is_attack=is_attack,
                        source_set=source_set,
                        label=label,
                    )
        return

    if not source.is_file():
        raise SystemExit(f"[13_cross_dataset_biblio_us17] source is not a file or directory: {source}")

    with tarfile.open(source, "r:*") as tf:
        for member in tf:
            name = member.name
            if member.isdir():
                continue
            is_clean = name.startswith("Biblio-US17/clean/") and name.endswith(".cl")
            is_attack = name.startswith("Biblio-US17/attack/") and name.endswith(".att")
            if not (is_clean or is_attack):
                continue
            if is_attack and max_attack_records is not None and attack_emitted >= max_attack_records:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            source_set = "clean" if is_clean else "attack"
            label = "benign" if is_clean else "attack"
            yield from consume_lines(
                fh,
                is_clean=is_clean,
                is_attack=is_attack,
                source_set=source_set,
                label=label,
            )

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
    if n_pos == 0 or n_neg == 0:
        return {"threshold": math.nan, "tpr": math.nan, "fpr": math.nan,
                "precision": math.nan, "tp": 0, "fp": 0}
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best: Dict[str, float] = {"j": -float("inf"), "threshold": float(scores[order[0]])}
    tp = fp = 0
    i = 0
    while i < len(order):
        threshold = scores[order[i]]
        while i < len(order) and scores[order[i]] == threshold:
            if labels[order[i]] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        j = tpr - fpr
        if j > best["j"]:
            best = {"j": j, "threshold": float(threshold), "tpr": tpr, "fpr": fpr,
                    "precision": precision, "tp": tp, "fp": fp}
    return best

def _mask_status(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out["status"] = ""
    return out

def _score_stream(
    detector,
    records: Iterable[Tuple[Dict[str, Any], int]],
    *,
    batch_size: int,
    mask_status: bool = False,
) -> Tuple[List[float], List[int]]:
    scores: List[float] = []
    labels: List[int] = []
    batch: List[Dict[str, Any]] = []
    batch_labels: List[int] = []
    total = 0

    def _flush(b: List[Dict[str, Any]]) -> List[float]:
        recs = [_mask_status(r) for r in b] if mask_status else b
        return [float(s) for s in detector.anomaly_score(recs)]

    for rec, y in records:
        batch.append(rec)
        batch_labels.append(y)
        if len(batch) >= batch_size:
            scores.extend(_flush(batch))
            labels.extend(batch_labels)
            total += len(batch)
            print(f"[13_cross_dataset_biblio_us17] scored {total} records...")
            batch = []
            batch_labels = []
    if batch:
        scores.extend(_flush(batch))
        labels.extend(batch_labels)
        total += len(batch)
        print(f"[13_cross_dataset_biblio_us17] scored {total} records.")
    return scores, labels

def _aggregate(commit: str) -> Path:
    table_dir = RESULTS_DIR / _TABLE
    cells = load_cell_files(table_dir)
    if not cells:
        raise SystemExit(f"[13_cross_dataset_biblio_us17] no cells under {table_dir/'cells'}")

    per_row: Dict[str, Dict[str, List[float]]] = {}
    row_seeds: Dict[str, List[int]] = {}
    embs = set()
    for c in cells:
        md = c.get("metadata", {})
        tm = md.get("threshold_metrics", {})
        row = c["row"]
        masked = bool(md.get("status_masked"))
        suffix = " (status-masked)" if masked else ""
        s = per_row.setdefault(row, {f"roc_auc{suffix}": [], f"pr_auc{suffix}": [],
                                     f"tpr{suffix}": [], f"fpr{suffix}": [],
                                     f"precision{suffix}": []})
        s[f"roc_auc{suffix}"].append(float(c["value"]))
        s[f"pr_auc{suffix}"].append(float(md["pr_auc"]))
        s[f"tpr{suffix}"].append(float(tm["tpr"]))
        s[f"fpr{suffix}"].append(float(tm["fpr"]))
        s[f"precision{suffix}"].append(float(tm["precision"]))
        embs.add(md.get("embedding_version"))
        row_seeds.setdefault(row, []).append(int(c["seed"]))
    if len(embs) > 1:
        raise SystemExit(f"[13_cross_dataset_biblio_us17] mixed embeddings in cells: {embs}")
    out_cells = []
    for row in sorted(per_row):
        seeds = sorted(set(row_seeds[row]))
        for col, vals in per_row[row].items():
            m, (lo, hi) = mean_ci95(vals)
            out_cells.append({"row": row, "col": col, "value": round(m, 4),
                              "ci_95": [round(lo, 4), round(hi, 4)], "seeds": seeds})
    seeds = sorted({s for ss in row_seeds.values() for s in ss})
    last = cells[-1].get("metadata", {})
    n_total = int(last.get("n_total", 0))
    n_attack = int(last.get("n_attack", 0))
    doc = {
        "table": _TABLE,
        "cells": out_cells,
        "metadata": {
            "commit": commit,
            "date": last.get("date") or "",
            "gpu": "cpu",
            "ci_method": "student_t",
            "n_cell_files": len(cells),
            "dataset": "biblio-us17",
            "dataset_version": last.get("dataset_version"),
            "embedding_version": embs.pop() if embs else None,
            "n_total": n_total,
            "n_attack": n_attack,
            "n_benign": int(last.get("n_benign", 0)),
            "base_rate_pos": round(n_attack / max(1, n_total), 4),
            "sampling": last.get("sampling", "systematic"),
            "attack_stride": int(last.get("attack_stride", 1)),
            "benign_stride": int(last.get("benign_stride", 1)),
            "max_field_chars": last.get("max_field_chars"),
            "youden_j_threshold": last.get("threshold_metrics", {}).get("threshold"),
            "note": "Zero-shot OWASP/Kaggle -> Biblio-US17 stress probe; detector "
                    "fit uses no Biblio record. Evaluation uses a deterministic "
                    "class-balanced systematic sample (every Nth record per class). "
                    "Full-field AUC is status-shortcut-inflated; the status-masked "
                    "row is the honest request-field transfer number.",
        },
    }
    out = table_dir / "results.json"
    dump_json(out, doc)
    return out

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Biblio-US17 zero-shot transfer stress probe."
    )
    parser.add_argument("--config", default="configs/finetune/label_0pct.yaml")
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                        help="One or more seeds. The probe is deterministic in the "
                             "seed, so multiple seeds get identical (zero-width-CI) "
                             "cells written from a single fit+score.")
    parser.add_argument("--archive", default=str(_DEFAULT_SOURCE),
                        help="Biblio-US17 source: either the unpacked Biblio-US17 directory "
                             "or the original .tar.gz archive.")

    parser.add_argument("--attack-stride", type=int, default=6,
                        help="Keep every Nth attack record (systematic sample; 1=all).")
    parser.add_argument("--benign-stride", type=int, default=777,
                        help="Keep every Nth benign record (systematic sample; 1=all).")
    parser.add_argument("--clean-per-day", type=int, default=None,
                        help="Optional legacy per-file benign prefix cap (applied before stride).")
    parser.add_argument("--max-attack-records", type=int, default=None,
                        help="Cap attack records for smoke runs.")
    parser.add_argument("--max-field-chars", type=int, default=2048,
                        help="Deterministically truncate Biblio path/query fields before "
                             "FastText encoding; prevents pathological multi-kB URIs from "
                             "dominating runtime. Use 0 to disable.")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--variants", choices=("both", "full", "masked"), default="both",
                        help="Score the full 6-field detector, the status-masked control, "
                             "or both from a single archive read (default: both).")
    parser.add_argument("--mask-status", action="store_true",
                        help="Deprecated alias for --variants masked (kept for back-compat).")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)

    if args.aggregate:
        out = _aggregate(git_commit())
        print(f"[13_cross_dataset_biblio_us17] aggregated results -> {out}")
        return 0

    cfg = dict(load_config(args.config))
    seeds = sorted({int(s) for s in args.seed})

    set_seed(seeds[0])
    ft_model, train_recs, _ = load_detector_inputs(cfg)
    detector = fit_detector(cfg, ft_model, train_recs, seeds[0])
    if detector.mode != "unsupervised":
        raise SystemExit("[13_cross_dataset_biblio_us17] use label_0pct/unsupervised config.")

    archive = Path(args.archive)
    records = [
        (rec, y) for rec, y in _iter_biblio_records(
            archive,
            attack_stride=int(args.attack_stride),
            benign_stride=int(args.benign_stride),
            clean_per_day=(int(args.clean_per_day) if args.clean_per_day is not None else None),
            max_attack_records=args.max_attack_records,
            max_field_chars=(int(args.max_field_chars) if int(args.max_field_chars) > 0 else None),
        )
    ]
    print(f"[13_cross_dataset_biblio_us17] loaded {len(records)} Biblio records "
          f"(systematic: attack_stride={args.attack_stride}, benign_stride={args.benign_stride})")

    variants = ["full", "masked"] if args.variants == "both" else [args.variants]
    if args.mask_status:
        variants = ["masked"]

    table_dir = RESULTS_DIR / _TABLE
    for variant in variants:
        mask_status = variant == "masked"
        scores, labels = _score_stream(detector, records, batch_size=int(args.batch_size),
                                       mask_status=mask_status)
        auc = _roc_auc(scores, labels)
        pr_auc = _pr_auc(scores, labels)
        thr = _threshold_metrics(scores, labels)

        n_attack = int(sum(labels))
        n_benign = len(labels) - n_attack
        vlabel = "request-fields-only (status-masked)" if mask_status else "full 6-field"
        print(f"[13_cross_dataset_biblio_us17] zero-shot Biblio-US17 [{vlabel}]: n={len(labels)} "
              f"(attack={n_attack}, benign={n_benign}) ROC-AUC={auc:.4f} PR-AUC={pr_auc:.4f}")
        print(f"[13_cross_dataset_biblio_us17] @best-J: TPR={thr['tpr']:.3f} "
              f"FPR={thr['fpr']:.3f} precision={thr['precision']:.3f}")

        if args.no_write:
            continue

        row = _ROW_MASKED if mask_status else _ROW
        col = "ROC-AUC (status-masked)" if mask_status else "ROC-AUC"
        note = (
            "Zero-shot transfer, STATUS-MASKED control: detector trained on OWASP/Kaggle, "
            "NO Biblio-US17 fit; the response-code field is blanked so this AUC reflects "
            "ONLY request-field (method/path/query/ua/timing) transfer, not the Biblio "
            "benign=200 / attack=non-200 status giveaway."
            if mask_status else
            "Zero-shot transfer: detector trained on OWASP/Kaggle, NO Biblio-US17 fit. "
            "Biblio-US17 uses real production HTTP requests; evaluation uses a "
            "deterministic class-balanced systematic sample (every Nth record per "
            "class across the whole corpus). NOTE: the full-field AUC is inflated by "
            "the status shortcut; see the status-masked row."
        )
        base_metadata = {
            "commit": git_commit(),
            "gpu": "cpu",
            "dataset_version": dataset_version(),
            "embedding_version": embedding_version(),
            "metric": "roc_auc",
            "mode": "unsupervised_zero_shot_transfer",
            "status_masked": mask_status,
            "checkpoint_loaded": True,
            "deterministic": True,

            "deterministic_seed_replicated": len(seeds) > 1,
            "eval_capped": args.max_attack_records is not None,
            "n_total": len(labels),
            "n_attack": n_attack,
            "n_benign": n_benign,
            "sampling": "systematic",
            "attack_stride": int(args.attack_stride),
            "benign_stride": int(args.benign_stride),
            "max_field_chars": (int(args.max_field_chars) if int(args.max_field_chars) > 0 else None),
            "pr_auc": round(float(pr_auc), 4),
            "threshold_metrics": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                  for k, v in thr.items() if k != "j"},
            "note": note,
        }
        for seed in seeds:
            write_cell_file(table_dir, table=_TABLE, row=row, col=col, seed=seed,
                            value=round(auc, 4), metadata=dict(base_metadata))
            try:
                append_run(stage="cross_dataset_biblio_us17", config_path=args.config, seed=seed,
                           artifact=str(table_dir),
                           notes=f"roc_auc={round(auc,4)}|n={len(labels)}|masked={mask_status}",
                           extra={"eval_capped": args.max_attack_records is not None,
                                  "status_masked": mask_status})
            except Exception as exc:
                print(f"[13_cross_dataset_biblio_us17] WARN ledger: {exc}")
        print(f"[13_cross_dataset_biblio_us17] wrote {variant} cells for seeds {seeds} "
              f"under {table_dir / 'cells'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
