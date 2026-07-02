
from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.data.labels import SUBTYPES
from src.utils.io import dump_json, read_jsonl
from src.utils.paths import ROOT, SPLITS_DIR

__all__ = [
    "SUBTYPES",
    "FIELD_ORDER",
    "subtype_index",
    "load_split_records",
    "render_record",
    "render_records",
    "select_label_budget",
    "subtype_labels",
    "macro_f1",
    "per_class_f1",
    "accuracy",
    "detection_summary",
    "git_commit",
    "results_path",
    "build_results",
    "write_results_json",
    "synthetic_records",
]

FIELD_ORDER: Tuple[str, ...] = ("method", "path", "query", "ua", "status", "timing")

_SUBTYPE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(SUBTYPES)}

def subtype_index(subtype: str) -> int:
    try:
        return _SUBTYPE_TO_INDEX[subtype]
    except KeyError as exc:
        raise KeyError(
            f"unknown attack_subtype {subtype!r}; known subtypes are {SUBTYPES}"
        ) from exc

def load_split_records(
    split: str,
    *,
    splits_dir: "str | Path | None" = None,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    base = Path(splits_dir) if splits_dir is not None else SPLITS_DIR
    path = base / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"split JSONL not found: {path}. Run scripts/02_make_splits.py first."
        )
    records: List[Dict[str, Any]] = []
    for rec in read_jsonl(path):
        records.append(rec)
        if max_records is not None and len(records) >= int(max_records):
            break
    return records

def _field_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field, "")
    if value is None:
        return ""
    return str(value)

def render_record(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str] = FIELD_ORDER,
    sep: str = " ",
) -> str:
    parts = [f"{f}={_field_text(record, f)}" for f in fields]
    return sep.join(parts)

def render_records(
    records: Iterable[Mapping[str, Any]],
    *,
    fields: Sequence[str] = FIELD_ORDER,
    sep: str = " ",
) -> List[str]:
    return [render_record(r, fields=fields, sep=sep) for r in records]

def subtype_labels(records: Sequence[Mapping[str, Any]]) -> List[int]:
    return [subtype_index(str(r["attack_subtype"])) for r in records]

def select_label_budget(n_records: int, budget: float) -> List[int]:
    if not (0.0 <= budget <= 1.0):
        raise ValueError(f"label_budget must be in [0, 1], got {budget}")
    k = int(round(float(budget) * int(n_records)))
    k = max(0, min(k, int(n_records)))
    return list(range(k))

def per_class_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int = len(SUBTYPES),
) -> List[float]:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        denom = precision + recall
        f1s.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    return f1s

def macro_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int = len(SUBTYPES),
) -> float:
    if not labels:
        return float("nan")
    f1s = per_class_f1(preds, labels, num_classes)
    return float(sum(f1s) / len(f1s)) if f1s else float("nan")

def class_support(
    labels: Sequence[int],
    num_classes: int = len(SUBTYPES),
) -> List[int]:
    counts = [0] * num_classes
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1
    return counts

def weighted_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int = len(SUBTYPES),
) -> float:
    if not labels:
        return float("nan")
    f1s = per_class_f1(preds, labels, num_classes)
    support = class_support(labels, num_classes)
    total = sum(support)
    if total == 0:
        return float("nan")
    return float(sum(f * s for f, s in zip(f1s, support)) / total)

def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return float("nan")
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    return correct / len(labels)

def detection_summary(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    if not scores:
        return {"mean_score": float("nan"), "n": 0}
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n if n else 0.0
    ordered = sorted(scores)
    median = ordered[n // 2]
    return {
        "mean_score": float(mean),
        "std_score": float(var ** 0.5),
        "median_score": float(median),
        "min_score": float(ordered[0]),
        "max_score": float(ordered[-1]),
        "n": int(n),
    }

def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def results_path(cfg: Mapping[str, Any]) -> Path:
    raw = cfg.get("results_path")
    if not raw:
        baseline = str(cfg.get("baseline", "baseline"))
        raw = f"results/table_03_rq1_baselines/{baseline}.json"
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)

def build_results(
    *,
    table: str,
    cells: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    norm_cells: List[Dict[str, Any]] = []
    for c in cells:
        norm_cells.append(
            {
                "row": c["row"],
                "col": c["col"],
                "value": c["value"],
                "ci_95": c.get("ci_95"),
                "seeds": list(c.get("seeds", [])),
            }
        )
    return {"table": table, "cells": norm_cells, "metadata": dict(metadata)}

def write_results_json(
    cfg: Mapping[str, Any],
    *,
    row: str,
    col: str,
    value: float,
    seed: int,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    table: str = "table_03_rq1_baselines",
    per_class: Optional[Mapping[str, float]] = None,
    path: "str | Path | None" = None,
    write_cell: bool = True,
    preds: Optional[Sequence[int]] = None,
    labels: Optional[Sequence[int]] = None,
    split: str = "owasp_test",
) -> Path:
    out_path = Path(path) if path is not None else results_path(cfg)

    try:
        from src.utils.runlog import dataset_version, embedding_version

        _data_ver, _emb_ver = dataset_version(), embedding_version()
    except Exception:
        _data_ver = _emb_ver = "absent"
    metadata: Dict[str, Any] = {
        "commit": git_commit(),
        "date": _dt.date.today().isoformat(),
        "gpu": "cpu",
        "dataset_version": _data_ver,
        "embedding_version": _emb_ver,
        "baseline": str(cfg.get("baseline", "unknown")),
        "family": str(cfg.get("family", "unknown")),
        "exploratory_single_seed": True,
        "note": "Single-seed baseline run; Table 3 aggregates seeds 42..46 with "
        "95% CIs via the aggregation stage (src.eval.aggregate).",
    }

    _cap = cfg.get("eval_max_records")
    metadata["eval_capped"] = _cap is not None
    if _cap is not None:
        metadata["max_records"] = int(_cap)
    if per_class is not None:
        metadata["per_class_f1"] = {k: round(float(v), 6) for k, v in per_class.items()}
    if extra_metadata:
        metadata.update(extra_metadata)

    if write_cell:

        from src.eval.aggregate import write_cell_file

        table_dir = out_path.parent
        write_cell_file(
            table_dir,
            table=table,
            row=row,
            col=col,
            seed=int(seed),
            value=value,
            metadata=metadata,
        )

        if preds is not None and labels is not None and not metadata.get("eval_capped"):
            from src.eval.predictions_io import dump_predictions, dump_shared_labels

            dump_shared_labels(
                table_dir, split=str(split), labels=labels,
                subtypes=list(SUBTYPES), num_classes=len(SUBTYPES),
                metadata={"commit": git_commit()},
            )
            dump_predictions(
                table_dir, row=row, col=col, seed=int(seed), split=str(split),
                preds=preds, labels=labels, num_classes=len(SUBTYPES),
                metric=str(metadata.get("metric", "macro_f1")), value=value,
                metadata={"commit": git_commit(),
                          "baseline": str(metadata.get("baseline", "unknown")),
                          "deterministic": True},
            )

    results = build_results(
        table=table,
        cells=[{"row": row, "col": col, "value": value, "ci_95": None, "seeds": [int(seed)]}],
        metadata=metadata,
    )
    dump_json(out_path, results)
    return out_path

def synthetic_records(n: int = 64, *, seed: int = 0) -> List[Dict[str, Any]]:
    import random

    rng = random.Random(seed)
    payloads = {
        "sql_injection": ("/login.php", "id=1' OR '1'='1", ["942100"]),
        "rce": ("/cgi-bin/x", "cmd=;cat /etc/passwd", ["932100"]),
        "php_injection": ("/index.php", "page=php://input", ["933150"]),
        "xss": ("/search", "q=<script>alert(1)</script>", ["941100"]),
        "path_traversal": ("/files", "f=../../../../etc/passwd", ["930110"]),
        "lfi": ("/wp-config.php", "", ["930130"]),
        "scanner": ("/admin", "", ["913100"]),
        "protocol": ("/robots.txt", "", ["920210"]),
    }
    severities = ["CRITICAL", "WARNING", "ERROR", "NOTICE"]
    uas = [
        "Mozilla/5.0 (compatible; DotBot/1.2)",
        "sqlmap/1.5",
        "curl/7.68.0",
        "Mozilla/5.0 (X11; Linux x86_64)",
    ]
    records: List[Dict[str, Any]] = []
    for i in range(int(n)):
        subtype = SUBTYPES[i % len(SUBTYPES)]
        path, query, rules = payloads[subtype]
        records.append(
            {
                "attack_subtype": subtype,
                "label": "attack",
                "method": rng.choice(["GET", "POST", "HEAD"]),
                "path": path,
                "query": query,
                "ua": rng.choice(uas),
                "status": rng.choice([200, 404, 403, 500]),
                "timing": rng.randint(10, 5000),
                "crs_rule_ids": rules,
                "crs_tags": [f"attack-{subtype.replace('_', '-')}", "OWASP_CRS"],
                "severities": [rng.choice(severities)],
                "client_ip": f"10.0.{i % 8}.{rng.randint(1, 254)}",
                "n_rules": len(rules),
            }
        )
    return records
