
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.baselines._common import macro_f1
from src.eval.aggregate import _slug as slug
from src.utils.io import dump_json, load_json

__all__ = [
    "PREDICTIONS_DIRNAME",
    "predictions_dir",
    "labels_file_path",
    "prediction_file_path",
    "labels_sha256",
    "dump_shared_labels",
    "dump_predictions",
    "load_shared_labels",
    "load_predictions",
]

PREDICTIONS_DIRNAME = "predictions"

def predictions_dir(table_dir: Path) -> Path:
    return Path(table_dir) / PREDICTIONS_DIRNAME

def labels_file_path(table_dir: Path, split: str) -> Path:
    return predictions_dir(table_dir) / f"labels__{slug(split)}.json"

def prediction_file_path(table_dir: Path, row: str, col: str, seed: int) -> Path:
    return predictions_dir(table_dir) / f"{slug(row)}__{slug(col)}__seed{int(seed):02d}.json"

def labels_sha256(labels: Sequence[int]) -> str:
    canon = json.dumps([int(y) for y in labels], separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()

def dump_shared_labels(
    table_dir: Path,
    *,
    split: str,
    labels: Sequence[int],
    subtypes: Sequence[str],
    num_classes: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    labels = [int(y) for y in labels]
    sha = labels_sha256(labels)
    out_path = labels_file_path(table_dir, split)
    if out_path.exists():
        existing = load_json(out_path)
        if str(existing.get("labels_sha256")) != sha:
            raise ValueError(
                f"shared-labels mismatch at {out_path}: on-disk hash "
                f"{existing.get('labels_sha256')!r} != new {sha!r}. The test split "
                f"changed; remove stale prediction sidecars before re-writing."
            )
    payload: Dict[str, Any] = {
        "split": str(split),
        "n": len(labels),
        "num_classes": int(num_classes),
        "subtypes": [str(s) for s in subtypes],
        "labels": labels,
        "labels_sha256": sha,
        "metadata": dict(metadata or {}),
    }
    dump_json(out_path, payload)
    return out_path

def dump_predictions(
    table_dir: Path,
    *,
    row: str,
    col: str,
    seed: int,
    split: str,
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int,
    metric: str,
    value: float,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    preds = [int(p) for p in preds]
    labels = [int(y) for y in labels]
    if len(preds) != len(labels):
        raise ValueError(
            f"preds ({len(preds)}) and labels ({len(labels)}) length mismatch for "
            f"row={row!r} col={col!r} seed={seed}."
        )
    recomputed = round(100.0 * macro_f1(preds, labels, num_classes), 4)
    if abs(recomputed - float(value)) > 1e-3:
        raise ValueError(
            f"preds for row={row!r} col={col!r} seed={seed} recompute macro-F1 "
            f"{recomputed} but the cell value is {value} (>1e-3 apart). The "
            f"prediction vector is inconsistent with the reported cell."
        )
    payload: Dict[str, Any] = {
        "row": str(row),
        "col": str(col),
        "seed": int(seed),
        "split": str(split),
        "n": len(preds),
        "num_classes": int(num_classes),
        "preds": preds,
        "labels_sha256": labels_sha256(labels),
        "metric": str(metric),
        "macro_f1": float(value),
        "metadata": dict(metadata or {}),
    }
    out_path = prediction_file_path(table_dir, row, col, seed)
    dump_json(out_path, payload)
    return out_path

def load_shared_labels(table_dir: Path, split: str) -> Dict[str, Any]:
    path = labels_file_path(table_dir, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"no shared-labels sidecar at {path}; run the evaluation that writes "
            f"predictions first (scripts/12_evaluate.py / scripts/20_run_baseline.py)."
        )
    return load_json(path)

def load_predictions(
    table_dir: Path,
    *,
    row: str,
    col: str,
    seed: int,
    expected_labels_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    path = prediction_file_path(table_dir, row, col, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"no prediction sidecar at {path} for row={row!r} col={col!r} seed={seed}."
        )
    doc = load_json(path)
    if expected_labels_sha256 is not None and str(doc.get("labels_sha256")) != str(
        expected_labels_sha256
    ):
        raise ValueError(
            f"label-hash mismatch at {path}: preds were scored on a different test "
            f"split (hash {doc.get('labels_sha256')!r}) than the shared labels "
            f"(hash {expected_labels_sha256!r}). Re-run evaluation to refresh the sidecar."
        )
    return doc
