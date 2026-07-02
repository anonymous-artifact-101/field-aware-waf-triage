
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.eval.metrics import bootstrap_ci, mean_ci95
from src.utils.io import dump_json, load_json

__all__ = [
    "CELLS_DIRNAME",
    "cells_dir",
    "cell_file_path",
    "write_cell_file",
    "load_cell_files",
    "aggregate_cells",
    "aggregate_table",
]

CELLS_DIRNAME = "cells"

_SLUG_KEEP = re.compile(r"[^0-9A-Za-z._-]+")

def _slug(text: str) -> str:
    s = _SLUG_KEEP.sub("-", str(text).strip().lower()).strip("-")
    return s or "_"

def cells_dir(table_dir: Path) -> Path:
    return Path(table_dir) / CELLS_DIRNAME

def cell_file_path(table_dir: Path, row: str, col: str, seed: int) -> Path:
    return cells_dir(table_dir) / f"{_slug(row)}__{_slug(col)}__seed{int(seed):02d}.json"

def write_cell_file(
    table_dir: Path,
    *,
    table: str,
    row: str,
    col: str,
    seed: int,
    value: float,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "table": str(table),
        "row": str(row),
        "col": str(col),
        "seed": int(seed),
        "value": value,
        "metadata": dict(metadata or {}),
    }
    out_path = cell_file_path(Path(table_dir), row, col, seed)
    dump_json(out_path, payload)
    return out_path

def load_cell_files(table_dir: Path) -> List[Dict[str, Any]]:
    cdir = cells_dir(Path(table_dir))
    if not cdir.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(cdir.glob("*.json")):
        doc = load_json(path)
        if not isinstance(doc, Mapping) or "row" not in doc or "col" not in doc or "value" not in doc:
            raise ValueError(
                f"malformed cell file {path}: expected an object with "
                f"'row', 'col', 'value', 'seed' keys."
            )
        out.append(dict(doc))
    return out

def _merge_metadata(
    per_seed_metas: Sequence[Mapping[str, Any]],
    *,
    n_seeds: int = 1,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    is_single_run = any(bool(m.get("single_run_measured")) for m in per_seed_metas)
    merged["single_run_measured"] = is_single_run

    if any("deterministic" in m for m in per_seed_metas):
        merged["deterministic"] = all(bool(m.get("deterministic")) for m in per_seed_metas)

    merged["exploratory_single_seed"] = (n_seeds < 2) and not is_single_run
    if any("eval_capped" in m for m in per_seed_metas):
        merged["eval_capped"] = any(bool(m.get("eval_capped")) for m in per_seed_metas)

    if any("checkpoint_loaded" in m for m in per_seed_metas):
        merged["checkpoint_loaded"] = all(bool(m.get("checkpoint_loaded")) for m in per_seed_metas)
    for key in ("metric", "mode"):
        vals = {str(m[key]) for m in per_seed_metas if key in m}
        if len(vals) == 1:
            merged[key] = next(iter(vals))
        elif len(vals) > 1:
            merged[key] = sorted(vals)

    wf1 = [float(m["weighted_f1"]) for m in per_seed_metas if m.get("weighted_f1") is not None]
    if wf1:
        merged["weighted_f1"] = round(sum(wf1) / len(wf1), 4)

    for key in ("footprint_mb", "latency_ms", "p95_ms", "vector_size", "min_n", "max_n"):
        vals = {m[key] for m in per_seed_metas if m.get(key) is not None}
        if len(vals) == 1:
            merged[key] = next(iter(vals))

    hw = [float(m["ci95_halfwidth"]) for m in per_seed_metas
          if m.get("ci95_halfwidth") is not None]
    if hw:
        merged["ci95_halfwidth"] = round(sum(hw) / len(hw), 6)
    nr = [int(m["n_runs"]) for m in per_seed_metas if m.get("n_runs") is not None]
    if nr:
        merged["n_runs"] = max(nr)
    return merged

def aggregate_cells(
    cell_docs: Sequence[Mapping[str, Any]],
    *,
    method: str = "student_t",
    round_to: Optional[int] = 1,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if method not in ("student_t", "bootstrap"):
        raise ValueError(f"unknown CI method {method!r}; use 'student_t' or 'bootstrap'.")

    grouped: "dict[Tuple[str, str], dict]" = {}
    order: List[Tuple[str, str]] = []
    for doc in cell_docs:
        key = (str(doc["row"]), str(doc["col"]))
        if key not in grouped:
            grouped[key] = {"seeds": [], "values": [], "metas": []}
            order.append(key)
        seed = int(doc.get("seed", 0))
        grouped[key]["seeds"].append(seed)
        grouped[key]["values"].append(float(doc["value"]))
        grouped[key]["metas"].append(dict(doc.get("metadata", {})))

    def _fmt(x: float) -> float:
        if round_to is None:
            return float(x)
        if isinstance(x, float) and x != x:
            return float(x)
        return round(float(x), round_to)

    cells: List[Dict[str, Any]] = []
    cell_flags: Dict[str, Any] = {}
    for key in order:
        row, col = key
        bucket = grouped[key]

        paired = sorted(zip(bucket["seeds"], bucket["values"], bucket["metas"]), key=lambda t: t[0])
        seeds = [s for s, _, _ in paired]
        values = [v for _, v, _ in paired]
        metas = [m for _, _, m in paired]

        if method == "bootstrap":
            mean, lo, hi = bootstrap_ci(values, n=bootstrap_n, seed=bootstrap_seed)
        else:
            mean, (lo, hi) = mean_ci95(values)

        merged_meta = _merge_metadata(metas, n_seeds=len(set(seeds)))

        if merged_meta.get("ci95_halfwidth") is not None:
            hw = float(merged_meta["ci95_halfwidth"])
            ci_95 = [_fmt(mean - hw), _fmt(mean + hw)]
        elif merged_meta.get("single_run_measured"):
            ci_95 = None
        else:
            ci_95 = [_fmt(lo), _fmt(hi)]
        cells.append(
            {
                "row": row,
                "col": col,
                "value": _fmt(mean),
                "ci_95": ci_95,
                "seeds": seeds,
            }
        )
        cell_flags[f"{row} | {col}"] = merged_meta

    agg_metadata: Dict[str, Any] = {"ci_method": method, "cell_flags": cell_flags}
    return cells, agg_metadata

def aggregate_table(
    table_dir: "str | Path",
    *,
    table_name: Optional[str] = None,
    method: str = "student_t",
    round_to: Optional[int] = 1,
    commit: str = "UNKNOWN",
    gpu: str = "CPU",
    run_date: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    out_path: "str | Path | None" = None,
) -> Path:
    tdir = Path(table_dir)
    name = table_name if table_name is not None else tdir.name

    cell_docs = load_cell_files(tdir)
    if not cell_docs:
        raise FileNotFoundError(
            f"no per-seed cell files under {cells_dir(tdir)}. "
            f"Run scripts/12_evaluate.py or scripts/20_run_baseline.py first "
            f"(they write {CELLS_DIRNAME}/<row>__<col>__seedNN.json)."
        )

    cells, agg_metadata = aggregate_cells(cell_docs, method=method, round_to=round_to)

    metadata: Dict[str, Any] = {
        "commit": commit,
        "date": run_date if run_date is not None else date.today().isoformat(),
        "gpu": gpu,
        "n_cell_files": len(cell_docs),
    }
    metadata.update(agg_metadata)
    if extra_metadata:
        for k, v in extra_metadata.items():
            metadata.setdefault(k, v)

    doc = {"table": name, "cells": cells, "metadata": metadata}
    target = Path(out_path) if out_path is not None else (tdir / "results.json")
    dump_json(target, doc)
    return target
