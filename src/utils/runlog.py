
from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.utils.paths import MODELS_DIR, RESULTS_DIR, ROOT, SPLITS_DIR

__all__ = [
    "RUNS_CSV",
    "RUN_FIELDS",
    "git_commit",
    "dataset_version",
    "embedding_version",
    "append_run",
]

RUNS_CSV: Path = RESULTS_DIR / "runs.csv"

_DEFAULT_FASTTEXT = MODELS_DIR / "detector" / "fasttext" / "weblog_fasttext.model"

RUN_FIELDS: Sequence[str] = (
    "timestamp_utc",
    "stage",
    "git_commit",
    "config_path",
    "seed",
    "dataset_version",
    "embedding_version",
    "command_line",
    "artifact",
    "notes",
)

def git_commit(default: str = "UNKNOWN") -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if status else rev
    except Exception:
        return default

def _hash_file(path: Path, *, chunk: int = 1 << 20) -> str:
    if not path.is_file():
        return "absent"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:12]

def dataset_version(split_names: Optional[Sequence[str]] = None) -> str:
    names = list(
        split_names
        if split_names is not None
        else ("owasp_train", "owasp_val", "owasp_test", "weblog_pretrain")
    )
    parts = []
    for name in names:
        parts.append(f"{name}:{_hash_file(SPLITS_DIR / f'{name}.jsonl')}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return digest

def embedding_version(fasttext_path: Optional[Path] = None) -> str:
    base = Path(fasttext_path) if fasttext_path is not None else _DEFAULT_FASTTEXT
    if not base.exists():
        return "absent"
    parts = []
    for sib in sorted(base.parent.glob(base.name + "*")):
        if sib.is_file():
            parts.append(f"{sib.name}:{_hash_file(sib)}")
    if not parts:
        return "absent"
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]

def append_run(
    *,
    stage: str,
    config_path: "str | Path",
    seed: int,
    artifact: "str | Path" = "",
    notes: str = "",
    command_line: Optional[str] = None,
    split_names: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_line if command_line is not None else " ".join(sys.argv)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": str(stage),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "seed": int(seed),
        "dataset_version": dataset_version(split_names),
        "embedding_version": embedding_version(),
        "command_line": cmd,
        "artifact": str(artifact),
        "notes": str(notes),
    }
    if extra:

        extra_str = "; ".join(f"{k}={v}" for k, v in extra.items())
        row["notes"] = f"{row['notes']} {extra_str}".strip()

    write_header = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RUN_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return RUNS_CSV
