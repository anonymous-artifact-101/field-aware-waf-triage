
from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPLITS = _REPO_ROOT / "data" / "splits"
_OUT = _REPO_ROOT / "results" / "table_02_dataset_profile" / "results.json"
_OWASP_DESCRIPTOR_BLOCKED_REQUESTS = 147_205

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"

def _jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def _split_counts(name: str) -> Dict[str, Any]:
    path = _SPLITS / f"{name}.jsonl"
    counts: Counter[str] = Counter()
    total = 0
    first_ts = None
    last_ts = None
    first_day = None
    last_day = None
    for rec in _jsonl_records(path):
        total += 1
        counts[str(rec.get("attack_subtype", "unknown"))] += 1
        ts = rec.get("timestamp")
        day = rec.get("day")
        if total == 1:
            first_ts = ts
            first_day = day
        last_ts = ts
        last_day = day
    return {
        "name": name,
        "total": total,
        "counts": dict(counts),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "first_day": first_day,
        "last_day": last_day,
    }

def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)

def main() -> int:
    splits = [_split_counts(name) for name in ("owasp_train", "owasp_val", "owasp_test")]
    cells: List[Dict[str, Any]] = []

    for split in splits:
        split_name = split["name"].replace("owasp_", "")
        total = int(split["total"])
        cells.append({
            "row": "OWASP split size",
            "col": split_name,
            "value": total,
            "ci_95": None,
            "seeds": [],
        })

        for key, value in (
            ("first_year", str(split["first_timestamp"])[:4]),
            ("first_month", str(split["first_timestamp"])[5:7]),
            ("first_day", str(split["first_timestamp"])[8:10]),
            ("last_year", str(split["last_timestamp"])[:4]),
            ("last_month", str(split["last_timestamp"])[5:7]),
            ("last_day", str(split["last_timestamp"])[8:10]),
        ):
            cells.append({
                "row": "OWASP split date",
                "col": f"{split_name}_{key}",
                "value": int(value),
                "ci_95": None,
                "seeds": [],
            })
        for subtype, count in sorted(split["counts"].items()):
            pct = 100.0 * float(count) / float(total) if total else 0.0
            cells.append({
                "row": subtype,
                "col": f"{split_name}_count",
                "value": int(count),
                "ci_95": None,
                "seeds": [],
            })
            cells.append({
                "row": subtype,
                "col": f"{split_name}_pct",
                "value": pct,
                "ci_95": None,
                "seeds": [],
            })

    owasp_total = sum(int(s["total"]) for s in splits)
    cells.append({
        "row": "OWASP ModSec",
        "col": "total_records",
        "value": owasp_total,
        "ci_95": None,
        "seeds": [],
    })
    cells.append({
        "row": "OWASP source descriptor",
        "col": "blocked_requests",
        "value": _OWASP_DESCRIPTOR_BLOCKED_REQUESTS,
        "ci_95": None,
        "seeds": [],
    })
    cells.append({
        "row": "OWASP parser minus descriptor",
        "col": "records",
        "value": owasp_total - _OWASP_DESCRIPTOR_BLOCKED_REQUESTS,
        "ci_95": None,
        "seeds": [],
    })
    weblog_path = _SPLITS / "weblog_pretrain.jsonl"
    if weblog_path.exists():
        cells.append({
            "row": "Benign weblog corpus",
            "col": "pretrain_records",
            "value": _count_lines(weblog_path),
            "ci_95": None,
            "seeds": [],
        })

    out = {
        "table": "table_02_dataset_profile",
        "cells": cells,
        "metadata": {
            "commit": _git_commit(),
            "date": date.today().isoformat(),
            "source": "data/splits/*.jsonl",
            "note": (
                "Backs manuscript dataset volumes and OWASP split subtype percentages; "
                "OWASP source-descriptor count is recorded separately from the parser "
                "output used for all experiments."
            ),
            "split_ranges": {
                s["name"].replace("owasp_", ""): {
                    "first_timestamp": s["first_timestamp"],
                    "last_timestamp": s["last_timestamp"],
                    "first_day": s["first_day"],
                    "last_day": s["last_day"],
                }
                for s in splits
            },
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[61] wrote {_OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
