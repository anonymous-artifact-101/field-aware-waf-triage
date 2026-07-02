
from __future__ import annotations

import importlib.metadata as im
import os
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import dump_json

_OUT = _REPO_ROOT / "results" / "environment" / "results.json"
_PACKAGES = (
    ("gensim", "gensim"),
    ("scikit-learn", "scikit-learn"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PyYAML", "PyYAML"),
    ("torch", "torch"),
)

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

def _major_minor(version: str) -> float:
    nums = re.findall(r"\d+", version)
    if not nums:
        return float("nan")
    if len(nums) == 1:
        return float(nums[0])
    return float(f"{int(nums[0])}.{int(nums[1])}")

def main() -> int:
    cells: List[Dict[str, Any]] = []
    py_version = platform.python_version()
    cells.append({
        "row": "Python",
        "col": "major_minor",
        "value": _major_minor(py_version),
        "ci_95": None,
        "seeds": [],
    })
    package_versions: Dict[str, str] = {}
    for label, dist in _PACKAGES:
        try:
            version = im.version(dist)
        except im.PackageNotFoundError:
            version = "not-installed"
        package_versions[label] = version
        if version != "not-installed":
            cells.append({
                "row": label,
                "col": "major_minor",
                "value": _major_minor(version),
                "ci_95": None,
                "seeds": [],
            })

    cpu = os.environ.get("PECTI_HARDWARE") or platform.processor() or platform.machine()
    out = {
        "table": "environment",
        "cells": cells,
        "metadata": {
            "commit": _git_commit(),
            "date": date.today().isoformat(),
            "hardware": {
                "cpu": cpu,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "logical_cores": os.cpu_count(),
                "thread_caps": {
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                },
            },
            "software": {
                "python": py_version,
                "packages": package_versions,
            },
            "note": (
                "Backs manuscript hardware/software-environment statements. "
                "Version cells store major.minor numeric values so the number "
                "auditor can trace version numbers in the paper."
            ),
        },
    }
    dump_json(_OUT, out)
    print(f"[64] wrote {_OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
