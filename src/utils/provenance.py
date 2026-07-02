
from __future__ import annotations

import datetime as _dt
import subprocess

from src.utils.paths import ROOT

__all__ = ["git_commit", "today_iso"]

def git_commit(default: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or default
    except Exception:
        return default

def today_iso() -> str:
    return _dt.date.today().isoformat()
