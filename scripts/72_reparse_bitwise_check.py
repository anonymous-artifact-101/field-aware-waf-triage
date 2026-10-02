
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.utils.paths import RESULTS_DIR

_PROCESSED = _REPO_ROOT / "data" / "processed" / "owasp_parsed"
_RAW = _REPO_ROOT / "data" / "raw" / "owasp_modsec_30day"
_OUT = RESULTS_DIR / "table_19_owasp_reconciliation" / "reparse_bitwise_check.json"

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    raw_days = sorted(p.name for p in _RAW.iterdir() if (p / "modsec_audit.anon.log").is_file())
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(_REPO_ROOT / "scripts" / "01_parse_all.py"),
                        "--datasets", "owasp", "--raw-dir", str(_RAW), "--out-dir", tmp],
                       check=True, cwd=_REPO_ROOT)
        files = {}
        for f in sorted(_PROCESSED.glob("day_*.jsonl")):
            other = Path(tmp) / f.name
            a, b = _sha(f), (_sha(other) if other.is_file() else None)
            files[f.name] = {"processed_sha256": a, "reparsed_sha256": b, "identical": a == b,
                             "records": sum(1 for _ in f.open(encoding="utf-8"))}
    out = {
        "raw_days_staged": len(raw_days),
        "processed_day_files": len(files),
        "identical_day_files": sum(v["identical"] for v in files.values()),
        "total_records": sum(v["records"] for v in files.values()),
        "all_identical": all(v["identical"] for v in files.values()),
        "files": files,
        "metadata": {"commit": git_commit(),
                     "note": "Independent re-parse of all staged raw days vs processed JSONL (SHA-256)."},
    }
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({k: out[k] for k in ("raw_days_staged", "processed_day_files", "identical_day_files",
                               "total_records", "all_identical")})
    return 0 if out["all_identical"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
