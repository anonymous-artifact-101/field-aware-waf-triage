
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.utils.paths import RESULTS_DIR

_DIR = RESULTS_DIR / "table_11_benign_fpr_apache_indo"
_IN = _DIR / "results.json"
_OUT = _DIR / "wilson_bounds.json"

_Z_TWO_SIDED_95 = 1.959963984540054
_Z_ONE_SIDED_95 = 1.6448536269514722

def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def main() -> int:
    src = json.loads(_IN.read_text(encoding="utf-8"))
    n = int(src["metadata"]["n_benign"])
    rows = []
    for cell in sorted(src["cells"], key=lambda c: c["col"]):
        fpr = float(cell["value"])
        k = int(round(fpr * n))
        _, up2 = wilson(k, n, _Z_TWO_SIDED_95)
        _, up1 = wilson(k, n, _Z_ONE_SIDED_95)
        rows.append({
            "threshold": cell["col"],
            "false_positives": k,

            "count_exact": k == 0,
            "n_benign": n,
            "observed_fpr_pct": round(100.0 * k / n, 4),
            "wilson95_two_sided_upper_pct": round(100.0 * up2, 4),
            "wilson95_one_sided_upper_pct": round(100.0 * up1, 4),
        })
    out = {
        "table": "table_11_benign_fpr_apache_indo",
        "rows": rows,
        "metadata": {
            "commit": git_commit(),
            "source": str(_IN.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "method": "Wilson score interval; false positives = round(observed FPR x n_benign)",
            "z_two_sided_95": _Z_TWO_SIDED_95,
            "z_one_sided_95": _Z_ONE_SIDED_95,
        },
    }
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[75] wrote {_OUT}")
    for r in rows:
        print(r)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
