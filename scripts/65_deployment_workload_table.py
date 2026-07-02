
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.utils.io import dump_json, load_json
from src.utils.paths import RESULTS_DIR

_TABLE = "table_18_deployment_workload"
_ROWS = {
    "Proposed": "Proposed end-to-end (10-run)",
    "TF-IDF flat + LinearSVC": "TF-IDF flat + LinearSVC (latency)",
    "TF-IDF typed + LinearSVC": "TF-IDF typed + LinearSVC (latency)",
    "Hashing char-ngram + SGD": "Hashing char-ngram + SGD (latency)",
    "Char-CNN": "Char-CNN (latency)",
    "TF-IDF + SVD + HGBDT": "TF-IDF + SVD + HGBDT (latency)",
}

def _latency_lookup(table08: Path) -> Dict[str, float]:
    doc = load_json(table08)
    out: Dict[str, float] = {}
    for cell in doc.get("cells", []):
        if cell.get("col") == "mean" and isinstance(cell.get("value"), (int, float)):
            out[str(cell.get("row"))] = float(cell["value"])
    return out

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Deployment workload CPU-core table.")
    parser.add_argument("--table08", default=str(RESULTS_DIR / "table_08_latency" / "results.json"))
    parser.add_argument("--rates", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--target-utilization", type=float, default=0.70)
    args = parser.parse_args(argv)

    lat = _latency_lookup(Path(args.table08))
    util = float(args.target_utilization)
    cells = []
    details: Dict[str, dict] = {}
    for label, latency_row in _ROWS.items():
        if latency_row not in lat:
            continue
        ms = lat[latency_row]
        cells.append({
            "row": label,
            "col": "mean_latency_ms",
            "value": round(ms, 6),
            "ci_95": None,
            "seeds": [],
        })
        details[label] = {"latency_ms": round(ms, 6), "rates": {}}
        for rate in args.rates:
            saturated = float(rate) * ms / 1000.0
            provisioned = saturated / util if util > 0 else saturated
            cells.extend([
                {
                    "row": label,
                    "col": f"{int(rate)} rps cores@100%",
                    "value": round(saturated, 4),
                    "ci_95": None,
                    "seeds": [],
                },
                {
                    "row": label,
                    "col": f"{int(rate)} rps cores@{int(round(util * 100))}%",
                    "value": round(provisioned, 4),
                    "ci_95": None,
                    "seeds": [],
                },
            ])
            details[label]["rates"][str(int(rate))] = {
                "cores_at_100pct": round(saturated, 4),
                "cores_at_target_utilization": round(provisioned, 4),
            }

    out = {
        "table": _TABLE,
        "cells": cells,
        "metadata": {
            "commit": git_commit(),
            "date": date.today().isoformat(),
            "source": str(Path(args.table08)),
            "unit": "CPU cores",
            "latency_source": "mean end-to-end ms/record from Table 8",
            "formula": "cores = traffic_rps * latency_ms / 1000 / target_utilization",
            "target_utilization": util,
            "rates_rps": [int(r) for r in args.rates],
            "details": details,
        },
    }
    out_path = RESULTS_DIR / _TABLE / "results.json"
    dump_json(out_path, out)
    print(f"[65] wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
