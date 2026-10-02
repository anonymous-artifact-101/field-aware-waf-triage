
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit, load_split_records
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

DESCRIPTOR = 147205
OUT = RESULTS_DIR / "table_19_owasp_reconciliation" / "descriptor_gap_filters.json"

def main() -> int:
    recs = []
    for s in ("owasp_train", "owasp_val", "owasp_test"):
        recs += load_split_records(s)
    filters = {
        "all parsed records": len(recs),
        "status == 403": sum(r["status"] == 403 for r in recs),
        "status >= 400": sum(r["status"] >= 400 for r in recs),
        "status >= 400 and != 404": sum(r["status"] >= 400 and r["status"] != 404 for r in recs),
        "not fired only by custom rule 444444": sum(r["crs_rule_ids"] != ["444444"] for r in recs),
        "severity includes CRITICAL": sum("CRITICAL" in r["severities"] for r in recs),
        "unique (method, path, query, ua, timestamp)": len(
            {(r["method"], r["path"], r["query"], r["ua"], r["timestamp"]) for r in recs}),
        "unique (client_ip, timestamp, path, query)": len(
            {(r["client_ip"], r["timestamp"], r["path"], r["query"]) for r in recs}),
    }
    proto = [r for r in recs if r["attack_subtype"] == "protocol"]
    only_444 = sum(r["crs_rule_ids"] == ["444444"] for r in proto)
    out = {
        "descriptor_count": DESCRIPTOR,
        "filters": {k: {"count": v, "minus_descriptor": v - DESCRIPTOR} for k, v in filters.items()},
        "exact_match_found": any(v == DESCRIPTOR for v in filters.values()),
        "status_counts": dict(Counter(r["status"] for r in recs).most_common()),
        "protocol_class": {
            "n": len(proto),
            "fired_only_by_custom_rule_444444": only_444,
            "share_only_444444": round(only_444 / len(proto), 4),
        },
        "metadata": {"commit": git_commit(), "dataset_version": dataset_version(),
                     "note": "No tested filter reproduces the descriptor count; gap unresolved."},
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "status_counts"}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
