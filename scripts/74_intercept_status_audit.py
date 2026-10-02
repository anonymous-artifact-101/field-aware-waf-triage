
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.data.parsers.modsec_audit import (
    _iter_raw_transactions,
    _split_sections,
    parse_transaction,
)
from src.utils.paths import RESULTS_DIR

_RAW = _REPO_ROOT / "data" / "raw" / "owasp_modsec_30day"
_OUT = RESULTS_DIR / "table_19_owasp_reconciliation" / "intercept_status_audit.json"

def main() -> int:
    total = 0
    intercepted = 0
    status_all: Counter = Counter()
    status_intercepted: Counter = Counter()
    status_not_intercepted: Counter = Counter()
    only_444: Counter = Counter()
    for day_dir in sorted(p for p in _RAW.iterdir() if (p / "modsec_audit.anon.log").is_file()):
        text = (day_dir / "modsec_audit.anon.log").read_text(encoding="utf-8", errors="replace")
        for raw in _iter_raw_transactions(text):
            rec = parse_transaction(raw, day_dir.name)
            if rec is None:
                continue
            h = _split_sections(raw).get("H", [])
            is_int = any(line.startswith("Action: Intercepted") for line in h)
            status = int(rec.get("status") or 0)
            is_444 = list(rec.get("crs_rule_ids") or []) == ["444444"]
            total += 1
            intercepted += int(is_int)
            status_all[status] += 1
            (status_intercepted if is_int else status_not_intercepted)[status] += 1
            only_444[f"intercepted={is_int},only_444444={is_444}"] += 1
    out = {
        "transactions": total,
        "intercepted": intercepted,
        "intercepted_share": round(intercepted / max(1, total), 4),
        "status_403": status_all.get(403, 0),
        "status_403_share": round(status_all.get(403, 0) / max(1, total), 4),
        "status_403_share_among_intercepted": round(
            status_intercepted.get(403, 0) / max(1, intercepted), 4),
        "intercepted_share_among_status_403": round(
            status_intercepted.get(403, 0) / max(1, status_all.get(403, 0)), 4),
        "top_status_intercepted": status_intercepted.most_common(8),
        "top_status_not_intercepted": status_not_intercepted.most_common(8),
        "intercept_by_custom_rule_444444": dict(only_444),
        "metadata": {"commit": git_commit(),
                     "note": "Action: Intercepted in audit section H = ModSecurity disrupted the "
                             "transaction; other records were logged by rule matches but served."},
    }
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
