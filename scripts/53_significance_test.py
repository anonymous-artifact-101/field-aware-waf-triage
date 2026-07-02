
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.aggregate import load_cell_files
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR

_PROPOSED = "Proposed (FastText field-aware)"
_DET_EPS = 1e-6

def _per_seed_values(table_dir: Path, col: str) -> Dict[str, List[float]]:
    by_row: Dict[str, Dict[int, float]] = defaultdict(dict)
    for d in load_cell_files(table_dir):
        if str(d.get("col")) != col:
            continue
        by_row[str(d["row"])][int(d.get("seed", 0))] = float(d["value"])
    return {r: [by_row[r][s] for s in sorted(by_row[r])] for r in by_row}

def _compare(proposed: List[float], other: List[float]) -> Dict[str, object]:
    from scipy import stats

    p_mean, o_mean = statistics.mean(proposed), statistics.mean(other)
    p_sd = statistics.stdev(proposed) if len(proposed) > 1 else 0.0
    o_sd = statistics.stdev(other) if len(other) > 1 else 0.0
    diff = p_mean - o_mean
    p_det, o_det = p_sd < _DET_EPS, o_sd < _DET_EPS

    out: Dict[str, object] = {
        "proposed_mean": round(p_mean, 4), "proposed_sd": round(p_sd, 4),
        "other_mean": round(o_mean, 4), "other_sd": round(o_sd, 4),
        "diff": round(diff, 4),
    }
    _BOOTSTRAP_NOTE = ("seed-level summary only; the inferential proposed-vs-baseline "
                       "significance (test-set sampling uncertainty) is in "
                       "bootstrap_significance.json (scripts/55_paired_bootstrap.py)")
    if p_det and o_det:

        out["test"] = "deterministic_gap_descriptive"
        out["p_value"] = None
        out["significant"] = None
        out["note"] = ("both deterministic in seed (sd=0); exact seed gap, "
                       "descriptive only. " + _BOOTSTRAP_NOTE)
    elif o_det and not p_det:

        out["test"] = "seed_descriptive (proposed varies, baseline constant)"
        out["p_value"] = None
        out["significant"] = None
        out["note"] = _BOOTSTRAP_NOTE
    elif p_det and not o_det:

        out["test"] = "seed_descriptive (baseline varies, proposed constant)"
        out["p_value"] = None
        out["significant"] = None
        out["note"] = _BOOTSTRAP_NOTE
    else:

        t, p = stats.ttest_ind(proposed, other, equal_var=False)
        out["test"] = "welch_two_sample_t"
        out["t_stat"] = round(float(t), 4)
        out["p_value"] = round(float(p), 4)
        out["significant"] = bool(p < 0.05)
    out["direction"] = "proposed_higher" if diff > 0 else ("tie" if abs(diff) < _DET_EPS else "proposed_lower")
    return out

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Significance tests: proposed vs baselines (Table 3).")
    parser.add_argument("--col", default="5%", help="Label-budget column to compare on (default 5%).")
    parser.add_argument("--table", default="table_03_rq1_baselines", help="Results table dir name.")
    args = parser.parse_args(argv)

    table_dir = RESULTS_DIR / args.table
    by_row = _per_seed_values(table_dir, args.col)
    if _PROPOSED not in by_row:
        raise SystemExit(f"[53_significance_test] no proposed cells at col {args.col!r} in {table_dir}")

    proposed = by_row[_PROPOSED]
    results: Dict[str, object] = {}
    print(f"[53_significance_test] col={args.col}  proposed={statistics.mean(proposed):.2f} "
          f"(sd={statistics.stdev(proposed) if len(proposed) > 1 else 0:.3f})\n")
    for row in sorted(by_row):
        if row == _PROPOSED:
            continue
        verdict = _compare(proposed, by_row[row])
        results[row] = verdict
        sv = verdict["significant"]
        sig = "descriptive" if sv is None else ("SIGNIFICANT" if sv else "n.s.")
        p = verdict.get("p_value")
        ptxt = "seed-gap" if p is None else f"p={p}"
        print(f"  vs {row:30s} diff={verdict['diff']:+6.1f}  {ptxt:10s} "
              f"[{verdict['test'].split(' ')[0]}]  {sig} ({verdict['direction']})")
    print("\n[53_significance_test] seed-level summary only; inferential "
          "proposed-vs-baseline significance is in bootstrap_significance.json "
          "(run scripts/55_paired_bootstrap.py).")

    out_path = table_dir / "significance.json"
    dump_json(out_path, {"col": args.col, "proposed": _PROPOSED,
                         "comparisons": results,
                         "note": "SEED-level descriptive summary only. The inferential "
                                 "proposed-vs-baseline significance (test-set sampling "
                                 "uncertainty) is in bootstrap_significance.json "
                                 "(scripts/55_paired_bootstrap.py). "
                                 "Deterministic-in-seed methods (sd=0) yield exact seed gaps; "
                                 "p-values only where a method has seed variance."})
    print(f"\n[53_significance_test] wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
