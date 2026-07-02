from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import openpyxl

from src.utils.io import dump_json
from src.utils.paths import DATA_DIR, RESULTS_DIR, ensure_dir
from src.utils.provenance import git_commit, today_iso

XLSX = DATA_DIR / "manual_verification" / "owasp_1k_subset_human.xlsx"
MANIFEST = DATA_DIR / "manual_verification" / "sample_manifest.json"
OUT = RESULTS_DIR / "manual_verification" / "results.json"

ATTACK8 = ["sql_injection", "rce", "php_injection", "xss",
           "path_traversal", "lfi", "scanner", "protocol"]

UNDECIDABLE = {"normal", "unsure", "undecidable", ""}

def _norm(v) -> str:
    v = (str(v) if v is not None else "").strip().lower()
    if v in ("false_positive", "false positive", "fp"):
        return "benign"
    return v

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))

def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must have equal length")
    n = len(labels_a)
    if n == 0:
        return {"kappa": None, "n": 0, "observed_agreement": None}
    cats = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        conf[idx[a]][idx[b]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row_m = [sum(conf[i][j] for j in range(k)) / n for i in range(k)]
    col_m = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row_m[i] * col_m[i] for i in range(k))
    if math.isclose(1.0 - pe, 0.0):
        kappa = 1.0 if math.isclose(po, 1.0) else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {
        "kappa": round(kappa, 4),
        "n": n,
        "observed_agreement": round(100 * po, 2),
        "categories": cats,
    }

def _score_one_column(rows, col: str, weight: dict) -> dict:
    by_crs = defaultdict(lambda: [0, 0])
    benign_by_crs = defaultdict(int)
    undecidable_by_crs = defaultdict(int)
    n_undecidable = 0
    num_w = den_w = 0.0
    benign_w = 0.0
    total_w = 0.0
    undec_w = 0.0
    human_dist = Counter()
    n_total = len(rows)

    for r in rows:
        crs = str(r["crs_subtype"]).strip()
        hv = _norm(r.get(col))
        human_dist[hv] += 1
        w = float(weight.get(crs, 1.0))
        total_w += w
        if hv in UNDECIDABLE:
            n_undecidable += 1
            undecidable_by_crs[crs] += 1
            undec_w += w
            continue
        by_crs[crs][1] += 1
        den_w += w
        if hv == crs:
            by_crs[crs][0] += 1
            num_w += w
        if hv == "benign":
            benign_by_crs[crs] += 1
            benign_w += w

    per_class = {}
    for crs in ATTACK8:
        k, nc = by_crs[crs]
        p, lo, hi = wilson(k, nc)
        per_class[crs] = {
            "n_decidable": nc,
            "human_eq_crs": k,
            "undecidable": undecidable_by_crs.get(crs, 0),
            "crs_precision": round(100 * p, 2) if nc else None,
            "ci95": [round(100 * lo, 2), round(100 * hi, 2)] if nc else None,
        }

    n_decidable = n_total - n_undecidable
    acc_w = round(100 * num_w / den_w, 2) if den_w else 0.0
    benign_w_pct = round(100 * benign_w / total_w, 2) if total_w else 0.0
    undec_w_pct = round(100 * undec_w / total_w, 2) if total_w else 0.0
    raw_acc = round(100 * sum(v[0] for v in by_crs.values()) / n_decidable, 2) if n_decidable else 0.0

    return {
        "column": col,
        "n_sample": n_total,
        "n_decidable": n_decidable,
        "n_undecidable": n_undecidable,
        "human_label_distribution": dict(human_dist.most_common()),
        "crs_accuracy_weighted_pct": acc_w,
        "crs_accuracy_raw_pct": raw_acc,
        "benign_fp_rate_weighted_pct": benign_w_pct,
        "undecidable_rate_weighted_pct": undec_w_pct,
        "undecidable_rate_raw_pct": round(100 * n_undecidable / n_total, 2),
        "per_class_crs_precision": per_class,
        "benign_by_crs_stratum": dict(benign_by_crs),
        "undecidable_by_crs_stratum": dict(undecidable_by_crs),
    }

def read_xlsx(path: Path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(it)]
    rows = []
    for raw in it:
        if raw is None or all(c is None for c in raw):
            continue
        rows.append({header[i]: raw[i] for i in range(min(len(header), len(raw)))})
    return header, rows

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the human expert label audit (.xlsx).")
    ap.add_argument("--xlsx", default=str(XLSX))
    ap.add_argument("--column", default="label_human", help="Primary expert column.")
    ap.add_argument(
        "--column2",
        default=None,
        help="Optional second expert column (e.g. label_human2) for Cohen's kappa.",
    )
    args = ap.parse_args(argv)

    path = Path(args.xlsx)
    if not path.is_file():
        ap.error(f"human worksheet not found: {path}")
    header, rows = read_xlsx(path)
    if args.column not in header or "crs_subtype" not in header:
        ap.error(f"worksheet missing {args.column}/crs_subtype; columns={header}")
    if args.column2 and args.column2 not in header:
        ap.error(f"worksheet missing {args.column2}; columns={header}")

    weight = {}
    if MANIFEST.is_file():
        weight = json.loads(MANIFEST.read_text(encoding="utf-8")).get("sample_weight", {})

    primary = _score_one_column(rows, args.column, weight)

    inter_rater = None
    secondary = None
    if args.column2:
        secondary = _score_one_column(rows, args.column2, weight)
        labels_a, labels_b = [], []
        for r in rows:
            a = _norm(r.get(args.column))
            b = _norm(r.get(args.column2))
            if a in UNDECIDABLE or b in UNDECIDABLE:
                continue
            labels_a.append(a)
            labels_b.append(b)
        inter_rater = cohen_kappa(labels_a, labels_b)
        inter_rater["n_both_decidable"] = inter_rater.pop("n")
        inter_rater["column_a"] = args.column
        inter_rater["column_b"] = args.column2

    audit_type = "dual_human_expert" if args.column2 else "single_human_expert"

    result = {
        "table": "manual_verification",
        "audit_type": audit_type,
        "n_sample": primary["n_sample"],
        "n_decidable": primary["n_decidable"],
        "n_undecidable": primary["n_undecidable"],
        "human_label_distribution": primary["human_label_distribution"],
        "crs_label_audit": {
            "crs_accuracy_weighted_pct": primary["crs_accuracy_weighted_pct"],
            "crs_accuracy_raw_pct": primary["crs_accuracy_raw_pct"],
            "benign_fp_rate_weighted_pct": primary["benign_fp_rate_weighted_pct"],
            "undecidable_rate_weighted_pct": primary["undecidable_rate_weighted_pct"],
            "undecidable_rate_raw_pct": primary["undecidable_rate_raw_pct"],
            "per_class_crs_precision": primary["per_class_crs_precision"],
            "benign_by_crs_stratum": primary["benign_by_crs_stratum"],
            "undecidable_by_crs_stratum": primary["undecidable_by_crs_stratum"],
        },
        "metadata": {
            "commit": git_commit(),
            "date": today_iso(),
            "worksheet": str(path).replace("\\", "/"),
            "expert_column": args.column,
            "single_run_measured": True,
            "note": "Expert hand-labels compared to CRS subtypes. 'normal'/'unsure' = "
                    "undecidable from the anonymized log; excluded from CRS-precision "
                    "denominator. Weighted figures use sample_weight to undo rare-class "
                    "over-sampling.",
        },
    }
    if secondary:
        result["crs_label_audit_expert2"] = {
            k: secondary[k]
            for k in (
                "crs_accuracy_weighted_pct",
                "crs_accuracy_raw_pct",
                "benign_fp_rate_weighted_pct",
                "undecidable_rate_weighted_pct",
                "undecidable_rate_raw_pct",
                "per_class_crs_precision",
            )
        }
        result["inter_rater"] = inter_rater

    ensure_dir(OUT.parent)
    dump_json(OUT, result)

    print(f"[46_human_audit] {primary['n_sample']} records; decidable {primary['n_decidable']}, "
          f"undecidable {primary['n_undecidable']} ({100*primary['n_undecidable']/primary['n_sample']:.1f}%)")
    print(f"  column {args.column} label dist: {primary['human_label_distribution']}")
    print(f"  CRS accuracy (weighted, decidable): {primary['crs_accuracy_weighted_pct']:.1f}%   "
          f"(raw {primary['crs_accuracy_raw_pct']:.1f}%)")
    print(f"  benign / CRS-FP rate (weighted): {primary['benign_fp_rate_weighted_pct']:.1f}%")
    print(f"  undecidable rate (weighted): {primary['undecidable_rate_weighted_pct']:.1f}%   "
          f"(raw {primary['undecidable_rate_raw_pct']:.1f}%)")
    print("  per-class CRS precision (human==crs, decidable only):")
    for crs in ATTACK8:
        d = primary["per_class_crs_precision"][crs]
        if d["crs_precision"] is None:
            print(f"    {crs:16s} n/a (0 decidable)")
        else:
            print(f"    {crs:16s} {d['crs_precision']:6.1f}%  CI[{d['ci95'][0]:5.1f},{d['ci95'][1]:5.1f}]  "
                  f"(n={d['n_decidable']}, undec={d['undecidable']})")
    if inter_rater:
        print(f"\n  multi-rater agreement ({args.column} vs {args.column2}): "
              f"kappa={inter_rater['kappa']}, n={inter_rater['n_both_decidable']}, "
              f"agreement={inter_rater['observed_agreement']}%")
        print(f"  expert2 CRS accuracy (weighted): {secondary['crs_accuracy_weighted_pct']:.1f}%")
    print(f"\n  wrote {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
