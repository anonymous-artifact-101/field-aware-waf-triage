
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import dump_json, load_json
from src.utils.paths import RESULTS_DIR, ensure_dir

_FIG_DIR = "figure_03_latency_accuracy_pareto"
_ACC_TABLE = "table_03_rq1_baselines"
_LAT_TABLE = "table_08_latency"
_PROPOSED = "Proposed (FastText field-aware)"

_CRS_ASSISTED = {"ModSec-Learn", "ModSec-AdvLearn"}
_SHORTCUT_DIAGNOSTICS = {"Status-only (shortcut)"}

def _accuracy_points(acc_col: str) -> Dict[str, float]:
    doc = load_json(RESULTS_DIR / _ACC_TABLE / "results.json")
    out: Dict[str, float] = {}
    for c in doc.get("cells", []):
        if str(c.get("col")) == acc_col and c.get("value") is not None:
            out[str(c["row"])] = float(c["value"])
    return out

def _proposed_latency_footprint() -> Dict[str, Optional[float]]:
    lat = foot = None
    path = RESULTS_DIR / _LAT_TABLE / "results.json"
    if path.is_file():
        doc = load_json(path)
        for c in doc.get("cells", []):

            if c.get("row") == "Proposed end-to-end (10-run)" and c.get("col") == "mean":
                lat = float(c["value"])
            elif lat is None and c.get("row") == "end_to_end" and c.get("col") == "mean":
                lat = float(c["value"])
            if c.get("row") == "Footprint (MB)" and c.get("col") == "Total":
                foot = float(c["value"])
    return {"latency_ms": lat, "footprint_mb": foot}

def _baseline_latency_footprint() -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    path = RESULTS_DIR / _LAT_TABLE / "results.json"
    if not path.is_file():
        return out
    doc = load_json(path)
    for c in doc.get("cells", []):
        row = str(c.get("row", ""))
        if row.endswith(" (latency)") and c.get("col") == "mean":
            out.setdefault(row[: -len(" (latency)")], {})["latency_ms"] = float(c["value"])
        if row == "Footprint (MB)" and c.get("col") not in ("Total", "FastText model", "Classifier head"):
            out.setdefault(str(c["col"]), {})["footprint_mb"] = float(c["value"])
    return out

def _pareto_front(points: List[Dict[str, Any]]) -> List[str]:
    usable = [p for p in points
              if p["x"] is not None
              and p["y"] is not None
              and p["name"] not in _CRS_ASSISTED
              and p["name"] not in _SHORTCUT_DIAGNOSTICS]
    front: List[str] = []
    for p in usable:
        dominated = any(
            (q["y"] >= p["y"] and q["x"] <= p["x"]) and (q["y"] > p["y"] or q["x"] < p["x"])
            for q in usable if q is not p
        )
        if not dominated:
            front.append(p["name"])
    return front

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Latency-vs-accuracy Pareto figure from results/.")
    parser.add_argument("--acc-col", default="10%", help="Table 3 budget column for accuracy (default 10%%).")
    parser.add_argument("--out-dir", default=None, help="Override results figure dir.")
    args = parser.parse_args(argv)

    acc = _accuracy_points(args.acc_col)
    if not acc:
        raise SystemExit(f"[52_make_pareto_figure] no Table 3 accuracy at col {args.acc_col!r}; "
                         f"run the table-03 pipeline first.")
    prop = _proposed_latency_footprint()
    base_lat = _baseline_latency_footprint()

    points: List[Dict[str, Any]] = []
    for row, a in sorted(acc.items(), key=lambda kv: -kv[1]):
        is_prop = row == _PROPOSED
        if is_prop:
            x, fp = prop["latency_ms"], prop["footprint_mb"]
        else:
            b = base_lat.get(row, {})
            x, fp = b.get("latency_ms"), b.get("footprint_mb")
        points.append({
            "name": row,
            "x": x,
            "y": a,
            "footprint_mb": fp,
            "latency_measured": x is not None,
            "group": "crs_assisted" if row in _CRS_ASSISTED else "log_only",
        })

    front = _pareto_front(points)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (RESULTS_DIR / _FIG_DIR))
    fig_data = {
        "figure": _FIG_DIR,
        "x_col": "latency_ms (end_to_end mean, CPU)",
        "y_col": f"macro_f1 (Table 3, {args.acc_col})",
        "points": points,
        "pareto_front": front,
        "note": ("Latency is read from Table 8 for the proposed detector and any "
                 "per-record baselines timed by scripts/54_measure_baseline_latency.py; "
                 "baselines without measured per-record latency carry latency_ms=null "
                 "and are excluded from the front. ModSec-Learn/AdvLearn are "
                 "label-adjacent CRS-input references, not fair log-only peers; their "
                 "latency measures CRS firing-vector featurization plus the learned "
                 "predictor and excludes ModSecurity/CRS rule-engine execution. "
                 "Status-only is retained as a shortcut diagnostic and excluded from "
                 "the front."),
        "source": {"accuracy": f"results/{_ACC_TABLE}/results.json",
                   "latency": f"results/{_LAT_TABLE}/results.json"},
    }
    data_path = out_dir / "figure_data.json"
    dump_json(data_path, fig_data)
    print(f"[52_make_pareto_figure] wrote {data_path} "
          f"({len(points)} points, proposed latency={prop['latency_ms']} ms, "
          f"footprint={prop['footprint_mb']} MB)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for p in points:
            if p["x"] is None:
                continue
            size = 60 + 4 * (p["footprint_mb"] or 0)
            ax.scatter(p["x"], p["y"], s=size, label=p["name"])
            ax.annotate(p["name"], (p["x"], p["y"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")

        no_lat = [p["y"] for p in points if p["x"] is None]
        if no_lat:
            ax.axhspan(min(no_lat), max(no_lat), alpha=0.08, color="gray",
                       label="baselines (accuracy only)")
        ax.set_xlabel("CPU end-to-end latency (ms/record, lower better)")
        ax.set_ylabel(f"macro-F1 (Table 3, {args.acc_col})")
        ax.set_title("Accuracy vs. CPU latency (Pareto)")
        ax.legend(fontsize=6, loc="lower right")
        png_path = out_dir / "pareto.png"
        fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)
        print(f"[52_make_pareto_figure] wrote {png_path}")
    except ImportError:
        print("[52_make_pareto_figure] matplotlib not installed; wrote figure_data.json only "
              "(install matplotlib to render the PNG).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
