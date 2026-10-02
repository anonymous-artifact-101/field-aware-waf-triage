

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import (
    SUBTYPES,
    git_commit,
    load_split_records,
    macro_f1,
    subtype_labels,
)
from src.detector.evaluate import fit_detector
from src.detector.fasttext_embed import load_fasttext
from src.eval.metrics import mean_ci95
from src.eval.significance import paired_bootstrap_macro_f1
from src.utils.config import load_config
from src.utils.paths import ROOT, ensure_dir
from src.utils.seeds import set_seed

DEFAULT_CONFIG = "configs/optuna/classifier_5pct.yaml"
PAPER_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_BASELINE_TEST_MACRO_F1 = 65.1

def _fasttext_path(cfg: Mapping[str, Any]) -> str:
    return str(
        cfg.get("fasttext", {}).get(
            "model_path", "models/detector/fasttext/weblog_fasttext.model"
        )
    )

def _apply_params(base_cfg: Dict[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    det = cfg.setdefault("detector", {})
    p = dict(det.get("estimator_params") or {})

    estimator = str(params["estimator"])
    det["estimator"] = estimator
    det["mode"] = "supervised"
    p["C"] = float(params["C"])
    cw = params["class_weight"]
    p["class_weight"] = None if cw == "null" else "balanced"
    p["max_iter"] = int(params["max_iter"])
    if estimator == "logreg" and "solver" in params:
        p["solver"] = str(params["solver"])

    det["estimator_params"] = p
    return cfg

def _macro_f1_pct(preds, labels) -> float:
    return 100.0 * macro_f1(preds, labels, len(SUBTYPES))

def _evaluate_split(detector, records) -> Dict[str, float]:
    labels = subtype_labels(records)
    preds = detector.predict(records)
    return {
        "macro_f1": round(_macro_f1_pct(preds, labels), 4),
        "n_records": len(records),
    }

def _default_params() -> Dict[str, Any]:
    return {
        "estimator": "linear_svc",
        "C": 1.0,
        "class_weight": "null",
        "max_iter": 2000,
    }

def _find_latest_best_params() -> Path:
    optuna_dir = ROOT / "results" / "optuna"
    candidates = sorted(optuna_dir.glob("study_*/best_params.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No Optuna studies found under {optuna_dir}. Run run_study.py first."
        )
    return candidates[-1]

def _load_best_params(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload["best_params"])

def _aggregate(values: List[float], seeds: List[int], metric: str) -> Dict[str, Any]:
    mean, (lo, hi) = mean_ci95(values)
    return {
        "metric": metric,
        "seeds": seeds,
        "per_seed": {str(s): round(v, 4) for s, v in zip(seeds, values)},
        "mean": round(mean, 4),
        "ci_95": [round(lo, 4), round(hi, 4)],
        "ci_method": "student_t",
    }

def run_ci_validation(
    *,
    config_path: str,
    best_params_path: Optional[Path],
    seeds: List[int],
    out_dir: Optional[Path],
    bootstrap_B: int = 10000,
) -> Path:
    base_cfg = dict(load_config(config_path))
    bp_path = best_params_path or _find_latest_best_params()
    optuna_params = _load_best_params(bp_path)
    default_params = _default_params()

    optuna_meta = dict(base_cfg.get("optuna", {}))
    objective_split = str(optuna_meta.get("objective_split", "owasp_val"))
    report_split = str(optuna_meta.get("report_split", "owasp_test"))
    train_split = str(base_cfg.get("data", {}).get("train_split", "owasp_train"))
    label_budget = float(base_cfg.get("data", {}).get("label_budget", 0.05))

    paper_baseline = (
        DEFAULT_BASELINE_TEST_MACRO_F1 if abs(label_budget - 0.05) < 1e-9 else None
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir or (ROOT / "results" / "optuna" / f"ci_validation_{ts}")
    ensure_dir(run_dir)

    fasttext_model = load_fasttext(_fasttext_path(base_cfg))
    train_recs = load_split_records(train_split)
    val_recs = load_split_records(objective_split)
    test_recs = load_split_records(report_split)
    test_labels = subtype_labels(test_recs)

    per_seed: List[Dict[str, Any]] = []
    default_val_f1s: List[float] = []
    default_test_f1s: List[float] = []
    optuna_val_f1s: List[float] = []
    optuna_test_f1s: List[float] = []
    default_test_preds: Optional[List[int]] = None
    optuna_test_preds: Optional[List[int]] = None

    print(
        f"[ci_validation] config={config_path}\n"
        f"                best_params={bp_path}\n"
        f"                seeds={seeds}\n"
        f"                output={run_dir}"
    )

    for seed in seeds:
        set_seed(seed)
        row: Dict[str, Any] = {"seed": seed}

        def_cfg = _apply_params(base_cfg, default_params)
        def_cfg["seed"] = seed
        def_det = fit_detector(def_cfg, fasttext_model, train_recs, seed)
        def_val = _evaluate_split(def_det, val_recs)
        def_test = _evaluate_split(def_det, test_recs)
        row["default"] = {
            "val_macro_f1": def_val["macro_f1"],
            "test_macro_f1": def_test["macro_f1"],
        }
        default_val_f1s.append(def_val["macro_f1"])
        default_test_f1s.append(def_test["macro_f1"])
        default_test_preds = def_det.predict(test_recs)

        opt_cfg = _apply_params(base_cfg, optuna_params)
        opt_cfg["seed"] = seed
        opt_det = fit_detector(opt_cfg, fasttext_model, train_recs, seed)
        opt_val = _evaluate_split(opt_det, val_recs)
        opt_test = _evaluate_split(opt_det, test_recs)
        row["optuna_best"] = {
            "val_macro_f1": opt_val["macro_f1"],
            "test_macro_f1": opt_test["macro_f1"],
        }
        optuna_val_f1s.append(opt_val["macro_f1"])
        optuna_test_f1s.append(opt_test["macro_f1"])
        optuna_test_preds = opt_det.predict(test_recs)

        row["delta_test_macro_f1"] = round(opt_test["macro_f1"] - def_test["macro_f1"], 4)
        per_seed.append(row)
        print(
            f"[ci_validation] seed={seed} | default test={def_test['macro_f1']:.4f} | "
            f"optuna test={opt_test['macro_f1']:.4f} | delta={row['delta_test_macro_f1']:+.4f}"
        )

    assert default_test_preds is not None and optuna_test_preds is not None
    bootstrap = paired_bootstrap_macro_f1(
        optuna_test_preds,
        default_test_preds,
        test_labels,
        num_classes=len(SUBTYPES),
        B=bootstrap_B,
        seed=42,
        name_a="optuna_best",
        name_b="default",
    )

    def_test_agg = _aggregate(default_test_f1s, seeds, "test_macro_f1")
    opt_test_agg = _aggregate(optuna_test_f1s, seeds, "test_macro_f1")

    try:
        bp_rel = str(bp_path.relative_to(ROOT))
    except ValueError:
        bp_rel = str(bp_path)

    summary = {
        "timestamp_utc": ts,
        "git_commit": git_commit(),
        "config_path": config_path,
        "best_params_source": bp_rel,
        "best_params": optuna_params,
        "default_params": default_params,
        "seeds": seeds,
        "splits": {
            "train": train_split,
            "val": objective_split,
            "test": report_split,
        },
        "n_records": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        },
        "aggregated": {
            "default": {
                "val_macro_f1": _aggregate(default_val_f1s, seeds, "val_macro_f1"),
                "test_macro_f1": def_test_agg,
            },
            "optuna_best": {
                "val_macro_f1": _aggregate(optuna_val_f1s, seeds, "val_macro_f1"),
                "test_macro_f1": opt_test_agg,
            },
        },
        "comparison": {
            "label_budget": label_budget,
            "paper_baseline_test_macro_f1": paper_baseline,
            "default_mean_test_macro_f1": def_test_agg["mean"],
            "optuna_mean_test_macro_f1": opt_test_agg["mean"],
            "optuna_minus_default_test_delta_mean": round(
                opt_test_agg["mean"] - def_test_agg["mean"], 4
            ),
            "optuna_minus_baseline_test_delta_mean": (
                round(opt_test_agg["mean"] - paper_baseline, 4)
                if paper_baseline is not None else None
            ),
            "optuna_ci_excludes_baseline": (
                opt_test_agg["ci_95"][0] > paper_baseline
                if paper_baseline is not None else None
            ),
            "paired_bootstrap_test": bootstrap,
            "significant_vs_default": bootstrap["significant"],
        },
    }

    per_seed_path = run_dir / "per_seed.json"
    summary_path = run_dir / "summary.json"

    with per_seed_path.open("w", encoding="utf-8") as fh:
        json.dump(per_seed, fh, indent=2, sort_keys=True)
        fh.write("\n")

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    def_agg = summary["aggregated"]["default"]["test_macro_f1"]
    opt_agg = summary["aggregated"]["optuna_best"]["test_macro_f1"]
    print(f"\n[ci_validation] wrote {per_seed_path}")
    print(f"[ci_validation] wrote {summary_path}")
    print(
        f"\n[ci_validation] DEFAULT test macro-F1: {def_agg['mean']:.1f} "
        f"[{def_agg['ci_95'][0]:.1f}, {def_agg['ci_95'][1]:.1f}]"
    )
    print(
        f"[ci_validation] OPTUNA   test macro-F1: {opt_agg['mean']:.1f} "
        f"[{opt_agg['ci_95'][0]:.1f}, {opt_agg['ci_95'][1]:.1f}]"
    )
    print(
        f"[ci_validation] label_budget={label_budget}; baseline={paper_baseline}; "
        f"paired bootstrap significant vs default: {bootstrap['significant']} "
        f"(delta={bootstrap['delta']:+.4f}, p={bootstrap['p_value']:.6f})"
    )
    return run_dir

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-seed CI validation: Optuna best vs default baseline."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Optuna base YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--best-params",
        default=None,
        help="Path to best_params.json (default: latest study under results/optuna/).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=PAPER_SEEDS,
        help="Seeds for CI aggregation (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: results/optuna/ci_validation_<timestamp>/).",
    )
    parser.add_argument(
        "--bootstrap-B",
        type=int,
        default=10000,
        help="Resamples for paired record-level bootstrap significance test.",
    )
    args = parser.parse_args(argv)

    bp = Path(args.best_params) if args.best_params else None
    if bp is not None and not bp.is_absolute():
        bp = ROOT / bp
    out = Path(args.out_dir) if args.out_dir else None
    if out is not None and not out.is_absolute():
        out = ROOT / out

    run_ci_validation(
        config_path=args.config,
        best_params_path=bp,
        seeds=list(args.seeds),
        out_dir=out,
        bootstrap_B=args.bootstrap_B,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
