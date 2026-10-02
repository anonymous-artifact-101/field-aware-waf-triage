

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import (
    SUBTYPES,
    load_split_records,
    macro_f1,
    subtype_labels,
)
from src.detector.evaluate import fit_detector
from src.detector.fasttext_embed import load_fasttext
from src.utils.config import load_config
from src.utils.paths import ROOT, ensure_dir
from src.utils.seeds import set_seed

DEFAULT_CONFIG = "configs/optuna/classifier_5pct.yaml"

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def _fasttext_path(cfg: Mapping[str, Any]) -> str:
    return str(
        cfg.get("fasttext", {}).get(
            "model_path", "models/detector/fasttext/weblog_fasttext.model"
        )
    )

def _apply_trial_params(cfg: Dict[str, Any], trial) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    det = cfg.setdefault("detector", {})
    params = dict(det.get("estimator_params") or {})

    estimator = trial.suggest_categorical("estimator", ["linear_svc", "logreg"])
    det["estimator"] = estimator
    det["mode"] = "supervised"

    params["C"] = trial.suggest_float("C", 1e-2, 1e2, log=True)
    cw = trial.suggest_categorical("class_weight", ["null", "balanced"])
    params["class_weight"] = None if cw == "null" else "balanced"
    params["max_iter"] = trial.suggest_categorical("max_iter", [500, 1000, 2000, 5000])

    if estimator == "logreg":

        params["solver"] = "lbfgs"

    det["estimator_params"] = params
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

def _write_trials_csv(path: Path, study) -> None:
    if not study.trials:
        return
    fieldnames = [
        "number",
        "state",
        "value",
        "estimator",
        "C",
        "class_weight",
        "max_iter",
        "duration_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
            row = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "duration_sec": round(t.duration.total_seconds(), 3)
                if t.duration
                else None,
            }
            row.update(t.params)
            writer.writerow(row)

def run_study(
    *,
    config_path: str,
    n_trials: int,
    seed: int,
    max_records: Optional[int],
    storage: Optional[str],
    out_dir: Optional[Path],
) -> Path:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna is not installed. Run: pip install -r requirements-optuna.txt"
        ) from exc

    base_cfg = dict(load_config(config_path))
    optuna_meta = dict(base_cfg.get("optuna", {}))
    objective_split = str(optuna_meta.get("objective_split", "owasp_val"))
    report_split = str(optuna_meta.get("report_split", "owasp_test"))
    train_split = str(base_cfg.get("data", {}).get("train_split", "owasp_train"))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_dir = out_dir or (ROOT / "results" / "optuna" / f"study_{ts}")
    ensure_dir(study_dir)

    db_path = study_dir / "study.db"
    storage_url = storage or f"sqlite:///{db_path.as_posix()}"

    set_seed(seed)
    fasttext_model = load_fasttext(_fasttext_path(base_cfg))
    train_recs = load_split_records(train_split, max_records=max_records)
    val_recs = load_split_records(objective_split, max_records=max_records)
    test_recs = load_split_records(report_split, max_records=max_records)

    print(
        f"[optuna] config={config_path} seed={seed} n_trials={n_trials}\n"
        f"         train={train_split} n={len(train_recs)} | "
        f"val={objective_split} n={len(val_recs)} | "
        f"test={report_split} n={len(test_recs)}\n"
        f"         output={study_dir}"
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=f"classifier_5pct_seed{seed}",
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=False,
    )

    def objective(trial: "optuna.Trial") -> float:
        trial_cfg = _apply_trial_params(base_cfg, trial)
        trial_cfg["seed"] = seed
        set_seed(seed)
        detector = fit_detector(trial_cfg, fasttext_model, train_recs, seed)
        metrics = _evaluate_split(detector, val_recs)
        trial.set_user_attr("val_macro_f1", metrics["macro_f1"])
        trial.set_user_attr("n_val", metrics["n_records"])
        return metrics["macro_f1"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_cfg = _apply_trial_params_from_dict(base_cfg, study.best_params)
    best_cfg["seed"] = seed
    set_seed(seed)
    best_detector = fit_detector(best_cfg, fasttext_model, train_recs, seed)
    val_metrics = _evaluate_split(best_detector, val_recs)
    test_metrics = _evaluate_split(best_detector, test_recs)

    best_params_path = study_dir / "best_params.json"
    summary_path = study_dir / "summary.json"
    trials_csv_path = study_dir / "trials.csv"

    best_payload = {
        "best_trial": study.best_trial.number,
        "best_val_macro_f1": study.best_value,
        "best_params": study.best_params,
        "test_macro_f1_with_best_params": test_metrics["macro_f1"],
        "val_macro_f1_rerun": val_metrics["macro_f1"],
        "seed": seed,
        "n_trials": n_trials,
        "config_path": config_path,
        "splits": {
            "train": train_split,
            "objective": objective_split,
            "report": report_split,
        },
        "n_records": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        },
        "search_scope": optuna_meta.get("search_scope", "classifier_only"),
        "note": "Exploratory single-seed study; paper Table 3 uses seeds {42..46} with 95% CI.",
    }
    with best_params_path.open("w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    summary = {
        **best_payload,
        "study_name": study.study_name,
        "storage": storage_url,
        "completed_trials": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "fasttext_model": _fasttext_path(base_cfg),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    _write_trials_csv(trials_csv_path, study)

    print(
        f"\n[optuna] best trial #{study.best_trial.number}: "
        f"val macro-F1={study.best_value:.4f}\n"
        f"         test macro-F1 (best params)={test_metrics['macro_f1']:.4f}\n"
        f"         params={study.best_params}\n"
        f"         wrote {best_params_path}\n"
        f"         wrote {trials_csv_path}"
    )
    return study_dir

def _apply_trial_params_from_dict(base_cfg: Dict[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
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

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optuna classifier search for the FastText field-aware detector."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Optuna base YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config optuna.seed or 42).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap records per split (smoke runs only; earliest contiguous prefix).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: results/optuna/study_<timestamp>/).",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite under the study output dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("optuna", {}).get("seed", 42))
    out_dir = _resolve(args.out_dir) if args.out_dir else None

    run_study(
        config_path=args.config,
        n_trials=args.n_trials,
        seed=seed,
        max_records=args.max_records,
        storage=args.storage,
        out_dir=out_dir,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
