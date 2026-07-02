
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.splits import (
    load_records_by_day,
    make_time_ordered_splits,
    write_splits,
)
from src.eval.leakage_checks import (
    check_no_overlap,
    check_status_shortcut,
    check_temporal_order,
)
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import PROCESSED_DIR, ROOT, SPLITS_DIR
from src.utils.seeds import set_seed

LOGGER = logging.getLogger("02_make_splits")

DEFAULT_CONFIG = "configs/data/owasp_modsec.yaml"

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build time-ordered OWASP train/val/test splits."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to the data YAML config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed (splits are deterministic regardless; seed "
        "is set for pipeline consistency).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON split/leakage report "
        "(default: results/leakage_checks/owasp_splits_report.json).",
    )
    args = parser.parse_args(argv)

    _setup_logging()

    cfg = load_config(args.config)
    set_seed(int(args.seed if args.seed is not None else cfg.get("seed", 42)))

    processed_dir = (
        _resolve(cfg["processed_dir"])
        if cfg.get("processed_dir")
        else PROCESSED_DIR / "owasp_parsed"
    )
    splits_dir = _resolve(cfg["splits_dir"]) if cfg.get("splits_dir") else SPLITS_DIR
    output_prefix = cfg.get("output_prefix", "owasp")

    split_cfg = cfg["split"]
    train_days = list(split_cfg["train_days"])
    val_days = list(split_cfg["val_days"])
    test_days = list(split_cfg["test_days"])

    LOGGER.info(
        "Loading parsed records from %s (train=%d days, val=%d days, test=%d days)",
        processed_dir,
        len(train_days),
        len(val_days),
        len(test_days),
    )

    all_days = [*train_days, *val_days, *test_days]
    records = load_records_by_day(processed_dir, all_days)
    LOGGER.info("Loaded %d parsed transactions across %d days.", len(records), len(all_days))

    splits = make_time_ordered_splits(
        records,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
    )

    summary = write_splits(splits, splits_dir, output_prefix)

    for name in ("train", "val", "test"):
        info = summary[name]
        bal = info["class_balance"]
        LOGGER.info(
            "%-5s -> %s | n=%d | attack=%d benign=%d other=%d | attack_rate=%.4f",
            name,
            info["path"],
            info["count"],
            bal["attack"],
            bal["benign"],
            bal["other"],
            bal["attack_rate"],
        )
        LOGGER.info("%-5s subtype_balance: %s", name, info["subtype_balance"])

    temporal = check_temporal_order(splits)
    overlap = check_no_overlap(splits)
    if temporal["passed"]:
        LOGGER.info("Temporal order check PASSED.")
    else:
        LOGGER.error("Temporal order check FAILED: %s", temporal["violations"])
    if overlap["passed"]:
        LOGGER.info("No-overlap check PASSED (no shared unique_id across splits).")
    else:
        LOGGER.error(
            "No-overlap check FAILED: overlaps=%s duplicates=%s",
            list(overlap["overlaps"].keys()),
            list(overlap["duplicates_within_split"].keys()),
        )

    status_shortcut = check_status_shortcut(records)
    _STATUS_SHORTCUT_F1_GATE = 0.70
    logreg = status_shortcut.get("logreg")
    if logreg is None:
        status_shortcut["passed"] = True
        status_shortcut["gate"] = {
            "threshold_f1": _STATUS_SHORTCUT_F1_GATE,
            "skipped": True,
            "reason": status_shortcut.get(
                "logreg_note", "single-class corpus; status shortcut not applicable"
            ),
        }
        LOGGER.info(
            "Status-shortcut check SKIPPED (single-class: n_attack=%d n_benign=%d) "
            "-- no shortcut possible by construction.",
            status_shortcut.get("n_attack", 0),
            status_shortcut.get("n_benign", 0),
        )
    else:
        status_f1 = max(float(logreg.get("f1", 0.0)), float(status_shortcut.get("rule_f1", 0.0)))
        passed = status_f1 < _STATUS_SHORTCUT_F1_GATE
        status_shortcut["passed"] = passed
        status_shortcut["gate"] = {
            "threshold_f1": _STATUS_SHORTCUT_F1_GATE,
            "status_only_f1": status_f1,
            "skipped": False,
        }
        if passed:
            LOGGER.info(
                "Status-shortcut check PASSED (status-only F1=%.4f < %.2f gate).",
                status_f1,
                _STATUS_SHORTCUT_F1_GATE,
            )
        else:
            LOGGER.error(
                "Status-shortcut check FAILED: status alone separates attack/benign "
                "at F1=%.4f (>= %.2f gate). The split leaks via the status code; "
                "fix the corpus/labeling before training.",
                status_f1,
                _STATUS_SHORTCUT_F1_GATE,
            )

    report = {
        "config": str(_resolve(args.config)),
        "processed_dir": str(processed_dir),
        "splits_dir": str(splits_dir),
        "output_prefix": output_prefix,
        "splits": summary,
        "leakage": {
            "temporal_order": temporal,
            "no_overlap": overlap,
            "status_shortcut": status_shortcut,
        },
    }
    report_path = _resolve(
        args.report or "results/leakage_checks/owasp_splits_report.json"
    )
    dump_json(report_path, report)
    LOGGER.info("Wrote split/leakage report to %s", report_path)

    if not (temporal["passed"] and overlap["passed"] and status_shortcut["passed"]):
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
