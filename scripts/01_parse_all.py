

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.parsers.modsec_audit import parse_file
from src.data.parsers import apache_combined as weblog_parser
from src.data.parsers import csic as csic_parser
from src.utils.config import load_config
from src.utils.io import write_jsonl
from src.utils.logging_setup import get_logger
from src.utils.paths import PROCESSED_DIR, RAW_DIR, ROOT, SPLITS_DIR, ensure_dir

LOG_NAME = "parse_all"
LOG_FILE = "modsec_audit.anon.log"
DEFAULT_RAW_SUBDIR = "owasp_modsec_30day"
DEFAULT_OUT_SUBDIR = "owasp_parsed"

DEFAULT_CSIC_RAW_SUBDIR = "csic2010"
DEFAULT_CSIC_OUT_SUBDIR = "csic_parsed"
CSIC_CSV_CANDIDATES = ("csic_database.csv",)

DEFAULT_WEBLOG_CONFIG = "configs/data/web_access_logs.yaml"

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _parse_day_date(name: str) -> Optional[_dt.date]:
    parts = name.split("-")
    if len(parts) != 3:
        return None
    day_s, mon_s, year_s = parts
    try:
        day = int(day_s)
        month = _MONTHS.get(mon_s.strip().lower())
        year = int(year_s)
        if month is None:
            return None
        return _dt.date(year, month, day)
    except (ValueError, TypeError):
        return None

def discover_day_dirs(raw_dir: Path) -> List[Tuple[_dt.date, Path]]:
    found: List[Tuple[_dt.date, Path]] = []
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        date = _parse_day_date(child.name)
        if date is None:
            continue
        if (child / LOG_FILE).is_file():
            found.append((date, child))
    found.sort(key=lambda item: item[0])
    return found

def _first_existing(base: Path, candidates) -> Optional[Path]:
    for name in candidates:
        path = base / name
        if path.is_file():
            return path
    return None

def parse_owasp(raw_dir: Path, out_dir: Path, log) -> int:
    if not raw_dir.is_dir():
        log.error("OWASP raw directory does not exist: %s", raw_dir)
        log.error("Run scripts/00_download_data.sh first to populate data/raw/.")
        return 1

    day_dirs = discover_day_dirs(raw_dir)
    if not day_dirs:
        log.error("No day directories with %s found under %s", LOG_FILE, raw_dir)
        return 1

    ensure_dir(out_dir)
    log.info("OWASP: found %d day directories under %s", len(day_dirs), raw_dir)

    grand_total = 0
    attack_total = 0
    for index, (date, day_dir) in enumerate(day_dirs, start=1):
        day_label = day_dir.name
        in_path = day_dir / LOG_FILE
        out_path = out_dir / f"day_{index:02d}.jsonl"

        n_records = 0
        n_attack = 0

        def _records():
            nonlocal n_records, n_attack
            for record in parse_file(in_path, day_label):
                n_records += 1
                if record["label"] == "attack":
                    n_attack += 1
                yield record

        written = write_jsonl(out_path, _records())
        grand_total += written
        attack_total += n_attack
        log.info(
            "OWASP day_%02d (%s): %d records (%d attack / %d benign) -> %s",
            index,
            day_label,
            written,
            n_attack,
            written - n_attack,
            out_path.name,
        )

    log.info(
        "OWASP done: %d days, %d records (%d attack / %d benign) -> %s",
        len(day_dirs),
        grand_total,
        attack_total,
        grand_total - attack_total,
        out_dir,
    )
    return 0

def parse_csic(raw_dir: Path, out_dir: Path, log) -> int:
    csv_path = _first_existing(raw_dir, CSIC_CSV_CANDIDATES)
    if csv_path is None:
        log.warning(
            "CSIC: no CSV found under %s (looked for %s); skipping.",
            raw_dir,
            list(CSIC_CSV_CANDIDATES),
        )
        return 0

    ensure_dir(out_dir)
    out_path = out_dir / "csic.jsonl"
    n_records = 0
    n_attack = 0

    def _records():
        nonlocal n_records, n_attack
        for record in csic_parser.parse_file(csv_path):
            n_records += 1
            if record["label"] == "attack":
                n_attack += 1
            yield record

    written = write_jsonl(out_path, _records())
    log.info(
        "CSIC done: %d records (%d attack / %d benign) from %s -> %s",
        written,
        n_attack,
        written - n_attack,
        csv_path.name,
        out_path,
    )
    return 0

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)

def parse_weblog(config_path: Path, log) -> int:
    if not config_path.is_file():
        log.warning("weblog: config %s not found; skipping.", config_path)
        return 0

    cfg = load_config(str(config_path))
    raw_path = _resolve(str(cfg.get("raw_path", "data/raw/web_access_logs/access.log")))
    split_name = str(cfg.get("pretrain_split", "weblog_pretrain"))
    out_path = SPLITS_DIR / f"{split_name}.jsonl"

    processed_dir = _resolve(str(cfg.get("processed_dir", "data/processed/weblog_parsed")))

    max_records = cfg.get("max_records")
    max_records = int(max_records) if max_records is not None else None

    if not raw_path.is_file():
        log.warning(
            "weblog: no access log found at %s; skipping "
            "(stage it via scripts/00_download_data.sh).",
            raw_path,
        )
        return 0

    ensure_dir(SPLITS_DIR)
    n_records = 0

    def _records():
        nonlocal n_records
        for record in weblog_parser.parse_file(raw_path):
            if max_records is not None and n_records >= max_records:
                break
            n_records += 1
            yield record

    written = write_jsonl(out_path, _records())

    ensure_dir(processed_dir)
    mirror_path = processed_dir / f"{split_name}.jsonl"
    m_count = 0

    def _records_mirror():
        nonlocal m_count
        for record in weblog_parser.parse_file(raw_path):
            if max_records is not None and m_count >= max_records:
                break
            m_count += 1
            yield record

    write_jsonl(mirror_path, _records_mirror())

    cap_note = f"capped at first {max_records}" if max_records is not None else "no cap (full corpus)"
    log.info(
        "weblog done: %d benign records (%s) from %s -> %s (mirror: %s)",
        written,
        cap_note,
        raw_path,
        out_path,
        mirror_path,
    )
    return 0

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse raw access-log datasets (OWASP ModSec, CSIC, Kaggle "
        "web access logs) into deterministic processed JSONL.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("owasp", "csic", "weblog", "all"),
        default=["all"],
        help="Which datasets to parse (default: all). CSIC/weblog are skipped "
        "gracefully if their raw inputs are absent.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR / DEFAULT_RAW_SUBDIR,
        help="OWASP: directory containing the per-day audit-log subdirectories "
        f"(default: {RAW_DIR / DEFAULT_RAW_SUBDIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR / DEFAULT_OUT_SUBDIR,
        help="OWASP: directory to write day_XX.jsonl files into "
        f"(default: {PROCESSED_DIR / DEFAULT_OUT_SUBDIR}).",
    )
    parser.add_argument(
        "--csic-raw-dir",
        type=Path,
        default=RAW_DIR / DEFAULT_CSIC_RAW_SUBDIR,
        help=f"CSIC: directory holding the CSV "
        f"(default: {RAW_DIR / DEFAULT_CSIC_RAW_SUBDIR}).",
    )
    parser.add_argument(
        "--csic-out-dir",
        type=Path,
        default=PROCESSED_DIR / DEFAULT_CSIC_OUT_SUBDIR,
        help=f"CSIC: output directory "
        f"(default: {PROCESSED_DIR / DEFAULT_CSIC_OUT_SUBDIR}).",
    )
    parser.add_argument(
        "--weblog-config",
        type=Path,
        default=ROOT / DEFAULT_WEBLOG_CONFIG,
        help=f"weblog: the data config YAML driving the Kaggle web-access-log "
        f"pretrain corpus (default: {DEFAULT_WEBLOG_CONFIG}).",
    )
    return parser

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    log = get_logger(LOG_NAME)

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"owasp", "csic", "weblog"}

    rc = 0
    if "owasp" in selected:
        rc |= parse_owasp(args.raw_dir, args.out_dir, log)
    if "csic" in selected:
        rc |= parse_csic(args.csic_raw_dir, args.csic_out_dir, log)
    if "weblog" in selected:
        rc |= parse_weblog(args.weblog_config, log)
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
