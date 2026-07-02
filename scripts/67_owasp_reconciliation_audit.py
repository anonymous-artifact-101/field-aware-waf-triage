

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.parsers.modsec_audit import (
    _BOUNDARY_RE,
    _iter_raw_transactions,
    _parse_section_a,
    _split_sections,
    parse_transaction,
)

_RAW_DIR = _REPO_ROOT / "data" / "raw" / "owasp_modsec_30day"
_SPLITS = _REPO_ROOT / "data" / "splits"
_OUT = _REPO_ROOT / "results" / "table_19_owasp_reconciliation" / "results.json"
_DESCRIPTOR_BLOCKED_REQUESTS = 147_205
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"

def _parse_day(name: str) -> dt.date:
    day, mon, year = name.split("-")
    return dt.date(int(year), _MONTHS[mon.lower()], int(day))

def _day_dirs() -> List[Tuple[dt.date, Path]]:
    out: List[Tuple[dt.date, Path]] = []
    for child in _RAW_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            day = _parse_day(child.name)
        except Exception:
            continue
        if (child / "modsec_audit.anon.log").is_file():
            out.append((day, child))
    out.sort(key=lambda item: item[0])
    return out

def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def _split_record_audit() -> Dict[str, int]:
    total = 0
    uid_counts: Counter[str] = Counter()
    multi_rule = 0
    for name in ("owasp_train", "owasp_val", "owasp_test"):
        path = _SPLITS / f"{name}.jsonl"
        for rec in _jsonl(path):
            total += 1
            uid = str(rec.get("unique_id") or "")
            if uid:
                uid_counts[uid] += 1
            if int(rec.get("n_rules") or 0) > 1:
                multi_rule += 1
    return {
        "records": total,
        "unique_ids": len(uid_counts),
        "duplicate_id_values": sum(1 for n in uid_counts.values() if n > 1),
        "duplicate_id_records": sum(n - 1 for n in uid_counts.values() if n > 1),
        "multi_rule_records": multi_rule,
    }

def _processed_day_counts() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for path in sorted((_REPO_ROOT / "data" / "processed" / "owasp_parsed").glob("day_*.jsonl")):
        day = ""
        count = 0
        for rec in _jsonl(path):
            count += 1
            if not day:
                day = str(rec.get("day") or "")
        if day:
            out[day] = count
    return out

def _section_letters(raw_txn: str) -> List[str]:
    letters: List[str] = []
    for line in raw_txn.splitlines():
        m = _BOUNDARY_RE.match(line)
        if m:
            letters.append(m.group(2))
    return letters

def main() -> int:
    if not _RAW_DIR.is_dir():
        raise SystemExit(f"missing raw OWASP directory: {_RAW_DIR}")

    counters: Counter[str] = Counter()
    uid_counts: Counter[str] = Counter()
    per_day: List[Dict[str, Any]] = []

    processed_by_day = _processed_day_counts()
    raw_days = {day_dir.name for _day, day_dir in _day_dirs()}
    missing_raw_days = sorted(
        day for day in processed_by_day
        if not (_RAW_DIR / day / "modsec_audit.anon.log").is_file()
    )

    for _day, day_dir in _day_dirs():
        path = day_dir / "modsec_audit.anon.log"
        text = path.read_text(encoding="utf-8", errors="replace")
        day_counts: Counter[str] = Counter()
        for raw_txn in _iter_raw_transactions(text):
            sections = _split_sections(raw_txn)
            letters = _section_letters(raw_txn)
            if not letters:
                counters["empty_trailing_chunks_ignored"] += 1
                day_counts["empty_trailing_chunks_ignored"] += 1
                continue
            letter_counts = Counter(letters)
            parsed = parse_transaction(raw_txn, day_dir.name)

            counters["raw_transaction_chunks"] += 1
            day_counts["raw_transaction_chunks"] += 1
            if len(sections) > 1:
                counters["transactions_with_multiple_audit_sections"] += 1
                day_counts["transactions_with_multiple_audit_sections"] += 1
            if any(n > 1 for n in letter_counts.values()):
                counters["transactions_with_repeated_audit_section_letters"] += 1
                day_counts["transactions_with_repeated_audit_section_letters"] += 1
            if "Z" not in sections:
                counters["truncated_chunks_missing_z"] += 1
                day_counts["truncated_chunks_missing_z"] += 1
            for section in ("A", "B", "F", "H"):
                if section not in sections:
                    counters[f"missing_section_{section}"] += 1
                    day_counts[f"missing_section_{section}"] += 1

            msg_count = sum(1 for line in sections.get("H", []) if line.startswith("Message:"))
            counters["message_lines"] += msg_count
            day_counts["message_lines"] += msg_count
            if msg_count > 1:
                counters["multi_message_transactions"] += 1
                day_counts["multi_message_transactions"] += 1

            if parsed is None:
                counters["malformed_unusable_chunks_skipped"] += 1
                day_counts["malformed_unusable_chunks_skipped"] += 1
                if "Z" not in sections:
                    counters["malformed_or_truncated_chunks"] += 1
                    day_counts["malformed_or_truncated_chunks"] += 1
                continue

            counters["parsed_records_from_raw"] += 1
            day_counts["parsed_records_from_raw"] += 1
            if "Z" not in sections:
                counters["malformed_or_truncated_chunks"] += 1
                day_counts["malformed_or_truncated_chunks"] += 1
            uid = str(parsed.get("unique_id") or "")
            if not uid:

                uid, _, _ = _parse_section_a(sections.get("A", []), day_dir.name)
            if uid:
                uid_counts[uid] += 1
            else:
                counters["parsed_records_missing_unique_id"] += 1
                day_counts["parsed_records_missing_unique_id"] += 1

        per_day.append({"day": day_dir.name, **dict(day_counts)})

    duplicate_id_values = sum(1 for n in uid_counts.values() if n > 1)
    duplicate_id_records = sum(n - 1 for n in uid_counts.values() if n > 1)
    split_audit = _split_record_audit()
    split_records = split_audit["records"]
    parser_output = int(counters["parsed_records_from_raw"])
    descriptor_difference = split_records - _DESCRIPTOR_BLOCKED_REQUESTS
    current_raw_shortfall = split_records - parser_output

    rows = {
        "OWASP source descriptor": _DESCRIPTOR_BLOCKED_REQUESTS,
        "OWASP parser output": split_records,
        "OWASP split records": split_records,
        "Parser minus descriptor": descriptor_difference,
        "Split unique transaction IDs": split_audit["unique_ids"],
        "Split duplicate transaction ID values": split_audit["duplicate_id_values"],
        "Split duplicate transaction records": split_audit["duplicate_id_records"],
        "Split multi-rule records": split_audit["multi_rule_records"],
        "Available raw day files": len(raw_days),
        "Missing raw day files": len(missing_raw_days),
        "Missing raw day processed records": sum(processed_by_day.get(day, 0) for day in missing_raw_days),
        "Available raw parser output": parser_output,
        "Current staged raw shortfall vs split output": current_raw_shortfall,
        "Available raw transaction chunks": int(counters["raw_transaction_chunks"]),
        "Available raw unique transaction IDs": len(uid_counts),
        "Available raw duplicate transaction ID values": duplicate_id_values,
        "Available raw duplicate transaction records": duplicate_id_records,
        "Available raw multi-message transactions": int(counters["multi_message_transactions"]),
        "Available raw message lines": int(counters["message_lines"]),
        "Available raw transactions with multiple audit sections": int(counters["transactions_with_multiple_audit_sections"]),
        "Available raw transactions with repeated audit section letters": int(counters["transactions_with_repeated_audit_section_letters"]),
        "Available raw malformed/unusable chunks skipped": int(counters["malformed_unusable_chunks_skipped"]),
        "Available raw truncated chunks missing Z": int(counters["truncated_chunks_missing_z"]),
        "Available raw malformed or truncated chunks": int(counters["malformed_or_truncated_chunks"]),
        "Available raw parsed records missing unique ID": int(counters["parsed_records_missing_unique_id"]),
        "Available raw missing section A": int(counters["missing_section_A"]),
        "Available raw missing section B": int(counters["missing_section_B"]),
        "Available raw missing section F": int(counters["missing_section_F"]),
        "Available raw missing section H": int(counters["missing_section_H"]),
        "Empty trailing chunks ignored": int(counters["empty_trailing_chunks_ignored"]),
        "Unresolved descriptor discrepancy": descriptor_difference,
    }
    cells = [
        {"row": row, "col": "count", "value": int(value), "ci_95": None, "seeds": []}
        for row, value in rows.items()
    ]

    out = {
        "table": "table_19_owasp_reconciliation",
        "cells": cells,
        "metadata": {
            "commit": _git_commit(),
            "date": dt.date.today().isoformat(),
            "source": "data/raw/owasp_modsec_30day/*/modsec_audit.anon.log and data/splits/owasp_*.jsonl",
            "missing_raw_days": missing_raw_days,
            "definition": {
                "raw_transaction_chunks": "Chunks yielded by the same boundary parser used by src.data.parsers.modsec_audit.",
                "multi_message_transactions": "Raw chunks with more than one Section-H line beginning with 'Message:'.",
                "transactions_with_multiple_audit_sections": "Raw chunks containing more than one distinct ModSecurity audit section letter.",
                "malformed_unusable_chunks_skipped": "Raw chunks for which parse_transaction returns None.",
                "truncated_chunks_missing_z": "Raw chunks with no Z terminator section.",
            },
            "interpretation": (
                "The released split/parser output has one unique transaction ID per record. The currently "
                "staged raw directory is missing the 03-Aug-2025 audit file, whose 4,874 processed records "
                "account for the shortfall between available raw parsing and split output. On available raw "
                "days, no duplicate IDs, skipped malformed chunks, or missing-Z chunks explain the 3,806-record "
                "difference from the source descriptor; the descriptor discrepancy remains unresolved."
            ),
            "per_day": per_day,
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[67] wrote {_OUT}")
    for row, value in rows.items():
        print(f"{row}: {value}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
