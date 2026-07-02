
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl as _read_jsonl
from src.utils.io import write_jsonl as _write_jsonl

__all__ = [
    "SplitError",
    "DEFAULT_OWASP_TRAIN_DAYS",
    "DEFAULT_OWASP_VAL_DAYS",
    "DEFAULT_OWASP_TEST_DAYS",
    "split_by_day",
    "make_time_ordered_splits",
    "make_csic_splits",
    "iter_jsonl",
    "read_jsonl",
    "write_jsonl",
    "load_records_by_day",
    "write_splits",
    "split_counts",
    "class_balance",
    "subtype_balance",
]

Record = Mapping[str, Any]

_ALL_OWASP_DAYS: tuple[str, ...] = (
    "27-Jul-2025", "28-Jul-2025", "29-Jul-2025", "30-Jul-2025", "31-Jul-2025",
    "01-Aug-2025", "02-Aug-2025", "03-Aug-2025", "04-Aug-2025", "05-Aug-2025",
    "06-Aug-2025", "07-Aug-2025", "08-Aug-2025", "09-Aug-2025", "10-Aug-2025",
    "11-Aug-2025", "12-Aug-2025", "13-Aug-2025", "14-Aug-2025", "15-Aug-2025",
    "16-Aug-2025", "17-Aug-2025", "18-Aug-2025", "19-Aug-2025", "20-Aug-2025",
    "21-Aug-2025", "22-Aug-2025", "23-Aug-2025", "24-Aug-2025", "25-Aug-2025",
)

DEFAULT_OWASP_TRAIN_DAYS: tuple[str, ...] = _ALL_OWASP_DAYS[:20]
DEFAULT_OWASP_VAL_DAYS: tuple[str, ...] = _ALL_OWASP_DAYS[20:24]
DEFAULT_OWASP_TEST_DAYS: tuple[str, ...] = _ALL_OWASP_DAYS[24:]

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

class SplitError(ValueError):
    """Raised when a split definition is inconsistent or a record is malformed."""

def _build_day_index(ordered_days: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, day in enumerate(ordered_days):
        if day in index:
            raise SplitError(f"Duplicate day in split definition: {day!r}")
        index[day] = position
    return index

def _sort_key(record: Record, day_index: Mapping[str, int]) -> tuple[int, str, str]:
    day = record.get("day")
    if day not in day_index:
        raise SplitError(
            f"Record day {day!r} is not in the split definition "
            f"(unique_id={record.get('unique_id')!r})."
        )

    timestamp = record.get("timestamp")
    unique_id = record.get("unique_id")
    return (
        day_index[day],
        "" if timestamp is None else str(timestamp),
        "" if unique_id is None else str(unique_id),
    )

def split_by_day(
    records: Iterable[Record],
    train_days: Sequence[str],
    val_days: Sequence[str],
    test_days: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    buckets = {"train": list(train_days), "val": list(val_days), "test": list(test_days)}

    day_to_split: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for day in buckets[split_name]:
            if day in day_to_split:
                raise SplitError(
                    f"Day {day!r} assigned to both {day_to_split[day]!r} and "
                    f"{split_name!r}; splits must be disjoint."
                )
            day_to_split[day] = split_name

    ordered_days = [*train_days, *val_days, *test_days]
    day_index = _build_day_index(ordered_days)

    out: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for record in records:
        day = record.get("day")
        if day not in day_to_split:
            raise SplitError(
                f"Record day {day!r} is not assigned to any split "
                f"(unique_id={record.get('unique_id')!r})."
            )
        out[day_to_split[day]].append(dict(record))

    for name in SPLIT_NAMES:
        out[name].sort(key=lambda r: _sort_key(r, day_index))

    return out

def make_time_ordered_splits(
    records: Iterable[Record],
    *,
    train_days: Sequence[str] = DEFAULT_OWASP_TRAIN_DAYS,
    val_days: Sequence[str] = DEFAULT_OWASP_VAL_DAYS,
    test_days: Sequence[str] = DEFAULT_OWASP_TEST_DAYS,
) -> dict[str, list[dict[str, Any]]]:
    return split_by_day(records, train_days, val_days, test_days)

_CSIC_VAL_FRACTION = 0.20

def make_csic_splits(
    records: Iterable[Record],
    *,
    val_fraction: float = _CSIC_VAL_FRACTION,
) -> dict[str, list[dict[str, Any]]]:
    if not (0.0 <= val_fraction <= 1.0):
        raise SplitError(f"val_fraction must be in [0, 1]; got {val_fraction!r}")

    benign: list[dict[str, Any]] = []
    anomalous: list[dict[str, Any]] = []
    for record in records:
        label = record.get("label")
        if label == "benign":
            benign.append(dict(record))
        elif label == "attack":
            anomalous.append(dict(record))
        else:
            raise SplitError(
                f"CSIC record has unexpected label {label!r} "
                f"(unique_id={record.get('unique_id')!r}); expected "
                "'benign' or 'attack'."
            )

    n_benign = len(benign)
    half = n_benign // 2
    benign_train = benign[:half]
    benign_test_pool = benign[half:]

    n_val = int(round(len(benign_test_pool) * val_fraction))
    benign_val = benign_test_pool[:n_val]
    benign_test = benign_test_pool[n_val:]

    return {
        "train": benign_train,
        "val": benign_val,
        "test": benign_test + anomalous,
    }

def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for line_no, obj in enumerate(_read_jsonl(path), start=1):
        if not isinstance(obj, dict):
            raise SplitError(
                f"{path}:{line_no}: expected a JSON object, got "
                f"{type(obj).__name__}."
            )
        yield obj

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))

def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    return _write_jsonl(path, records)

def load_records_by_day(
    processed_dir: str | Path,
    days: Sequence[str],
) -> list[dict[str, Any]]:
    processed_dir = Path(processed_dir)
    requested = set(days)

    by_day: dict[str, list[dict[str, Any]]] = {d: [] for d in requested}
    found_files = sorted(processed_dir.glob("*.jsonl"))
    if not found_files:
        raise SplitError(
            f"No processed JSONL files found under {processed_dir}. "
            "Run scripts/01_parse_all.py first."
        )
    for path in found_files:
        for record in iter_jsonl(path):
            day = record.get("day")
            if day in by_day:
                by_day[day].append(record)

    missing = sorted(d for d in requested if not by_day[d])
    if missing:
        raise SplitError(
            f"No processed records found for day(s) {missing}. "
            f"Searched {len(found_files)} file(s) under {processed_dir}; "
            "check the day strings in the split config match the parsed 'day' "
            "field exactly."
        )

    records: list[dict[str, Any]] = []
    for day in days:
        records.extend(by_day[day])
    return records

def write_splits(
    splits: Mapping[str, Iterable[Record]],
    splits_dir: str | Path,
    output_prefix: str,
) -> dict[str, dict[str, Any]]:
    splits_dir = Path(splits_dir)
    summary: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        if name not in splits:
            continue
        records = list(splits[name])
        out_path = splits_dir / f"{output_prefix}_{name}.jsonl"
        count = write_jsonl(out_path, records)
        summary[name] = {
            "path": str(out_path),
            "count": count,
            "class_balance": class_balance(records),
            "subtype_balance": subtype_balance(records),
        }
    return summary

def split_counts(splits: Mapping[str, Sequence[Record]]) -> dict[str, int]:
    return {name: len(splits[name]) for name in SPLIT_NAMES if name in splits}

def class_balance(records: Sequence[Record]) -> dict[str, Any]:
    attack = benign = other = 0
    for record in records:
        label = record.get("label")
        if label == "attack":
            attack += 1
        elif label == "benign":
            benign += 1
        else:
            other += 1
    total = attack + benign + other
    attack_rate = (attack / total) if total else 0.0
    return {
        "attack": attack,
        "benign": benign,
        "other": other,
        "total": total,
        "attack_rate": attack_rate,
    }

def subtype_balance(records: Sequence[Record]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        subtype = record.get("attack_subtype", "unknown")
        counts[subtype] = counts.get(subtype, 0) + 1

    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
