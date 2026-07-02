
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import (
    DEFAULT_OWASP_TEST_DAYS,
    DEFAULT_OWASP_TRAIN_DAYS,
    DEFAULT_OWASP_VAL_DAYS,
    SplitError,
    class_balance,
    load_records_by_day,
    make_time_ordered_splits,
    read_jsonl,
    split_by_day,
    split_counts,
    write_jsonl,
    write_splits,
)

def _rec(uid, day, ts, label="benign", status=200):
    return {
        "unique_id": uid,
        "day": day,
        "timestamp": ts,
        "label": label,
        "status": status,
        "method": "GET",
        "path": "/",
        "query": "",
        "ua": "",
        "timing": 1,
    }

def test_default_owasp_day_ranges():

    assert DEFAULT_OWASP_TRAIN_DAYS[0] == "27-Jul-2025"
    assert DEFAULT_OWASP_TRAIN_DAYS[-1] == "15-Aug-2025"
    assert DEFAULT_OWASP_VAL_DAYS == (
        "16-Aug-2025",
        "17-Aug-2025",
        "18-Aug-2025",
        "19-Aug-2025",
    )
    assert DEFAULT_OWASP_TEST_DAYS[0] == "20-Aug-2025"
    assert DEFAULT_OWASP_TEST_DAYS[-1] == "25-Aug-2025"
    assert len(DEFAULT_OWASP_TRAIN_DAYS) == 20
    assert len(DEFAULT_OWASP_VAL_DAYS) == 4
    assert len(DEFAULT_OWASP_TEST_DAYS) == 6

def test_split_by_day_assigns_buckets():
    records = [
        _rec("a", "27-Jul-2025", "2025-07-27T00:00:00+02:00"),
        _rec("b", "16-Aug-2025", "2025-08-16T00:00:00+02:00"),
        _rec("c", "25-Aug-2025", "2025-08-25T00:00:00+02:00"),
    ]
    splits = split_by_day(
        records,
        DEFAULT_OWASP_TRAIN_DAYS,
        DEFAULT_OWASP_VAL_DAYS,
        DEFAULT_OWASP_TEST_DAYS,
    )
    assert split_counts(splits) == {"train": 1, "val": 1, "test": 1}
    assert splits["train"][0]["unique_id"] == "a"
    assert splits["val"][0]["unique_id"] == "b"
    assert splits["test"][0]["unique_id"] == "c"

def test_within_split_sorted_by_day_then_timestamp_not_shuffled():

    records = [
        _rec("late", "10-Aug-2025", "2025-08-10T05:00:00+02:00"),
        _rec("early", "02-Aug-2025", "2025-08-02T23:59:59+02:00"),
        _rec("mid_b", "05-Aug-2025", "2025-08-05T10:00:00+02:00"),
        _rec("mid_a", "05-Aug-2025", "2025-08-05T09:00:00+02:00"),
    ]
    splits = make_time_ordered_splits(records)
    order = [r["unique_id"] for r in splits["train"]]

    assert order == ["early", "mid_a", "mid_b", "late"]

def test_timestamp_tiebreak_by_unique_id():
    records = [
        _rec("zzz", "03-Aug-2025", "2025-08-03T00:00:00+02:00"),
        _rec("aaa", "03-Aug-2025", "2025-08-03T00:00:00+02:00"),
    ]
    splits = make_time_ordered_splits(records)
    assert [r["unique_id"] for r in splits["train"]] == ["aaa", "zzz"]

def test_overlapping_day_definition_raises():
    with pytest.raises(SplitError):
        split_by_day([], ["01-Aug-2025"], ["01-Aug-2025"], ["02-Aug-2025"])

def test_record_day_not_in_any_split_raises():
    records = [_rec("x", "31-Aug-2025", "2025-08-31T00:00:00+02:00")]
    with pytest.raises(SplitError):
        make_time_ordered_splits(records)

def test_class_balance():
    records = [
        _rec("a", "01-Aug-2025", "2025-08-01T00:00:00+02:00", label="attack"),
        _rec("b", "01-Aug-2025", "2025-08-01T00:01:00+02:00", label="benign"),
        _rec("c", "01-Aug-2025", "2025-08-01T00:02:00+02:00", label="benign"),
    ]
    bal = class_balance(records)
    assert bal == {
        "attack": 1,
        "benign": 2,
        "other": 0,
        "total": 3,
        "attack_rate": pytest.approx(1 / 3),
    }

def test_write_jsonl_is_deterministic_and_lf_terminated(tmp_path: Path):
    records = [
        _rec("b", "01-Aug-2025", "2025-08-01T00:01:00+02:00"),
        _rec("a", "01-Aug-2025", "2025-08-01T00:00:00+02:00"),
    ]
    out = tmp_path / "split.jsonl"
    n = write_jsonl(out, records)
    assert n == 2
    raw = out.read_bytes()

    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")

    write_jsonl(out, records)
    assert out.read_bytes() == raw

    first_line = raw.decode("utf-8").splitlines()[0]
    assert first_line.startswith('{"client_ip"') is False
    assert first_line.index('"day"') < first_line.index('"label"')

    assert read_jsonl(out)[0]["unique_id"] == "b"

def test_write_splits_filenames_and_summary(tmp_path: Path):
    records = [
        _rec("a", "01-Aug-2025", "2025-08-01T00:00:00+02:00", label="attack"),
        _rec("b", "21-Aug-2025", "2025-08-21T00:00:00+02:00", label="benign"),
        _rec("c", "25-Aug-2025", "2025-08-25T00:00:00+02:00", label="benign"),
    ]
    splits = make_time_ordered_splits(records)
    summary = write_splits(splits, tmp_path, "owasp")
    assert (tmp_path / "owasp_train.jsonl").exists()
    assert (tmp_path / "owasp_val.jsonl").exists()
    assert (tmp_path / "owasp_test.jsonl").exists()
    assert summary["train"]["count"] == 1
    assert summary["train"]["class_balance"]["attack"] == 1

def test_load_records_by_day_two_filename_conventions(tmp_path: Path):

    write_jsonl(
        tmp_path / "01-Aug-2025.jsonl",
        [_rec("a", "01-Aug-2025", "2025-08-01T00:00:00+02:00")],
    )
    write_jsonl(
        tmp_path / "day_02.jsonl",
        [_rec("b", "02-Aug-2025", "2025-08-02T00:00:00+02:00")],
    )
    loaded = load_records_by_day(tmp_path, ["01-Aug-2025", "02-Aug-2025"])
    assert {r["unique_id"] for r in loaded} == {"a", "b"}

def test_load_records_by_day_missing_raises(tmp_path: Path):
    with pytest.raises(SplitError):
        load_records_by_day(tmp_path, ["09-Aug-2025"])
