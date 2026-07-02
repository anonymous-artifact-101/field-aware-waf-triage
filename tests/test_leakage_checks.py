
from __future__ import annotations

import pytest

from src.eval.leakage_checks import (
    LeakageError,
    assert_no_overlap,
    assert_temporal_order,
    check_no_overlap,
    check_status_shortcut,
    check_temporal_order,
)

def _rec(uid, ts, label="benign", status=200):
    return {"unique_id": uid, "timestamp": ts, "label": label, "status": status}

def _ordered_splits():
    return {
        "train": [
            _rec("t1", "2025-08-01T00:00:00+02:00"),
            _rec("t2", "2025-08-20T23:59:59+02:00"),
        ],
        "val": [
            _rec("v1", "2025-08-21T00:00:00+02:00"),
            _rec("v2", "2025-08-24T23:59:59+02:00"),
        ],
        "test": [
            _rec("e1", "2025-08-25T00:00:00+02:00"),
            _rec("e2", "2025-08-30T23:59:59+02:00"),
        ],
    }

def test_temporal_order_passes_when_chronologically_separated():
    report = check_temporal_order(_ordered_splits())
    assert report["passed"] is True
    assert report["violations"] == []

    assert len(report["comparisons"]) == 2

def test_temporal_order_detects_future_in_past():
    splits = _ordered_splits()

    splits["test"].append(_rec("leak", "2025-08-10T00:00:00+02:00"))
    report = check_temporal_order(splits)
    assert report["passed"] is False
    assert any(v["earlier"] == "val" and v["later"] == "test" for v in report["violations"])
    with pytest.raises(LeakageError):
        assert_temporal_order(splits)

def test_temporal_order_respects_timezone_offsets():

    splits = {
        "train": [_rec("t", "2025-08-20T23:00:00+00:00")],
        "val": [_rec("v", "2025-08-20T20:00:00-04:00")],
    }

    assert check_temporal_order(splits)["passed"] is True

    splits_bad = {
        "train": [_rec("t", "2025-08-20T23:00:00+00:00")],
        "val": [_rec("v", "2025-08-21T01:00:00+05:00")],
    }
    assert check_temporal_order(splits_bad)["passed"] is False

def test_temporal_order_skips_empty_val():
    splits = _ordered_splits()
    splits["val"] = []
    report = check_temporal_order(splits)
    assert report["passed"] is True

    assert report["comparisons"][0]["earlier"] == "train"
    assert report["comparisons"][0]["later"] == "test"

def test_no_overlap_passes_for_disjoint_ids():
    report = check_no_overlap(_ordered_splits())
    assert report["passed"] is True
    assert report["n_overlapping_ids"] == 0

def test_no_overlap_detects_cross_split_id():
    splits = _ordered_splits()
    splits["test"][0] = _rec("t1", "2025-08-25T00:00:00+02:00")
    report = check_no_overlap(splits)
    assert report["passed"] is False
    assert "t1" in report["overlaps"]["train|test"]
    with pytest.raises(LeakageError):
        assert_no_overlap(splits)

def test_no_overlap_detects_within_split_duplicate():
    splits = _ordered_splits()
    splits["train"].append(_rec("t1", "2025-08-02T00:00:00+02:00"))
    report = check_no_overlap(splits)
    assert report["passed"] is False
    assert "t1" in report["duplicates_within_split"]["train"]

def test_status_shortcut_perfectly_separable():

    records = (
        [_rec(f"b{i}", "2025-08-01T00:00:00+02:00", label="benign", status=200) for i in range(10)]
        + [_rec(f"a{i}", "2025-08-01T00:00:00+02:00", label="attack", status=403) for i in range(10)]
    )
    out = check_status_shortcut(records)
    assert out["n"] == 20
    assert out["n_attack"] == 10
    assert out["rule_f1"] == pytest.approx(1.0)
    assert out["rule_predict_attack_statuses"] == [403]
    assert out["logreg"] is not None
    assert out["logreg"]["f1"] == pytest.approx(1.0)

    assert out["per_status"]["403"]["attack_rate"] == pytest.approx(1.0)
    assert out["per_status"]["200"]["attack_rate"] == pytest.approx(0.0)

def test_status_shortcut_not_separable():

    records = (
        [_rec(f"b{i}", "2025-08-01T00:00:00+02:00", label="benign", status=200) for i in range(10)]
        + [_rec(f"a{i}", "2025-08-01T00:00:00+02:00", label="attack", status=200) for i in range(10)]
    )
    out = check_status_shortcut(records)

    assert out["rule_predict_attack_statuses"] == []
    assert out["rule_f1"] == pytest.approx(0.0)

def test_status_shortcut_single_class_skips_logreg():
    records = [
        _rec("b1", "2025-08-01T00:00:00+02:00", label="benign", status=200),
        _rec("b2", "2025-08-01T00:00:01+02:00", label="benign", status=200),
    ]
    out = check_status_shortcut(records)
    assert out["logreg"] is None
    assert "logreg_note" in out

def test_status_shortcut_handles_unknown_status_and_bad_label():
    records = [
        _rec("b1", "2025-08-01T00:00:00+02:00", label="benign", status=None),
        {"unique_id": "x", "timestamp": "2025-08-01T00:00:00+02:00", "label": "??", "status": 500},
        _rec("a1", "2025-08-01T00:00:01+02:00", label="attack", status=500),
    ]
    out = check_status_shortcut(records)

    assert out["n"] == 2

    assert "0" in out["per_status"]
