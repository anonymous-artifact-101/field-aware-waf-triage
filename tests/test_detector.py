
from __future__ import annotations

import numpy as np
import pytest

from src.baselines._common import subtype_labels, synthetic_records
from src.data.labels import SUBTYPES
from src.detector.classifier import FieldAwareDetector, build_detector
from src.detector.evaluate import _train_split_for_mode, budget_to_col, fit_detector
from src.detector.fasttext_embed import (
    record_tokens,
    tokenize_field,
    train_fasttext,
)
from src.detector.field_encoder import FieldEncoder, resolve_groups

@pytest.fixture(scope="module")
def ft_model():
    recs = synthetic_records(96, seed=11)
    return train_fasttext(
        recs,
        {"vector_size": 16, "epochs": 3, "min_count": 1, "min_n": 2, "max_n": 4},
        seed=42,
    )

@pytest.fixture(scope="module")
def records():
    return synthetic_records(96, seed=11)

def test_tokenize_field_splits_on_url_delimiters():
    toks = tokenize_field("/files?f=../../../../etc/passwd")
    assert "files" in toks and "etc" in toks and "passwd" in toks

    assert tokenize_field("") == []
    assert tokenize_field(None) == []

def test_record_tokens_covers_all_fields():
    rec = {"method": "GET", "path": "/a/b", "query": "x=1", "ua": "curl/7",
           "status": 200, "timing": 50}
    toks = record_tokens(rec)
    for expect in ("get", "a", "b", "x", "1", "curl", "7", "200", "50"):
        assert expect in toks

def test_fasttext_oov_and_determinism(records):
    m1 = train_fasttext(records, {"vector_size": 16, "epochs": 2, "min_count": 1,
                                  "min_n": 2, "max_n": 4}, seed=42)
    m2 = train_fasttext(records, {"vector_size": 16, "epochs": 2, "min_count": 1,
                                  "min_n": 2, "max_n": 4}, seed=42)
    oov = "never-seen-token-zzz"

    assert np.array_equal(m1.wv[oov], m2.wv[oov])
    assert m1.wv[oov].shape == (16,)

def test_train_fasttext_empty_corpus_raises():
    with pytest.raises(ValueError):
        train_fasttext([{"method": "", "path": "", "query": "", "ua": "",
                         "status": "", "timing": ""}], {"vector_size": 8})

@pytest.mark.parametrize("granularity,expected_blocks", [(6, 6), (3, 3), (1, 1)])
def test_encoder_feature_dims(ft_model, records, granularity, expected_blocks):
    enc = FieldEncoder(ft_model, granularity=granularity)
    assert enc.num_blocks == expected_blocks
    X = enc.encode_records(records[:10])
    assert X.shape == (10, expected_blocks * enc.vector_size)
    assert np.isfinite(X).all()

def test_encoder_empty_fields_zero_block(ft_model):
    enc = FieldEncoder(ft_model, granularity=6)
    feat = enc.encode_record({"method": "", "path": "", "query": "", "ua": "",
                              "status": "", "timing": ""})
    assert feat.shape == (6 * enc.vector_size,)
    assert np.allclose(feat, 0.0)

def test_encoder_block_slice_matches_concat(ft_model, records):
    enc = FieldEncoder(ft_model, granularity=6)
    rec = records[0]
    feat = enc.encode_record(rec)
    blocks = enc.encode_record_blocks(rec)
    for i, blk in enumerate(blocks):
        assert np.array_equal(feat[enc.block_slice(i)], blk)

def test_resolve_groups_explicit_and_validation():
    groups, labels = resolve_groups(None, groups=[["method", "path"], ["status"]])
    assert groups == [("method", "path"), ("status",)]
    assert labels == ["method_path", "status"]
    with pytest.raises(ValueError):
        resolve_groups(None, groups=[["not_a_field"]])
    with pytest.raises(ValueError):
        resolve_groups(99)

def test_max_pooling(ft_model, records):
    enc = FieldEncoder(ft_model, granularity=6, pooling="max")
    X = enc.encode_records(records[:5])
    assert X.shape[1] == 6 * enc.vector_size and np.isfinite(X).all()

@pytest.mark.parametrize("estimator", ["hist_gbdt", "logreg", "linear_svc"])
def test_supervised_fit_predict(ft_model, records, estimator):
    enc = FieldEncoder(ft_model, granularity=6)
    det = FieldAwareDetector(enc, mode="supervised", estimator=estimator, seed=42)
    det.fit(records, subtype_labels(records))
    preds = det.predict(records[:16])
    assert len(preds) == 16
    assert all(0 <= p < len(SUBTYPES) for p in preds)
    proba = det.predict_proba(records[:16])
    assert proba.shape[0] == 16

    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

def test_supervised_determinism(ft_model, records):
    enc = FieldEncoder(ft_model, granularity=6)
    a = FieldAwareDetector(enc, mode="supervised", estimator="logreg", seed=42)
    a.fit(records, subtype_labels(records))
    b = FieldAwareDetector(enc, mode="supervised", estimator="logreg", seed=42)
    b.fit(records, subtype_labels(records))
    assert a.predict(records[:20]) == b.predict(records[:20])

def test_supervised_requires_labels(ft_model, records):
    det = FieldAwareDetector(FieldEncoder(ft_model, granularity=6), mode="supervised")
    with pytest.raises(ValueError):
        det.fit(records, None)

@pytest.mark.parametrize("scorer", ["centroid", "isolation_forest"])
def test_unsupervised_anomaly_score(ft_model, records, scorer):
    enc = FieldEncoder(ft_model, granularity=6)
    det = FieldAwareDetector(enc, mode="unsupervised", scorer=scorer, seed=42)
    det.fit(records)
    scores = det.anomaly_score(records[:24])
    assert scores.shape == (24,)
    assert np.isfinite(scores).all()

def test_unsupervised_predict_raises(ft_model, records):
    det = FieldAwareDetector(FieldEncoder(ft_model, granularity=6),
                             mode="unsupervised", scorer="centroid")
    det.fit(records)
    with pytest.raises(RuntimeError):
        det.predict(records[:4])

def test_field_attribution_shape_and_labels(ft_model, records):
    enc = FieldEncoder(ft_model, granularity=6)
    det = FieldAwareDetector(enc, mode="supervised", estimator="logreg", seed=42)
    det.fit(records, subtype_labels(records))
    attr = det.field_attribution(records[:5])
    assert len(attr) == 5
    for a in attr:
        assert a["field"] in enc.block_labels
        assert set(a["deltas"]) == set(enc.block_labels)
        assert isinstance(a["reference_score"], float)

def test_build_detector_from_config(ft_model, records):
    cfg = {
        "granularity": 3,
        "encoder": {"pooling": "mean"},
        "detector": {"mode": "supervised", "estimator": "logreg"},
        "seed": 42,
    }
    det = build_detector(cfg, ft_model)
    assert det.encoder.num_blocks == 3
    det.fit(records, subtype_labels(records))
    assert len(det.predict(records[:8])) == 8

def test_build_detector_granularity_from_model_num_fields(ft_model):

    cfg = {"model": {"num_fields": 1}, "detector": {"mode": "supervised", "estimator": "logreg"}}
    det = build_detector(cfg, ft_model)
    assert det.encoder.num_blocks == 1

def test_budget_to_col():
    assert budget_to_col(0.0) == "0%"
    assert budget_to_col(0.05) == "5%"
    assert budget_to_col(0.10) == "10%"

def test_unsupervised_fits_on_benign_split_not_owasp_attack():

    sup_cfg = {"detector": {"mode": "supervised"},
               "data": {"train_split": "owasp_train",
                        "normal_train_split": "weblog_pretrain"}}
    assert _train_split_for_mode(sup_cfg) == "owasp_train"

    uns_cfg = {"detector": {"mode": "unsupervised"},
               "data": {"train_split": "owasp_train",
                        "normal_train_split": "weblog_pretrain"}}
    assert _train_split_for_mode(uns_cfg) == "weblog_pretrain"

    uns_default = {"detector": {"mode": "unsupervised"}, "data": {}}
    assert _train_split_for_mode(uns_default) == "weblog_pretrain"
    assert _train_split_for_mode(uns_default) != "owasp_train"

def test_fit_detector_respects_label_budget(ft_model, records):
    cfg = {
        "granularity": 6,
        "detector": {"mode": "supervised", "estimator": "logreg"},
        "data": {"label_budget": 0.25},
    }
    det = fit_detector(cfg, ft_model, records, seed=42)

    assert det._n_labeled == round(0.25 * len(records))
