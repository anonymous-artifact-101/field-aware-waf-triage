
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.detector.classifier import FieldAwareDetector
from src.detector.fasttext_embed import train_fasttext
from src.detector.field_encoder import FieldEncoder

def _synthetic_records():
    benign = [
        {"method": "GET", "path": f"/index/{i}", "query": "", "ua": "Mozilla/5.0",
         "status": "200", "timing": "12"}
        for i in range(12)
    ]
    sqli = [
        {"method": "GET", "path": "/item", "query": f"id=1 OR 1=1--{i}",
         "ua": "sqlmap/1.0", "status": "403", "timing": "5"}
        for i in range(12)
    ]
    traversal = [
        {"method": "GET", "path": f"/../../etc/passwd{i}", "query": "",
         "ua": "curl/7.0", "status": "404", "timing": "3"}
        for i in range(12)
    ]
    records = benign + sqli + traversal
    labels = [0] * len(benign) + [1] * len(sqli) + [2] * len(traversal)
    return records, labels

def main() -> int:
    records, labels = _synthetic_records()

    model = train_fasttext(
        records,
        {"vector_size": 16, "epochs": 3, "min_count": 1, "min_n": 2, "max_n": 4,
         "workers": 1, "seed": 42},
        seed=42,
    )
    print("[smoke] FastText fit OK (vector_size=16)")

    enc = FieldEncoder(model, granularity=6)
    X = enc.encode_records(records)
    assert X.shape == (len(records), 6 * 16), f"unexpected feature shape {X.shape}"
    print(f"[smoke] FieldEncoder OK (features {X.shape[0]}x{X.shape[1]})")

    det = FieldAwareDetector(enc, mode="supervised", estimator="linear_svc", seed=42)
    det.fit(records, labels)
    preds = det.predict(records)
    assert len(preds) == len(records), "predict length mismatch"
    acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
    assert set(preds) <= {0, 1, 2}, f"unexpected class labels {set(preds)}"
    assert acc >= 0.8, f"re-substitution accuracy too low: {acc:.2f}"
    print(f"[smoke] LinearSVC fit/predict OK (re-substitution acc {acc:.2f})")

    proba = det.predict_proba(records[:3])
    assert proba.shape[0] == 3, "predict_proba row mismatch"
    attr = det.field_attribution(records[:3])
    assert len(attr) == 3, "field_attribution length mismatch"
    print("[smoke] predict_proba + field_attribution OK")

    uns = FieldAwareDetector(enc, mode="unsupervised", scorer="centroid", seed=42)
    uns.fit(records)
    scores = uns.anomaly_score(records[:5])
    assert len(scores) == 5, "anomaly_score length mismatch"
    print("[smoke] unsupervised anomaly scorer OK")

    det2 = FieldAwareDetector(enc, mode="supervised", estimator="linear_svc", seed=42)
    det2.fit(records, labels)
    assert det2.predict(records) == preds, "non-deterministic predictions across fits"
    print("[smoke] determinism OK (identical predictions across fits)")

    print("SMOKE TEST: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
