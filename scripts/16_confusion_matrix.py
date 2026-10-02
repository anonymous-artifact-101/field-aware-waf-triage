
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CLASSES = [
    "sql_injection", "rce", "php_injection", "xss",
    "path_traversal", "lfi", "scanner", "protocol",
]

REPO = Path(__file__).resolve().parents[1]
PRED_DIR = REPO / "results" / "table_03_rq1_baselines" / "predictions"
LABELS_FILE = PRED_DIR / "labels__owasp_test.json"
OUT_DIR = REPO / "results" / "table_06_per_class_owasp"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, default=5, choices=(5, 10))
    budget = parser.parse_args().budget
    PRED_FILE = PRED_DIR / f"proposed-fasttext-field-aware__{budget}__seed42.json"
    OUT_FILE = OUT_DIR / ("confusion_matrix.json" if budget == 5 else f"confusion_matrix_{budget}pct.json")
    pred_doc = json.loads(PRED_FILE.read_text())
    lab_doc = json.loads(LABELS_FILE.read_text())
    preds = pred_doc["preds"]
    labels = lab_doc["labels"]

    if len(preds) != len(labels):
        raise SystemExit(f"length mismatch: {len(preds)} preds vs {len(labels)} labels")

    if pred_doc.get("labels_sha256") != lab_doc.get("labels_sha256"):
        raise SystemExit("labels_sha256 mismatch between prediction and label sidecars")

    counts = Counter(zip(labels, preds))

    matrix = [[int(counts.get((t, p), 0)) for p in range(len(CLASSES))]
              for t in range(len(CLASSES))]
    support = [sum(row) for row in matrix]

    out = {
        "classes": CLASSES,
        "orientation": "rows=true, cols=predicted",
        "matrix": matrix,
        "support": support,
        "n": int(pred_doc["n"]),
        "metadata": {
            "source_predictions": str(PRED_FILE.relative_to(REPO)).replace("\\", "/"),
            "labels_sha256": pred_doc.get("labels_sha256"),
            "budget": pred_doc.get("col"),
            "seed": pred_doc.get("seed"),
            "macro_f1": pred_doc.get("macro_f1"),
            "commit": pred_doc.get("metadata", {}).get("commit"),
            "date": "2026-06-03" if budget == 5 else "2026-09-24",
            "note": "Derived from released prediction sidecars; no model re-run. "
                    f"Seed-deterministic proposed detector at the {budget}% budget.",
        },
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[16_confusion_matrix] wrote {OUT_FILE.relative_to(REPO)} "
          f"({sum(support)} records over {len(CLASSES)} classes)")

if __name__ == "__main__":
    main()
