
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.detector.evaluate import fit_detector, load_detector_inputs
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version, embedding_version
from src.utils.seeds import set_seed

_SEED = 42
_CONFIG = "configs/ablation/six_field.yaml"

_EXAMPLES = [("lfi", "path"), ("scanner", "ua")]

def _round_deltas(deltas):
    return {k: round(float(v), 3) for k, v in deltas.items()}

def main() -> int:
    cfg = dict(load_config(_CONFIG))
    set_seed(_SEED)

    ft_model, train_recs, test_recs = load_detector_inputs(cfg)
    detector = fit_detector(cfg, ft_model, train_recs, _SEED)

    examples = []
    for subtype, true_field in _EXAMPLES:
        chosen = None
        for rec in test_recs:
            if rec.get("attack_subtype") != subtype:
                continue
            attr = detector.field_attribution([rec])[0]

            if attr["field"] == true_field:
                chosen = (rec, attr)
                break
        if chosen is None:
            raise SystemExit(f"[47] no correctly-attributed {subtype} record found in test split")
        rec, attr = chosen
        examples.append({
            "subtype": subtype,
            "true_field": true_field,
            "argmax_field": attr["field"],
            "unique_id": rec.get("unique_id"),
            "path": rec.get("path"),
            "ua": rec.get("ua"),
            "deltas": _round_deltas(attr["deltas"]),
            "reference_score": round(float(attr["reference_score"]), 6),
        })

    payload = {
        "figure": "fig:attr",
        "note": ("Per-field block-zeroing deltas for two example requests plotted in "
                 "Fig. attribution; regenerated from the fitted 6-field detector "
                 "(5% budget, seed 42). The argmax field matches the attack's true field."),
        "config": _CONFIG,
        "seed": _SEED,
        "examples": examples,
        "metadata": {
            "commit": git_commit(),
            "dataset_version": dataset_version(),
            "embedding_version": embedding_version(),
            "gpu": "cpu",
            "single_run_measured": True,
        },
    }

    out_path = RESULTS_DIR / "table_07_attribution" / "figure_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[47] wrote {out_path}")
    for ex in examples:
        print(f"  {ex['subtype']:8s} argmax={ex['argmax_field']:7s} deltas={ex['deltas']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
