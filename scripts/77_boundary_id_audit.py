
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.data.parsers.modsec_audit as modsec
from src.baselines._common import (
    SUBTYPES,
    git_commit,
    macro_f1,
    select_label_budget,
    subtype_labels,
)
from src.baselines.tfidf_typed_linearsvc import (
    _predict,
    fit_typed_tfidf_linearsvc,
)
from src.data.splits import make_time_ordered_splits, read_jsonl
from src.detector.evaluate import fit_detector
from src.detector.fasttext_embed import load_fasttext
from src.utils.config import load_config
from src.utils.paths import PROCESSED_DIR, RESULTS_DIR
from src.utils.seeds import set_seed

_RAW = _REPO_ROOT / "data" / "raw" / "owasp_modsec_30day"
_OUT = RESULTS_DIR / "table_19_owasp_reconciliation" / "boundary_id_audit.json"
DESCRIPTOR = 147205
SEED = 42
_ANY_A = re.compile(r"^--([0-9A-Za-z]+)-A--\s*$")
_HEX_A = re.compile(r"^--([0-9a-fA-F]+)-A--\s*$")
_ALNUM_BOUNDARY = re.compile(r"^--([0-9A-Za-z]+)-([A-Z])--\s*$")

@contextmanager
def _alphanumeric_boundaries():
    original = modsec._BOUNDARY_RE
    modsec._BOUNDARY_RE = _ALNUM_BOUNDARY
    try:
        yield
    finally:
        modsec._BOUNDARY_RE = original

def _day_dirs():
    return sorted(p for p in _RAW.iterdir() if (p / "modsec_audit.anon.log").is_file())

def _raw_boundary_counts() -> dict:
    any_a = hex_a = 0
    for d in _day_dirs():
        for line in (d / "modsec_audit.anon.log").read_text(encoding="utf-8", errors="replace").split("\n"):
            if _ANY_A.match(line):
                any_a += 1
                hex_a += int(bool(_HEX_A.match(line)))
    return {"a_boundaries_alphanumeric": any_a, "a_boundaries_hexadecimal": hex_a,
            "a_boundaries_non_hexadecimal": any_a - hex_a}

def _full_archive_records() -> list:
    recs = []
    with _alphanumeric_boundaries():
        for d in _day_dirs():
            recs += list(modsec.parse_file(d / "modsec_audit.anon.log", d.name))
    return recs

def _descriptor_filters(recs) -> dict:

    filters = {
        "all raw entries": len(recs),
        "status == 403": sum(r["status"] == 403 for r in recs),
        "status >= 400": sum(r["status"] >= 400 for r in recs),
        "status >= 400 and != 404": sum(r["status"] >= 400 and r["status"] != 404 for r in recs),
        "not fired only by custom rule 444444": sum(r["crs_rule_ids"] != ["444444"] for r in recs),
        "severity includes CRITICAL": sum("CRITICAL" in r["severities"] for r in recs),
        "unique (method, path, query, ua, timestamp)": len(
            {(r["method"], r["path"], r["query"], r["ua"], r["timestamp"]) for r in recs}),
        "unique (client_ip, timestamp, path, query)": len(
            {(r["client_ip"], r["timestamp"], r["path"], r["query"]) for r in recs}),
    }
    return {
        "filters": {k: {"count": v, "minus_descriptor": v - DESCRIPTOR} for k, v in filters.items()},
        "exact_match_found": any(v == DESCRIPTOR for v in filters.values()),
    }

def _sensitivity(splits_by_corpus: dict) -> dict:
    ft = load_fasttext(load_config("configs/finetune/_base.yaml")["fasttext"]["model_path"])
    base_cfg = load_config("configs/finetune/label_10pct.yaml")
    out: dict = {"proposed": {}, "tfidf_flat_linearsvc": {}, "tfidf_typed_linearsvc": {}}
    for corpus, sp in splits_by_corpus.items():
        yv, yt = subtype_labels(sp["val"]), subtype_labels(sp["test"])
        for budget in (0.01, 0.05, 0.10, 0.20, 0.50):
            cfg = json.loads(json.dumps(base_cfg))
            cfg.setdefault("data", {})["label_budget"] = budget
            det = fit_detector(cfg, ft, sp["train"], SEED)
            out["proposed"].setdefault(str(budget), {})[corpus] = {
                "val_macro_f1": round(100 * macro_f1(det.predict(sp["val"]), yv, len(SUBTYPES)), 2),
                "test_macro_f1": round(100 * macro_f1(det.predict(sp["test"]), yt, len(SUBTYPES)), 2),
            }
        for name in ("tfidf_flat_linearsvc", "tfidf_typed_linearsvc"):
            bcfg = load_config(f"configs/baselines/{name}.yaml")
            for budget in (0.05, 0.10):
                set_seed(SEED)
                idx = select_label_budget(len(sp["train"]), budget)
                lab = [sp["train"][i] for i in idx]
                enc, clf = fit_typed_tfidf_linearsvc(
                    lab, subtype_labels(lab), bcfg["model"], bcfg["features"], SEED)
                out[name].setdefault(str(budget), {})[corpus] = {
                    "test_macro_f1": round(100 * macro_f1(_predict(enc, clf, sp["test"]), yt, len(SUBTYPES)), 2)}
    for model in out.values():
        for cell in model.values():
            for metric in ("val_macro_f1", "test_macro_f1"):
                if metric in cell["released"]:
                    delta = cell["full_archive"][metric] - cell["released"][metric]
                    cell[f"delta_{metric}"] = round(delta, 2)
                    cell[f"delta_{metric}_1dp"] = round(delta, 1)
    return out

def main() -> int:
    split_cfg = load_config("configs/data/owasp_modsec.yaml")["split"]

    def splits_of(recs):
        return make_time_ordered_splits(recs, train_days=split_cfg["train_days"],
                                        val_days=split_cfg["val_days"], test_days=split_cfg["test_days"])

    released = [r for f in sorted((PROCESSED_DIR / "owasp_parsed").glob("day_*.jsonl")) for r in read_jsonl(f)]
    full = _full_archive_records()
    released_ids = {r["unique_id"] for r in released}
    excluded = [r for r in full if r["unique_id"] not in released_ids]
    sp_released, sp_full = splits_of(released), splits_of(full)
    split_of = {r["unique_id"]: name for name, recs in sp_full.items() for r in recs}

    out = {
        "descriptor_count": DESCRIPTOR,
        "raw_boundaries": _raw_boundary_counts(),
        "raw_entries": len(full),
        "raw_unique_ids": len({r["unique_id"] for r in full}),
        "released_records": len(released),
        "excluded_non_hex_boundary": len(excluded),
        "excluded_share_pct": round(100 * len(excluded) / len(full), 2),
        "raw_minus_descriptor": len(full) - DESCRIPTOR,
        "released_minus_descriptor": len(released) - DESCRIPTOR,
        "excluded_by_split": dict(Counter(split_of[r["unique_id"]] for r in excluded)),
        "excluded_by_subtype": dict(Counter(r["attack_subtype"] for r in excluded).most_common()),
        "excluded_by_split_subtype": {
            s: dict(Counter(r["attack_subtype"] for r in excluded if split_of[r["unique_id"]] == s))
            for s in ("train", "val", "test")},
        "excluded_by_status": dict(Counter(r["status"] for r in excluded).most_common()),
        "split_sizes": {"released": {k: len(v) for k, v in sp_released.items()},
                        "full_archive": {k: len(v) for k, v in sp_full.items()}},
        "descriptor_filters_full_archive": _descriptor_filters(full),
        "sensitivity_seed42": _sensitivity({"released": sp_released, "full_archive": sp_full}),
        "metadata": {
            "commit": git_commit(),
            "note": "Excluded entries are well-formed transactions with unique IDs whose boundary "
                    "ID is not hexadecimal; the released corpus is unchanged. 'released' rows are "
                    "a control and must equal the published values.",
        },
    }
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("sensitivity_seed42",)}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
