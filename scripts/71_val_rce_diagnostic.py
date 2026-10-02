
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import SUBTYPES, git_commit, load_split_records, select_label_budget
from src.data.labels import load_tag_map, subtype_from_tags
from src.detector.evaluate import evaluate_detector
from src.detector.fasttext_embed import load_fasttext
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import dataset_version

OUT = RESULTS_DIR / "diagnostics" / "val_rce_10pct.json"
RCE = SUBTYPES.index("rce")

def _path_head(path: str) -> str:
    parts = [p for p in str(path).split("/") if p]
    return "/" + "/".join(parts[:2]) if parts else "/"

def main() -> int:
    splits = {s: load_split_records(f"owasp_{s}") for s in ("train", "val", "test")}
    tag_map = load_tag_map()

    mapping = {}
    for s, recs in splits.items():
        bad = [r["unique_id"] for r in recs
               if subtype_from_tags(r.get("crs_tags", []), tag_map) != r["attack_subtype"]]
        mapping[s] = {"n": len(recs), "mismatches": len(bad), "examples": bad[:5]}

    split_info = {}
    for s, recs in splits.items():
        ts = [r["timestamp"] for r in recs]
        split_info[s] = {"first": min(ts), "last": max(ts),
                         "time_ordered": all(a <= b for a, b in zip(ts, ts[1:])),
                         "days": sorted(Counter(r["day"] for r in recs))}
    ids = {s: {r["unique_id"] for r in recs} for s, recs in splits.items()}
    split_info["id_overlap"] = {f"{a}&{b}": len(ids[a] & ids[b])
                                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    split_info["boundaries_ok"] = (split_info["train"]["last"] < split_info["val"]["first"]
                                   and split_info["val"]["last"] < split_info["test"]["first"])

    ft = load_fasttext("models/detector/fasttext/weblog_fasttext.model")
    base = load_config("configs/finetune/label_10pct.yaml")
    evals = {}
    for s in ("val", "test"):
        cfg = copy.deepcopy(base)
        cfg["data"]["test_split"] = f"owasp_{s}"
        res = evaluate_detector(cfg, seed=42, fasttext_model=ft)
        y, p = np.asarray(res["test_labels"]), np.asarray(res["preds"])
        sk = f1_score(y, p, labels=list(range(len(SUBTYPES))), average=None, zero_division=0)
        evals[s] = {
            "ours_macro_f1": res["value"],
            "sklearn_macro_f1": round(100 * float(sk.mean()), 4),
            "sklearn_per_class_f1": {c: round(100 * float(v), 2) for c, v in zip(SUBTYPES, sk)},
            "labels_match_split_order": [SUBTYPES[i] for i in y[:3]] ==
                                        [r["attack_subtype"] for r in splits[s][:3]],
            "confusion_rows_true_cols_pred": confusion_matrix(
                y, p, labels=list(range(len(SUBTYPES)))).tolist(),
            "_y": y, "_p": p,
        }

    prefix = [splits["train"][i] for i in select_label_budget(len(splits["train"]), 0.10)]
    prefix_rce = [r for r in prefix if r["attack_subtype"] == "rce"]
    prefix_rules = Counter(rid for r in prefix_rce for rid in r["crs_rule_ids"])
    prefix_heads = Counter(_path_head(r["path"]) for r in prefix_rce)

    anatomy = {}
    for s in ("val", "test"):
        y, p = evals[s]["_y"], evals[s]["_p"]
        recs = splits[s]
        idx = [i for i in range(len(recs)) if y[i] == RCE]
        rules = Counter(rid for i in idx for rid in recs[i]["crs_rule_ids"])
        heads = Counter(_path_head(recs[i]["path"]) for i in idx)
        by_day = defaultdict(lambda: [0, 0])
        for i in idx:
            by_day[recs[i]["day"]][0] += 1
            by_day[recs[i]["day"]][1] += int(p[i] == RCE)
        missed = [i for i in idx if p[i] != RCE]
        rule_seen = sum(1 for i in idx if any(rid in prefix_rules for rid in recs[i]["crs_rule_ids"]
                                             if rid != "444444"))
        def _fam(i: int, prefix_: str) -> bool:
            return any(str(rid).startswith(prefix_) for rid in recs[i]["crs_rule_ids"])

        prefix_family = {"932150", "932130"}
        cofire = {
            "share_with_930xxx_lfi_rule": round(sum(_fam(i, "930") for i in idx) / max(1, len(idx)), 4),
            "missed_share_with_930xxx_lfi_rule": round(sum(_fam(i, "930") for i in missed) / max(1, len(missed)), 4),
            "share_with_prefix_family_932150_or_932130": round(
                sum(1 for i in idx if prefix_family & set(recs[i]["crs_rule_ids"])) / max(1, len(idx)), 4),
            "missed_share_with_prefix_family_932150_or_932130": round(
                sum(1 for i in missed if prefix_family & set(recs[i]["crs_rule_ids"])) / max(1, len(missed)), 4),
            "tags_of_missed": Counter(t for i in missed for t in recs[i]["crs_tags"]).most_common(8),
        }
        anatomy[s] = {
            "cofiring": cofire,
            "n_rce": len(idx),
            "recall": round(100 * (len(idx) - len(missed)) / max(1, len(idx)), 2),
            "missed_predicted_as": dict(Counter(SUBTYPES[p[i]] for i in missed).most_common()),
            "false_rce_true_class": dict(Counter(SUBTYPES[y[i]] for i in range(len(recs))
                                                 if p[i] == RCE and y[i] != RCE).most_common()),
            "top_rule_ids": rules.most_common(8),
            "top_path_heads": heads.most_common(8),
            "missed_top_path_heads": Counter(_path_head(recs[i]["path"]) for i in missed).most_common(8),
            "missed_query_empty_share": round(sum(1 for i in missed if not recs[i]["query"]) / max(1, len(missed)), 4),
            "per_day_n_and_recall": {d: [n, round(100 * k / n, 1)] for d, (n, k) in sorted(by_day.items())},
            "share_with_rule_seen_in_prefix_rce": round(rule_seen / max(1, len(idx)), 4),
            "share_with_path_head_seen_in_prefix_rce": round(
                sum(1 for i in idx if _path_head(recs[i]["path"]) in prefix_heads) / max(1, len(idx)), 4),
            "missed_examples": [{k: recs[i][k] for k in ("day", "method", "path", "query", "status",
                                                          "crs_rule_ids", "crs_tags")}
                                for i in missed[:6]],
        }
    for s in evals:
        evals[s].pop("_y"); evals[s].pop("_p")

    out = {
        "calculation": evals, "label_mapping": mapping, "split": split_info,
        "prefix_rce": {"n": len(prefix_rce), "top_rule_ids": prefix_rules.most_common(8),
                       "top_path_heads": prefix_heads.most_common(8)},
        "rce_anatomy": anatomy,
        "metadata": {"commit": git_commit(), "dataset_version": dataset_version(),
                     "label_budget": 0.10, "seed": 42},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("label_mapping", "split")}, indent=1, default=str))
    for s in ("val", "test"):
        e, a = evals[s], anatomy[s]
        print(f"== {s}: ours={e['ours_macro_f1']} sklearn={e['sklearn_macro_f1']} "
              f"rceF1={e['sklearn_per_class_f1']['rce']} n_rce={a['n_rce']} recall={a['recall']}")
        for k in ("missed_predicted_as", "false_rce_true_class", "top_rule_ids", "top_path_heads",
                  "missed_top_path_heads", "missed_query_empty_share", "per_day_n_and_recall",
                  "share_with_rule_seen_in_prefix_rce", "share_with_path_head_seen_in_prefix_rce"):
            print(f"   {k}: {a[k]}")
    print("prefix_rce:", out["prefix_rce"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
