
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.labels import SUBTYPES
from src.utils.io import dump_json, read_jsonl
from src.utils.paths import DATA_DIR, SPLITS_DIR, ensure_dir
from src.utils.provenance import git_commit, today_iso

_CONTEXT_COLUMNS = [
    "request_id",
    "stratum",
    "sample_weight",
    "crs_subtype",
    "crs_rule_id",
    "crs_tags",
    "method",
    "path",
    "query",
    "ua",
    "status",
]

_DEFAULT_RATERS = ["human"]
_VALID_LABELS_HINT = "|".join(SUBTYPES) + "|benign|unsure"

def _truncate(value, limit: int = 300) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."

def _plan_strata(by_class: "dict[str, list[int]]", n: int, floor: int) -> "OrderedDict":
    present = [c for c in SUBTYPES if by_class.get(c)]
    counts = {c: len(by_class[c]) for c in present}

    small = [c for c in present if counts[c] <= floor]
    large = [c for c in present if counts[c] > floor]

    take = {c: counts[c] for c in small}
    budget_left = n - sum(take.values())

    if large and budget_left > 0:

        for c in large:
            take[c] = min(floor, counts[c])
        budget_left = n - sum(take.values())

        if budget_left > 0:
            weight_total = sum(counts[c] for c in large)

            raw = {c: budget_left * counts[c] / weight_total for c in large}
            alloc = {c: int(raw[c]) for c in large}
            assigned = sum(alloc.values())

            order = sorted(large, key=lambda c: (raw[c] - alloc[c], counts[c]), reverse=True)
            for c in order:
                if assigned >= budget_left:
                    break
                if take[c] + alloc[c] < counts[c]:
                    alloc[c] += 1
                    assigned += 1
            for c in large:
                take[c] = min(counts[c], take[c] + alloc[c])
    elif large and budget_left <= 0:

        for c in large:
            take[c] = 0

    return OrderedDict((c, take[c]) for c in SUBTYPES if c in take and take[c] > 0)

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stratified, multi-rater OWASP worksheet for manual label verification."
    )
    parser.add_argument("--split", default="owasp_test", help="Split JSONL under data/splits/.")
    parser.add_argument("--n", type=int, default=1000, help="Total records to sample.")
    parser.add_argument(
        "--floor",
        type=int,
        default=60,
        help="Per-class minimum sample size; classes smaller than this are censused.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (determinism).")
    parser.add_argument(
        "--raters",
        default=",".join(_DEFAULT_RATERS),
        help="Comma-separated annotator ids; one label_/notes_ column pair each.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path; defaults to data/manual_verification/owasp_1k_subset.csv.",
    )
    args = parser.parse_args(argv)

    split_path = SPLITS_DIR / f"{args.split}.jsonl"
    if not split_path.is_file():
        parser.error(f"split JSONL not found: {split_path}")

    records = list(read_jsonl(split_path))
    total = len(records)
    if total == 0:
        parser.error(f"split {split_path} is empty")

    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    if not raters:
        parser.error("--raters resolved to an empty list")

    by_class: "dict[str, list[int]]" = defaultdict(list)
    for i, rec in enumerate(records):
        by_class[rec.get("attack_subtype", "protocol")].append(i)
    corpus_counts = {c: len(by_class[c]) for c in by_class}

    n = min(int(args.n), total)
    plan = _plan_strata(by_class, n, int(args.floor))

    chosen_idx: "list[int]" = []
    sampled_counts: "dict[str, int]" = {}
    for s_idx, (cls, k) in enumerate(plan.items()):
        pool = by_class[cls]

        rng = random.Random(f"{args.seed}:{s_idx}:{cls}")
        picks = pool if k >= len(pool) else rng.sample(pool, k)
        chosen_idx.extend(picks)
        sampled_counts[cls] = len(picks)
    chosen_idx = sorted(set(chosen_idx))

    sample_weight = {
        cls: corpus_counts[cls] / sampled_counts[cls] for cls in sampled_counts
    }

    rater_columns: "list[str]" = []
    for r in raters:
        rater_columns.append(f"label_{r}")
        rater_columns.append(f"notes_{r}")
    columns = _CONTEXT_COLUMNS + rater_columns

    out_path = (
        Path(args.out)
        if args.out
        else DATA_DIR / "manual_verification" / "owasp_1k_subset.csv"
    )
    ensure_dir(out_path.parent)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i in chosen_idx:
            rec = records[i]
            cls = rec.get("attack_subtype", "protocol")
            rule_ids = rec.get("crs_rule_ids", [])
            tags = rec.get("crs_tags", [])

            attack_tags = [
                t for t in (tags if isinstance(tags, list) else [])
                if "attack" in str(t).lower() or "WEB_ATTACK" in str(t)
                or "AUTOMATION" in str(t) or "PROTOCOL_VIOLATION" in str(t)
            ]
            row = {
                "request_id": rec.get("unique_id", ""),
                "stratum": cls,
                "sample_weight": f"{sample_weight[cls]:.4f}",
                "crs_subtype": cls,
                "crs_rule_id": ";".join(str(r) for r in rule_ids)
                if isinstance(rule_ids, list)
                else str(rule_ids),
                "crs_tags": _truncate(";".join(str(t) for t in attack_tags), 200),
                "method": _truncate(rec.get("method"), 16),
                "path": _truncate(rec.get("path")),
                "query": _truncate(rec.get("query")),
                "ua": _truncate(rec.get("ua")),
                "status": rec.get("status", ""),
            }
            for r in raters:
                row[f"label_{r}"] = ""
                row[f"notes_{r}"] = ""
            writer.writerow(row)

    manifest = {
        "split": args.split,
        "split_path": str(split_path),
        "n_requested": int(args.n),
        "n_sampled": len(chosen_idx),
        "n_total": total,
        "seed": args.seed,
        "floor": int(args.floor),
        "raters": raters,
        "label_vocabulary": list(SUBTYPES) + ["benign", "unsure"],
        "sampling": "stratified by CRS attack_subtype; rare classes over-sampled to "
        "--floor (or censused if smaller), remaining budget proportional to corpus "
        "size; weighted statistics reweight to corpus rate via sample_weight.",
        "corpus_counts": corpus_counts,
        "sampled_counts": sampled_counts,
        "sample_weight": sample_weight,
        "sampled_indices": chosen_idx,
        "columns": columns,
        "commit": git_commit(),
        "date": today_iso(),
        "note": "Deterministic stratified sample without replacement; re-run with the "
        "same --seed/--split/--n/--floor to reproduce the exact subset. Each "
        "row's sample_weight = corpus_count/sampled_count for its stratum.",
    }
    manifest_path = out_path.parent / "sample_manifest.json"
    dump_json(manifest_path, manifest)

    print(
        f"[42_manual_verification] stratified sample {len(chosen_idx)}/{total} from "
        f"{args.split} (seed={args.seed}, floor={args.floor}, raters={len(raters)})"
    )
    print("[42_manual_verification] per-class (corpus -> sampled, weight):")
    for cls in SUBTYPES:
        if cls in sampled_counts:
            print(
                f"    {cls:16s} {corpus_counts[cls]:6d} -> {sampled_counts[cls]:4d}"
                f"  (w={sample_weight[cls]:.2f})"
            )
    print(f"[42_manual_verification] wrote {out_path} and {manifest_path}")
    print(f"[42_manual_verification] valid label values: {_VALID_LABELS_HINT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
