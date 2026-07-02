
from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import (
    SUBTYPES,
    git_commit,
    macro_f1,
    select_label_budget,
    subtype_labels,
    weighted_f1,
)
from src.data.labels import load_tag_map, subtype_from_tags
from src.data.parsers.modsec_audit import (
    _parse_section_b,
    _parse_section_f,
    _split_sections,
)
from src.detector.classifier import build_detector
from src.detector.fasttext_embed import load_fasttext
from src.eval.aggregate import aggregate_table, write_cell_file
from src.utils.config import load_config
from src.utils.io import dump_json
from src.utils.paths import RESULTS_DIR, ROOT
from src.utils.provenance import today_iso
from src.utils.runlog import (
    append_run,
    dataset_version,
    embedding_version,
)
from src.utils.seeds import set_seed

_LOGGER = logging.getLogger("17_post_body_pilot")

RAW_DIR = ROOT / "data" / "raw" / "owasp_modsec_30day"
TABLE_DIR = RESULTS_DIR / "table_13_post_body_pilot"
_TXN_RE = re.compile(r"(?=--[a-f0-9]+-A--)")
_TS_RE = re.compile(r"\[([^\]]+)\]")
_MAX_BODY_CHARS = 2048

def _parse_section_h_tags(lines: List[str]) -> List[str]:
    tags: List[str] = []
    for line in lines:
        for m in re.finditer(r"\[tag \"([^\"]+)\"\]", line):
            tags.append(m.group(1))
    return tags

def _decode_body(raw_body: str) -> str:
    body = raw_body.strip()
    if not body:
        return ""
    try:
        body = urllib.parse.unquote_plus(body)
    except Exception:
        pass
    return body[:_MAX_BODY_CHARS]

def _iter_post_records(max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    tag_map = load_tag_map()
    records: List[Dict[str, Any]] = []
    day_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    skipped_days: List[str] = []
    for day_dir in day_dirs:
        log = day_dir / "modsec_audit.anon.log"
        if not log.exists():
            skipped_days.append(f"{day_dir.name}(absent)")
            continue

        try:
            raw = log.read_bytes().decode("utf-8", errors="replace")
        except OSError as exc:
            _LOGGER.warning("skipping unreadable day %s: %s", day_dir.name, exc)
            skipped_days.append(f"{day_dir.name}({type(exc).__name__})")
            continue
        for txn in _TXN_RE.split(raw):
            if "-C--" not in txn:
                continue
            sections = _split_sections(txn)
            if "C" not in sections or not any(s.strip() for s in sections.get("C", [])):
                continue
            method, path, query_get, ua, _host = _parse_section_b(sections.get("B", []))
            if method.upper() != "POST":
                continue
            status = _parse_section_f(sections.get("F", []))
            tags = _parse_section_h_tags(sections.get("H", []))
            subtype = subtype_from_tags(tags, tag_map)
            body = _decode_body("\n".join(sections.get("C", [])))

            a_lines = sections.get("A", [])
            ts = ""
            if a_lines:
                m = _TS_RE.search(a_lines[0])
                ts = m.group(1) if m else ""
            rec = {
                "method": method,
                "path": path,
                "query_get": query_get,
                "query_get_post": (query_get + " " + body).strip(),
                "ua": ua,
                "status": status,
                "timing": 0,
                "attack_subtype": subtype,
                "label": "attack",
                "_ts": ts,
                "_day": day_dir.name,
            }
            records.append(rec)
            if max_records is not None and len(records) >= max_records:
                return records
    if skipped_days:
        _LOGGER.warning("POST-body pilot skipped %d day file(s): %s",
                        len(skipped_days), ", ".join(skipped_days))
    return records

def _project_query(records: List[Dict[str, Any]], which: str) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        c = dict(r)
        c["query"] = r["query_get"] if which == "get" else r["query_get_post"]
        out.append(c)
    return out

def _fit_eval(cfg, ft_model, records, seed, budget) -> Dict[str, Any]:
    set_seed(int(seed))
    cfg = dict(cfg)
    cfg["seed"] = int(seed)
    detector = build_detector(cfg, ft_model)
    n = len(records)
    idx = select_label_budget(n, budget)
    train = [records[i] for i in idx]
    train_set = set(idx)
    test = [records[i] for i in range(n) if i not in train_set]
    detector.fit(train, subtype_labels(train))
    preds = detector.predict(test)
    gold = subtype_labels(test)
    return {
        "macro_f1": round(100.0 * macro_f1(preds, gold, len(SUBTYPES)), 4),
        "weighted_f1": round(100.0 * weighted_f1(preds, gold, len(SUBTYPES)), 4),
        "n_train": len(train),
        "n_test": len(test),
    }

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="POST-body (section C) detection pilot (Table 13).")
    parser.add_argument("--config", default="configs/ablation/six_field.yaml",
                        help="6-field detector config (the pilot is field-aware).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, default=0.5,
                        help="Labeled prefix fraction of the POST slice (default 0.5).")
    parser.add_argument("--max-records", type=int, default=None, help="Cap (smoke only).")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--aggregate", action="store_true",
                        help="Collect per-seed cells -> results.json and exit.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if args.aggregate:
        agg = aggregate_table(TABLE_DIR, commit=git_commit(), gpu="cpu")
        print(f"[17_post_body_pilot] aggregated -> {agg}")
        return 0

    cfg = load_config(args.config)
    ft_model = load_fasttext(cfg["fasttext"]["model_path"])

    records = _iter_post_records(max_records=args.max_records)
    if not records:
        _LOGGER.error("no POST-body (section C) records found under %s", RAW_DIR)
        return 1
    _LOGGER.info("POST-body slice: n=%d transactions (method=POST, section C present)", len(records))

    get_recs = _project_query(records, "get")
    getpost_recs = _project_query(records, "get_post")

    res_get = _fit_eval(cfg, ft_model, get_recs, args.seed, args.budget)
    res_getpost = _fit_eval(cfg, ft_model, getpost_recs, args.seed, args.budget)
    delta = round(res_getpost["macro_f1"] - res_get["macro_f1"], 4)

    _LOGGER.info(
        "POST-body pilot seed=%d budget=%.2f n_test=%d | GET-only macro_f1=%.2f -> "
        "GET+POST-body macro_f1=%.2f (delta=%+.2f)",
        args.seed, args.budget, res_get["n_test"],
        res_get["macro_f1"], res_getpost["macro_f1"], delta,
    )

    if args.no_write:
        print("[17_post_body_pilot] (nothing written: --no-write)")
        return 0

    meta = {
        "commit": git_commit(),
        "date": today_iso(),
        "dataset_version": dataset_version(),
        "embedding_version": embedding_version(),
        "gpu": "cpu",
        "deterministic": True,
        "n_post_slice": len(records),
        "n_test": res_get["n_test"],
        "budget": args.budget,
        "note": ("POST-body (section C) pilot on the OWASP POST-borne slice; "
                 "the only feature change between rows is the query field "
                 "(GET-only vs GET+decoded POST body). Same 6-field detector, "
                 "same CRS labels, same time-ordered split."),
    }
    table_name = TABLE_DIR.name
    for row, res in (("query = GET only", res_get),
                     ("query = GET + POST body", res_getpost)):
        write_cell_file(TABLE_DIR, table=table_name, row=row, col="macro_f1",
                        value=res["macro_f1"], seed=int(args.seed),
                        metadata={**meta, "weighted_f1": res["weighted_f1"]})
    write_cell_file(TABLE_DIR, table=table_name, row="POST-body lift (delta)",
                    col="macro_f1", value=delta, seed=int(args.seed), metadata=meta)

    append_run(
        stage="post_body_pilot_table13",
        config_path=args.config,
        seed=int(args.seed),
        command_line=f"scripts/17_post_body_pilot.py --seed {args.seed} --budget {args.budget}",
        artifact=str(TABLE_DIR),
        notes=f"get={res_get['macro_f1']} getpost={res_getpost['macro_f1']} delta={delta} n={len(records)}",
    )
    dump_json(TABLE_DIR / "pilot_summary.json", {
        "get_only": res_get, "get_plus_post": res_getpost, "delta": delta, "metadata": meta,
    })
    print(f"[17_post_body_pilot] wrote 3 cells + summary under {TABLE_DIR}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
