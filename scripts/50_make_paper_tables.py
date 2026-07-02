
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import load_json
from src.utils.paths import RESULTS_DIR, ROOT, ensure_dir

_TABLE_DIRS: Dict[str, str] = {
    "3": "table_03_rq1_baselines",
    "5": "table_05_field_granularity",
    "6": "table_06_per_class_owasp",
    "7": "table_07_attribution",
    "8": "table_08_latency",
    "9": "table_09_cross_dataset_biblio_us17",
    "10": "table_10_fasttext_sweep",
    "11": "table_11_benign_fpr_apache_indo",
    "12": "table_12_label_noise_cleanlab",
    "13": "table_13_post_body_pilot",
    "14": "table_14_budget_composition_ci",
}

_CAPTIONS: Dict[str, str] = {
    "3": "RQ1: supervised macro-F1 (%) vs. label budget.",
    "5": "RQ2: representation ablations and controls, macro-F1 (%) at 5% and 10% label budgets.",
    "6": "Per-class detection over the eight OWASP attack subtypes.",
    "7": "Exploratory field-saliency agreement (\\%) per attack subtype, with the "
    "per-class support and the payload-field populated rate (the corpus ceiling). "
    "Macros exclude the ill-posed \\texttt{protocol} catch-all from the denominator; "
    "the field-observable vs. query-borne split separates the corpus ceiling from "
    "method recovery. Random-field floor 16.7\\% and the degenerate always-path "
    "guard are reported for reference.",
    "8": "Inference latency on CPU (milliseconds) per pipeline stage.",
    "9": "Cross-dataset zero-shot transfer to Biblio-US17 (5 seeds, deterministic; no Biblio "
    "training). The status-masked row blanks the response-code field, isolating genuine "
    "request-field transfer from the Biblio benign=200 / attack=non-200 status shortcut.",
    "10": "FastText efficiency grid (5% budget, seed 42): macro-F1 vs. footprint over vector_size and n-gram range.",
    "11": "Benign false-positive rate (zero-shot) on the Apache-Indo benign probe.",
    "12": "Confident-learning (Cleanlab) audit of suspected CRS subtype-label issues.",
    "13": "POST-body pilot: folding the decoded request body into the query field on the POST-borne slice.",
    "14": "Budget-composition CI: macro-F1 over sliding labeled windows in the time-ordered train split.",
}

class MissingResultsError(FileNotFoundError):
    """Raised (loudly) when a table's results.json is absent or unreadable."""

class ExploratoryResultsError(ValueError):

    def __init__(self, violations: "list[Tuple[str, str, list[str]]]"):
        self.violations = violations
        lines = [
            f"  - cell ({row!r}, {col!r}): {', '.join(reasons)}"
            for row, col, reasons in violations
        ]
        super().__init__(
            "refusing to emit a paper table with exploratory/untrained/no-CI cells:\n"
            + "\n".join(lines)
        )

_EXPLORATORY_FLAGS = ("exploratory_single_seed",)

def _cell_flag(meta: Mapping[str, Any], row: str, col: str, key: str) -> Any:
    cell_flags = meta.get("cell_flags")
    if isinstance(cell_flags, Mapping):
        per = cell_flags.get(f"{row} | {col}")
        if isinstance(per, Mapping) and key in per:
            return per[key]
    return meta.get(key)

def _has_ci(cell: Mapping[str, Any]) -> bool:
    ci = cell.get("ci_95")
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        return False
    lo, hi = ci
    if lo is None or hi is None:
        return False
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(lo_f) or math.isnan(hi_f))

def find_exploratory_cells(doc: Mapping[str, Any]) -> "list[Tuple[str, str, list[str]]]":
    meta = doc.get("metadata", {})
    if not isinstance(meta, Mapping):
        meta = {}
    violations: "list[Tuple[str, str, list[str]]]" = []
    for cell in doc.get("cells", []):
        row, col = str(cell.get("row")), str(cell.get("col"))
        reasons: "list[str]" = []

        single_run = bool(_cell_flag(meta, row, col, "single_run_measured"))
        if not single_run and not _has_ci(cell):
            reasons.append("missing 95% CI (ci_95 is null/absent)")
        for flag in _EXPLORATORY_FLAGS:
            if bool(_cell_flag(meta, row, col, flag)):
                reasons.append(f"flagged {flag}=true")
        if bool(_cell_flag(meta, row, col, "eval_capped")):
            reasons.append("flagged eval_capped=true (not full split)")

        ckpt = _cell_flag(meta, row, col, "checkpoint_loaded")
        if ckpt is False:
            reasons.append("checkpoint_loaded=false (untrained model)")
        if reasons:
            violations.append((row, col, reasons))
    return violations

def _resolve_table_dir(table: str) -> str:
    key = str(table).strip()
    if key in _TABLE_DIRS:
        return _TABLE_DIRS[key]

    if key in _TABLE_DIRS.values():
        return key
    raise SystemExit(
        f"[50_make_paper_tables] unknown --table {table!r}. "
        f"Known: {sorted(_TABLE_DIRS)} or a results dir name {sorted(set(_TABLE_DIRS.values()))}."
    )

def _load_results(results_root: Path, table_dir: str) -> Dict[str, Any]:
    path = results_root / table_dir / "results.json"
    if not path.is_file():
        raise MissingResultsError(
            f"results.json not found for {table_dir}: {path}\n"
            f"Refusing to emit a paper table without its backing results.json. "
            f"Run the experiment that writes it first (e.g. scripts/12, 40, 41, ...)."
        )
    try:
        doc = load_json(path)
    except Exception as exc:
        raise MissingResultsError(f"failed to read {path}: {exc}") from exc
    if not isinstance(doc, Mapping) or "cells" not in doc:
        raise MissingResultsError(
            f"{path} is malformed: expected an object with a 'cells' list "
            f"(fixed schema, PROJECT_STRUCTURE.md section 6)."
        )
    return dict(doc)

def _grid(
    cells: List[Mapping[str, Any]],
    *,
    col_order: "Optional[Sequence[str]]" = None,
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Mapping[str, Any]]]:
    rows: List[str] = []
    cols: List[str] = []
    lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for cell in cells:
        row = str(cell["row"])
        col = str(cell["col"])
        if row not in rows:
            rows.append(row)
        if col not in cols:
            cols.append(col)
        lookup[(row, col)] = cell
    if col_order:
        ordered = [c for c in col_order if c in cols]
        ordered += [c for c in cols if c not in ordered]
        cols = ordered
    return rows, cols, lookup

_METRIC_DECIMALS: Dict[str, int] = {
    "latency_ms": 3,
    "footprint_mb": 2,
    "count": 0,
    "populated_pct": 1,
}
_DEFAULT_DECIMALS = 1

_TABLE7_COLS = ("agreement", "support", "populated")
_TABLE7_ROW_ORDER = (
    "lfi", "scanner", "php_injection", "path_traversal",
    "sql_injection", "rce", "xss", "protocol",
    "Macro (mean)", "Macro (excl. protocol)",
    "Macro (field-observable)", "Macro (query-borne)",
    "Random-field floor", "Always-path baseline", "Query-empty rate",
)

_TABLE7_BLOCK_ROWS = (
    "Macro (mean)", "Macro (excl. protocol)",
    "Macro (field-observable)", "Macro (query-borne)",
    "Random-field floor", "Always-path baseline", "Query-empty rate",
)
_TABLE7_CAPTION = (
    "Exploratory field-saliency agreement (\\%) per attack subtype: agreement "
    "is a triage diagnostic, support the per-class test count, and \\emph{populated} "
    "the fraction of that subtype's records whose payload field the log records "
    "(the corpus ceiling). Macros exclude the ill-posed \\texttt{protocol} "
    "catch-all; the field-observable vs.\\ query-borne split separates the corpus "
    "ceiling from method recovery."
)

_SPLIT_TABLE_NUMS = ("3",)
_ANOMALY_METRICS = ("mean_anomaly_score", "anomaly_rate_pct")
_CRS_REFERENCE_ROWS = ("ModSec-Learn", "ModSec-AdvLearn")

_TABLE3_F1_COLS = ("1%", "5%", "10%", "20%", "50%")

_TABLE3_F1_ROW_ORDER = (
    "Status-only (shortcut)",
    "TF-IDF + LogReg",
    "Char-CNN",
    "TF-IDF + SVD + HGBDT",
    "Proposed (FastText field-aware)",
)
_TABLE3_CAPTION = (
    "RQ1: supervised macro-F1 (\\%). The 10\\% column is the reported "
    "frontier point. CRS-input references$^{\\ast}$ are label-adjacent, not "
    "raw-log peers."
)

_TABLE3_FOOTNOTE_LATEX = (
    r"\multicolumn{{ncol}}{l}{\footnotesize $^{\ast}$Label-adjacent CRS-input "
    r"references; HGBDT CI $[66.8,68.7]$; proposed weighted-$F_1=94.9$.}"
)
_TABLE3_CAPTION_MD = (
    "RQ1: supervised macro-F1 (%). The 10% column is the reported frontier "
    "point. CRS-input references (*) are label-adjacent, not raw-log peers."
)
_TABLE3_FOOTNOTE_MD = (
    "_* Label-adjacent CRS-input references; HGBDT CI [66.8,68.7]; "
    "proposed weighted-F1=94.9._"
)

def _is_anomaly_row(meta: Mapping[str, Any], row: str, cols: List[str]) -> bool:
    has_anomaly = False
    for col in cols:
        metric = _cell_flag(meta, row, col, "metric")
        if isinstance(metric, str):
            if metric == "macro_f1":
                return False
            if metric in _ANOMALY_METRICS:
                has_anomaly = True
    return has_anomaly

def _split_table3(
    doc: Mapping[str, Any],
) -> "Tuple[dict, dict]":
    meta = dict(doc.get("metadata", {}))
    rows, cols, _ = _grid(list(doc["cells"]))
    anomaly_rows = {r for r in rows if _is_anomaly_row(meta, r, cols)}

    f1_cells: List[Mapping[str, Any]] = []
    anom_by_row: Dict[str, Mapping[str, Any]] = {}
    for cell in doc["cells"]:
        row, col = str(cell["row"]), str(cell["col"])
        if row in anomaly_rows:
            if col == "0%":
                anom_by_row[row] = {**cell, "col": "Value"}
        elif col == "0%":
            anom_by_row[f"{row} (centroid)"] = {**cell, "row": f"{row} (centroid)", "col": "Value"}
        elif col in _TABLE3_F1_COLS:
            f1_cells.append(cell)

    f1_rows = [r for r in _TABLE3_F1_ROW_ORDER if r in rows]
    f1_rows += [r for r in rows if r not in anomaly_rows and r not in f1_rows and r not in _CRS_REFERENCE_ROWS]
    f1_rows += [r for r in _CRS_REFERENCE_ROWS if r in rows]

    f1_lookup = {(str(c["row"]), str(c["col"])): c for c in f1_cells}
    ordered_f1: List[Mapping[str, Any]] = []
    for r in f1_rows:
        for c in _TABLE3_F1_COLS:
            if (r, c) in f1_lookup:
                ordered_f1.append(f1_lookup[(r, c)])

    anom_rows = [r for r in anom_by_row]
    ordered_anom = [anom_by_row[r] for r in anom_rows]

    f1_doc = {"cells": ordered_f1, "metadata": meta, "table": doc.get("table", "")}
    anom_doc = {"cells": ordered_anom, "metadata": meta, "table": doc.get("table", "")}
    return f1_doc, anom_doc

def _fmt_value(
    cell: Optional[Mapping[str, Any]],
    *,
    latex: bool,
    exploratory: bool = False,
    decimals: int = _DEFAULT_DECIMALS,
) -> str:
    if cell is None:
        return "--"
    value = cell.get("value")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{float(value):.{decimals}f}"
    ci = cell.get("ci_95")
    if ci and isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        half = (float(ci[1]) - float(ci[0])) / 2.0
        if half > 0 and not math.isnan(half):
            text += (r" $\pm$ " if latex else " +/- ") + f"{half:.{decimals}f}"
    if exploratory:
        text += (r" \textbf{[EXPLORATORY]}" if latex else " **[EXPLORATORY]**")
    return text

def _cell_decimals(meta: Mapping[str, Any], row: str, col: str, table_num: str = "") -> int:

    if str(table_num) == "9":
        col_l = col.lower()
        if any(key in col_l for key in ("auc", "tpr", "fpr", "precision")):
            return 3
    metric = _cell_flag(meta, row, col, "metric")
    if isinstance(metric, str) and metric in _METRIC_DECIMALS:
        return _METRIC_DECIMALS[metric]
    return _DEFAULT_DECIMALS

def _latex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(a, b)
    return text

def render_latex(
    doc: Mapping[str, Any],
    table_num: str,
    *,
    exploratory_cells: "Optional[set[Tuple[str, str]]]" = None,
    caption: "Optional[str]" = None,
    label: "Optional[str]" = None,
    split_block_rows: "Optional[Sequence[str]]" = None,
    footnote_latex: "Optional[str]" = None,
    col_order: "Optional[Sequence[str]]" = None,
) -> str:
    rows, cols, lookup = _grid(list(doc["cells"]), col_order=col_order)
    flagged = exploratory_cells or set()
    meta = doc.get("metadata", {})
    cap = caption if caption is not None else _latex_escape(
        _CAPTIONS.get(table_num, doc.get("table", "results"))
    )
    if label is not None:
        label_line = f"\\label{{{label}}}"
    elif table_num.isdigit():
        label_line = f"\\label{{tab:table_{int(table_num):02d}}}"
    else:
        label_line = r"\label{tab:results}"
    col_spec = "l" + "r" * len(cols)
    lines = [
        r"% Auto-generated by scripts/50_make_paper_tables.py -- do not edit by hand.",
        f"% Source: results/{doc.get('table', '')}/results.json "
        f"(commit {doc.get('metadata', {}).get('commit', '?')}, "
        f"date {doc.get('metadata', {}).get('date', '?')})",
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{cap}}}",
        label_line,
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join([""] + [_latex_escape(c) for c in cols]) + r" \\",
        r"\midrule",
    ]

    def _row_line(row: str) -> str:
        cells = [
            _fmt_value(lookup.get((row, c)), latex=True, exploratory=(row, c) in flagged,
                       decimals=_cell_decimals(meta, row, c, table_num))
            for c in cols
        ]
        return " & ".join([_latex_escape(row)] + cells) + r" \\"

    split = list(split_block_rows or [])
    for row in rows:
        if row in split:
            continue
        lines.append(_row_line(row))
    if split:
        lines.append(r"\midrule")
        split_rows = [row for row in split if any((row, c) in lookup for c in cols)]
        for idx, row in enumerate(split_rows):
            if (row, cols[0]) in lookup or any((row, c) in lookup for c in cols):
                lines.append(_row_line(row))
                if table_num == "3" and idx < len(split_rows) - 1:
                    lines.append(r"\addlinespace[1pt]")

    lines.append(r"\bottomrule")
    if footnote_latex:

        lines.append(footnote_latex.replace("{ncol}", str(len(cols) + 1)))
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)

def render_markdown(
    doc: Mapping[str, Any],
    table_num: str,
    *,
    exploratory_cells: "Optional[set[Tuple[str, str]]]" = None,
    caption: "Optional[str]" = None,
    split_block_rows: "Optional[Sequence[str]]" = None,
    footnote_md: "Optional[str]" = None,
    col_order: "Optional[Sequence[str]]" = None,
) -> str:
    rows, cols, lookup = _grid(list(doc["cells"]), col_order=col_order)
    flagged = exploratory_cells or set()
    cap = caption if caption is not None else _CAPTIONS.get(table_num, doc.get("table", "results"))
    meta = doc.get("metadata", {})
    header = ["| | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 1)]

    def _row_line(row: str) -> str:
        cells = [
            _fmt_value(lookup.get((row, c)), latex=False, exploratory=(row, c) in flagged,
                       decimals=_cell_decimals(meta, row, c, table_num))
            for c in cols
        ]
        return "| " + " | ".join([row] + cells) + " |"

    split = list(split_block_rows or [])
    body: List[str] = [_row_line(row) for row in rows if row not in split]
    body += [_row_line(row) for row in split if any((row, c) in lookup for c in cols)]

    lines = [
        f"<!-- Auto-generated by scripts/50_make_paper_tables.py from "
        f"results/{doc.get('table', '')}/results.json -->",
        f"### {cap}",
        "",
        *header,
        *body,
        "",
    ]
    if footnote_md:
        lines += [footnote_md, ""]
    lines += [
        f"_Source: `results/{doc.get('table','')}/results.json` "
        f"(commit `{meta.get('commit','?')}`, date {meta.get('date','?')})._",
        "",
    ]
    return "\n".join(lines)

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Render a paper table from its results.json.")
    parser.add_argument("--table", required=True, help="Table number (3,5,6,7,8,9,10,11) or results dir name.")
    parser.add_argument("--results-dir", default=None, help="Override results/ root.")
    parser.add_argument("--out-dir", default=None, help="Override paper/tables/ output dir.")
    parser.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="DRAFT ONLY: do not refuse a table with exploratory/untrained/no-CI cells; "
        "instead annotate each offending cell inline with [EXPLORATORY].",
    )
    args = parser.parse_args(argv)

    table_dir = _resolve_table_dir(args.table)
    results_root = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (ROOT / "paper" / "tables"))

    try:
        doc = _load_results(results_root, table_dir)
    except MissingResultsError as exc:
        raise SystemExit(f"[50_make_paper_tables] ERROR: {exc}")

    violations = find_exploratory_cells(doc)
    exploratory_cells: "set[Tuple[str, str]]" = set()
    if violations:
        if not args.allow_exploratory:
            raise SystemExit(
                f"[50_make_paper_tables] ERROR: {ExploratoryResultsError(violations)}\n"
                f"Aggregate the seed set {{42..46}} into proper 95% CIs first "
                f"(src.eval.aggregate.aggregate_table), or pass --allow-exploratory "
                f"to render a DRAFT table with [EXPLORATORY] annotations."
            )
        exploratory_cells = {(row, col) for row, col, _ in violations}
        print(
            "[50_make_paper_tables] WARNING: --allow-exploratory set; "
            f"{len(exploratory_cells)} cell(s) annotated [EXPLORATORY] (DRAFT ONLY, do not ship):"
        )
        for row, col, reasons in violations:
            print(f"    - ({row!r}, {col!r}): {', '.join(reasons)}")

    table_num = str(args.table).strip()
    num_for_name = table_num if table_num.isdigit() else table_dir
    stem = f"table_{int(num_for_name):02d}" if str(num_for_name).isdigit() else table_dir
    num_arg = table_num if table_num.isdigit() else "?"

    def _write(stem_: str, tex: str, md: str) -> None:
        (out_dir / f"{stem_}.tex").write_text(tex, encoding="utf-8")
        (out_dir / f"{stem_}.md").write_text(md, encoding="utf-8")
        print(f"[50_make_paper_tables] wrote {out_dir / f'{stem_}.tex'} and {out_dir / f'{stem_}.md'}")

    if table_num in _SPLIT_TABLE_NUMS:

        f1_doc, _anom_doc = _split_table3(doc)
        _write(
            stem,
            render_latex(
                f1_doc, num_arg, exploratory_cells=exploratory_cells,
                caption=_TABLE3_CAPTION, label="tab:t3",
                split_block_rows=_CRS_REFERENCE_ROWS, footnote_latex=_TABLE3_FOOTNOTE_LATEX,
                col_order=_TABLE3_F1_COLS,
            ),
            render_markdown(
                f1_doc, num_arg, exploratory_cells=exploratory_cells,
                caption=_TABLE3_CAPTION_MD, split_block_rows=_CRS_REFERENCE_ROWS,
                col_order=_TABLE3_F1_COLS, footnote_md=_TABLE3_FOOTNOTE_MD,
            ),
        )
        print(
            f"[50_make_paper_tables] table {args.table} ({table_dir}): "
            f"{len(f1_doc['cells'])} supervised macro-F1 cells rendered; 0%-label native metrics left to text"
        )
        return 0

    if table_dir == "table_07_attribution":

        tex = render_latex(
            doc, num_arg, exploratory_cells=exploratory_cells,
            caption=_TABLE7_CAPTION, label="tab:t7",
            col_order=_TABLE7_COLS, split_block_rows=list(_TABLE7_BLOCK_ROWS),
        )
        md = render_markdown(
            doc, num_arg, exploratory_cells=exploratory_cells,
            caption=_TABLE7_CAPTION, col_order=_TABLE7_COLS,
            split_block_rows=list(_TABLE7_BLOCK_ROWS),
        )
        _write(stem, tex, md)
        print(f"[50_make_paper_tables] table {args.table} ({table_dir}): {len(doc['cells'])} cells")
        return 0

    tex = render_latex(doc, num_arg, exploratory_cells=exploratory_cells)
    md = render_markdown(doc, num_arg, exploratory_cells=exploratory_cells)
    _write(stem, tex, md)
    print(f"[50_make_paper_tables] table {args.table} ({table_dir}): {len(doc['cells'])} cells")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
