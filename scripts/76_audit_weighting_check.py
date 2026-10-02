
from __future__ import annotations

import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import git_commit
from src.utils.paths import DATA_DIR, RESULTS_DIR

_XLSX = DATA_DIR / "manual_verification" / "owasp_1k_subset_human.xlsx"
_MANIFEST = DATA_DIR / "manual_verification" / "sample_manifest.json"
_OUT = RESULTS_DIR / "manual_verification" / "weighting_check.json"

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

_UNDECIDABLE = {"normal", "unsure", "undecidable", ""}

def _norm(v: str) -> str:
    v = (v or "").strip().lower()
    return "benign" if v in ("false_positive", "false positive", "fp") else v

def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1

def read_sheet(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    table = []
    for row in sheet.iter(f"{{{_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            if c.get("t") == "inlineStr":
                text = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
            elif v is None:
                text = ""
            elif c.get("t") == "s":
                text = shared[int(v.text)]
            else:
                text = v.text or ""
            cells[_col_index(c.get("r"))] = text
        if cells:
            table.append([cells.get(i, "") for i in range(max(cells) + 1)])
    header = [h.strip() for h in table[0]]
    return [dict(zip(header, r + [""] * (len(header) - len(r)))) for r in table[1:]
            if any(x.strip() for x in r)]

def main() -> int:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    weight = manifest["sample_weight"]
    population = manifest["corpus_counts"]
    rows = read_sheet(_XLSX)

    n, und, dec, agree = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        crs = r["crs_subtype"].strip()
        hv = _norm(r.get("label_human", ""))
        n[crs] += 1
        if hv in _UNDECIDABLE:
            und[crs] += 1
        else:
            dec[crs] += 1
            agree[crs] += int(hv == crs)

    strata = sorted(n)
    w_n = sum(weight[c] * n[c] for c in strata)
    w_und = sum(weight[c] * und[c] for c in strata)
    w_dec = sum(weight[c] * dec[c] for c in strata)
    w_agree = sum(weight[c] * agree[c] for c in strata)

    out = {
        "table": "manual_verification",
        "population": {
            "split": manifest.get("split"),
            "n_total": manifest["n_total"],
            "note": "stratified sample of the TEST split; weights = test_count / sampled_count",
        },
        "per_stratum": {
            c: {
                "test_count": population[c],
                "sampled": n[c],
                "undecidable": und[c],
                "decidable": dec[c],
                "agree": agree[c],
                "weight": weight[c],
            }
            for c in strata
        },
        "totals": {
            "sampled": sum(n.values()),
            "undecidable": sum(und.values()),
            "decidable": sum(dec.values()),
            "agree": sum(agree.values()),
            "sum_weight_x_sampled": w_n,
        },
        "rates_unrounded_pct": {
            "undecidable_raw": 100.0 * sum(und.values()) / sum(n.values()),
            "undecidable_weighted": 100.0 * w_und / w_n,
            "agreement_raw_on_decidable": 100.0 * sum(agree.values()) / sum(dec.values()),
            "agreement_weighted_on_decidable": 100.0 * w_agree / w_dec,
        },
        "formulas": {
            "undecidable_weighted": "sum_c w_c*undecidable_c / sum_c w_c*n_c",
            "agreement_weighted_on_decidable": "sum_c w_c*agree_c / sum_c w_c*decidable_c",
        },
        "metadata": {
            "commit": git_commit(),
            "worksheet": str(_XLSX.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "manifest": str(_MANIFEST.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "column": "label_human",
            "undecidable_vocabulary": sorted(_UNDECIDABLE),
        },
    }
    _OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[76] wrote {_OUT}")
    for k, v in out["rates_unrounded_pct"].items():
        print(f"  {k}: {v:.6f}%")
    for c in strata:
        print(f"  {c:15s} n={n[c]:4d} und={und[c]:4d} dec={dec[c]:4d} agree={agree[c]:4d} w={weight[c]:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
