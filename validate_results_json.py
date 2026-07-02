
from __future__ import annotations

import json
import pathlib
import sys

def main(argv: "list[str] | None" = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python tools/validate_results_json.py <export_dir>")
        return 2
    root = pathlib.Path(args[0])
    files = sorted(root.glob("results/**/results.json"))
    bad = []
    for p in files:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append((p.relative_to(root), str(e)))
    if bad:
        print("[export] JSON VALIDATION FAILED:")
        for p, e in bad[:20]:
            print(f"  - {p}: {e}")
        return 1
    print(f"[export] JSON validation: PASS ({len(files)} results.json files)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
