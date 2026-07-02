
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_HASHED_DIRS = ("src", "scripts", "configs")
_HASHED_EXT = {".py", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml", ".sh"}
_SKIP_DIR_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}

def source_content_hash(root: Path) -> "tuple[str, int]":
    h = hashlib.sha256()
    n = 0
    for d in _HASHED_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            if any(part in _SKIP_DIR_PARTS for part in f.parts):
                continue
            if f.suffix.lower() not in _HASHED_EXT:
                continue
            rel = f.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
            n += 1
    return h.hexdigest(), n

def main(argv: "list[str] | None" = None) -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    manifest_path = root / "PROVENANCE.json"
    if not manifest_path.is_file():
        print(f"[verify] PROVENANCE.json not found at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("source_content_hash", {}).get("value")
    if not expected:
        print("[verify] manifest has no source_content_hash.value")
        return 1

    actual, n = source_content_hash(root)
    if actual == expected:
        print(f"[verify] source_content_hash MATCH ({n} files): {actual}")
        print("[verify] PASS -- artifact source matches PROVENANCE.json.")
        return 0
    print("[verify] source_content_hash MISMATCH")
    print(f"  expected: {expected}")
    print(f"  actual:   {actual}  ({n} files)")
    print("[verify] FAIL -- artifact source differs from the manifest.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
