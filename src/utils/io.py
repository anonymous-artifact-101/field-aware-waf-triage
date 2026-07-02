
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = ["write_jsonl", "read_jsonl", "dump_json", "load_json"]

_JSONL_SEPARATORS = (",", ":")

def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: str | os.PathLike[str], records: Iterable[Any]) -> int:
    path = Path(path)
    _ensure_parent(path)
    count = 0

    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            line = json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                separators=_JSONL_SEPARATORS,
            )
            handle.write(line)
            handle.write("\n")
            count += 1
    return count

def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of '{path}': {exc}"
                ) from exc

def dump_json(path: str | os.PathLike[str], obj: Any) -> None:
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(obj, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")

def load_json(path: str | os.PathLike[str]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
