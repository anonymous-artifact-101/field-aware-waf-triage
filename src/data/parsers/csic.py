
from __future__ import annotations

import csv
import io
import os
from typing import Dict, Iterator, List, Optional, Tuple

__all__ = [
    "parse_file",
    "parse_row",
    "split_url",
    "RECORD_KEYS",
    "MODEL_FIELDS",
]

MODEL_FIELDS: Tuple[str, ...] = ("method", "path", "query", "ua", "status", "timing")

RECORD_KEYS: Tuple[str, ...] = (
    "unique_id",
    "day",
    "method",
    "path",
    "query",
    "ua",
    "status",
    "timing",
    "host",
    "content",
    "csic_class",
    "label",
    "attack_subtype",
)

_LABEL_NORMAL = "normal"
_LABEL_ANOMALOUS = "anomalous"

_CSIC_ATTACK_SUBTYPE = "protocol"

_CSIC_DAY = "csic2010"

def split_url(url: str) -> Tuple[str, str]:
    s = url.strip()
    if not s:
        return "", ""

    last_sp = s.rfind(" ")
    if last_sp != -1:
        tail = s[last_sp + 1:]
        if tail[:5] == "HTTP/":
            s = s[:last_sp].rstrip()

    scheme_idx = s.find("://")
    if scheme_idx != -1:
        after_scheme = s[scheme_idx + 3:]
        slash = after_scheme.find("/")
        if slash == -1:

            s = ""
        else:
            s = after_scheme[slash:]

    if "?" in s:
        path, query = s.split("?", 1)
    else:
        path, query = s, ""
    return path, query

def _find_label_column(header: List[str], sample_row: Optional[List[str]]) -> int:
    if sample_row is not None:
        for idx, cell in enumerate(sample_row):
            if cell.strip().lower() in (_LABEL_NORMAL, _LABEL_ANOMALOUS):
                return idx
    return 0

def _col(header_index: Dict[str, int], row: List[str], name: str) -> str:
    idx = header_index.get(name.lower())
    if idx is None or idx >= len(row):
        return ""
    return row[idx].strip()

def parse_row(
    row: List[str],
    header: List[str],
    label_col: Optional[int] = None,
    row_index: int = 0,
) -> Optional[Dict]:
    if not row or all(cell.strip() == "" for cell in row):
        return None

    header_index = {h.strip().lower(): i for i, h in enumerate(header)}

    if label_col is None:
        label_col = _find_label_column(header, row)
    raw_class = row[label_col].strip() if label_col < len(row) else ""
    is_attack = raw_class.lower() == _LABEL_ANOMALOUS

    method = _col(header_index, row, "method")
    ua = _col(header_index, row, "user-agent")
    host = _col(header_index, row, "host")
    url = _col(header_index, row, "url")
    content = _col(header_index, row, "content")

    path, query = split_url(url)

    if query == "" and content:
        query = content

    label = "attack" if is_attack else "benign"

    attack_subtype = _CSIC_ATTACK_SUBTYPE if is_attack else ""

    return {
        "unique_id": f"csic-{row_index:07d}",
        "day": _CSIC_DAY,
        "method": method,
        "path": path,
        "query": query,
        "ua": ua,
        "status": 0,
        "timing": 0,
        "host": host,
        "content": content,
        "csic_class": raw_class,
        "label": label,
        "attack_subtype": attack_subtype,
    }

def _iter_rows(text: str) -> Iterator[Tuple[int, List[str], List[str]]]:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return
    for row_index, row in enumerate(reader):
        yield row_index, header, row

def parse_file(path: "str | os.PathLike[str]") -> Iterator[Dict]:
    with open(path, "r", encoding="latin-1", newline="") as fh:
        text = fh.read()

    label_col: Optional[int] = None
    for row_index, header, row in _iter_rows(text):
        if label_col is None and row and any(c.strip() for c in row):
            label_col = _find_label_column(header, row)
        record = parse_row(row, header, label_col=label_col, row_index=row_index)
        if record is not None:
            yield record

def _self_test() -> None:
    import json

    header = [
        "", "Method", "User-Agent", "Pragma", "Cache-Control", "Accept",
        "Accept-encoding", "Accept-charset", "language", "host", "cookie",
        "content-type", "connection", "lenght", "content", "classification",
        "URL",
    ]
    sample_rows = [

        [
            "Normal", "GET",
            "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)",
            "no-cache", "no-cache", "text/xml", "gzip", "utf-8", "en",
            "localhost:8080", "JSESSIONID=ABC", "", "close", "", "", "0",
            "http://localhost:8080/tienda1/publico/anadir.jsp?id=3&nombre=Vino HTTP/1.1",
        ],

        [
            "Anomalous", "GET",
            "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)",
            "no-cache", "no-cache", "text/xml", "gzip", "utf-8", "en",
            "localhost:8080", "JSESSIONID=DEF", "", "close", "", "", "1",
            "http://localhost:8080/tienda1/publico/anadir.jsp?id=2&cantidad=%27%3B+DROP+TABLE+usuarios%3B+-- HTTP/1.1",
        ],

        [
            "Anomalous", "POST",
            "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)",
            "no-cache", "no-cache", "text/xml", "gzip", "utf-8", "en",
            "localhost:8080", "JSESSIONID=GHI", "application/x-www-form-urlencoded",
            "Connection: close", "Content-Length: 40",
            "id=2&precio=85&cantidad=%27+OR+%271%27%3D%271", "1",
            "http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1",
        ],
    ]
    print("[csic self-test] parsing", len(sample_rows), "hard-coded sample rows")
    for i, row in enumerate(sample_rows):
        rec = parse_row(row, header, row_index=i)
        print(json.dumps(rec, sort_keys=True, ensure_ascii=False))

if __name__ == "__main__":
    _self_test()
