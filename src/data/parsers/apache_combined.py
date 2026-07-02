
from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

__all__ = [
    "parse_file",
    "parse_line",
    "split_target",
    "RECORD_KEYS",
    "MODEL_FIELDS",
]

MODEL_FIELDS: Tuple[str, ...] = ("method", "path", "query", "ua", "status", "timing")

RECORD_KEYS: Tuple[str, ...] = (
    "unique_id",
    "timestamp",
    "day",
    "method",
    "path",
    "query",
    "ua",
    "status",
    "timing",
    "host",
    "bytes",
    "referer",
    "label",
    "attack_subtype",
)

_PREFIX_RE = re.compile(
    r"^(?P<host>\S+)\s+(?P<ident>\S+)\s+(?P<authuser>\S+)\s+"
    r"\[(?P<ts>[^\]]*)\]\s*"
)

_HTTP_VERSION_RE = re.compile(r"^HTTP/\d(?:\.\d)?$")

_APACHE_TS_RE = re.compile(
    r"^(?P<d>\d{2})/(?P<mon>[A-Za-z]{3})/(?P<y>\d{4})"
    r":(?P<H>\d{2}):(?P<M>\d{2}):(?P<S>\d{2})"
    r"(?:\s+(?P<tz>[+-]\d{4}))?\s*$"
)

_MONTHS = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}

_DAY_RE = _APACHE_TS_RE

def _apache_ts_to_iso(ts: str) -> str:
    m = _APACHE_TS_RE.match(ts.strip())
    if not m:
        return ts.strip()
    mon = _MONTHS.get(m.group("mon").title())
    if mon is None:
        return ts.strip()
    base = f"{m.group('y')}-{mon}-{m.group('d')}T{m.group('H')}:{m.group('M')}:{m.group('S')}"
    tz = m.group("tz")
    if tz:
        base += f"{tz[:3]}:{tz[3:]}"
    return base

def _apache_ts_to_day(ts: str) -> str:
    m = _DAY_RE.match(ts.strip())
    if not m:
        return ""
    return f"{m.group('d')}-{m.group('mon').title()}-{m.group('y')}"

def split_target(target: str) -> Tuple[str, str]:
    s = target.strip()
    if not s:
        return "", ""
    scheme_idx = s.find("://")
    if scheme_idx != -1:
        after_scheme = s[scheme_idx + 3:]
        slash = after_scheme.find("/")
        s = after_scheme[slash:] if slash != -1 else ""
    if "?" in s:
        path, query = s.split("?", 1)
    else:
        path, query = s, ""
    return path, query

def _parse_request_line(request: str) -> Tuple[str, str, str]:
    line = request.strip()
    if not line:
        return "", "", ""
    first_sp = line.find(" ")
    if first_sp == -1:

        return line, "", ""
    method = line[:first_sp]
    remainder = line[first_sp + 1:].strip()
    last_sp = remainder.rfind(" ")
    if last_sp != -1 and _HTTP_VERSION_RE.match(remainder[last_sp + 1:]):
        target = remainder[:last_sp].rstrip()
    else:
        target = remainder
    path, query = split_target(target)
    return method, path, query

def _tokenize_quoted(rest: str) -> Tuple[List[str], List[str]]:
    quoted: List[str] = []
    bare: List[str] = []
    n = len(rest)
    i = 0
    cur_bare: List[str] = []

    def flush_bare() -> None:
        if cur_bare:
            bare.append("".join(cur_bare))
            cur_bare.clear()

    while i < n:
        c = rest[i]
        if c == '"':
            flush_bare()
            i += 1
            buf: List[str] = []
            while i < n:
                ch = rest[i]
                if ch == "\\" and i + 1 < n:
                    buf.append(rest[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    break
                buf.append(ch)
                i += 1
            quoted.append("".join(buf))
            if i < n and rest[i] == '"':
                i += 1
            continue
        if c.isspace():
            flush_bare()
            i += 1
            continue
        cur_bare.append(c)
        i += 1
    flush_bare()
    return quoted, bare

def _to_int(token: str) -> int:
    token = token.strip()
    if not token or token == "-":
        return 0
    try:
        return int(token)
    except ValueError:
        return 0

def parse_line(line: str, line_index: int = 0) -> Optional[Dict]:
    if not line or not line.strip():
        return None

    m = _PREFIX_RE.match(line)
    if not m:
        return None

    host = m.group("host")
    ts = m.group("ts")
    rest = line[m.end():]

    quoted, bare = _tokenize_quoted(rest)

    request = quoted[0] if len(quoted) >= 1 else ""
    referer = quoted[1] if len(quoted) >= 2 else ""
    ua = quoted[2] if len(quoted) >= 3 else ""

    status = _to_int(bare[0]) if len(bare) >= 1 else 0
    nbytes = _to_int(bare[1]) if len(bare) >= 2 else 0

    method, path, query = _parse_request_line(request)

    if method == "" and host in ("", "-"):
        return None

    return {
        "unique_id": f"weblog-{line_index:09d}",
        "timestamp": _apache_ts_to_iso(ts),
        "day": _apache_ts_to_day(ts),
        "method": method,
        "path": path,
        "query": query,
        "ua": ua,
        "status": status,
        "timing": 0,
        "host": host,
        "bytes": nbytes,
        "referer": "" if referer == "-" else referer,
        "label": "benign",
        "attack_subtype": "",
    }

def parse_file(path: "str | os.PathLike[str]") -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for line_index, raw_line in enumerate(fh):
            record = parse_line(raw_line, line_index=line_index)
            if record is not None:
                yield record

def _self_test() -> None:
    import json

    from pathlib import Path

    sample_lines = [
        '54.36.149.41 - - [22/Jan/2019:03:56:14 +0330] "GET /filter/27?p=53 HTTP/1.1" 200 30577 "-" "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)" "-"',
        '31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET /image/60844/productModel/200x200 HTTP/1.1" 200 5667 "https://www.zanbil.ir/m/filter/b113" "Mozilla/5.0 (Linux; Android 6.0) Chrome/66.0 Mobile Safari/537.36" "-"',
        '207.46.13.136 - - [22/Jan/2019:03:56:18 +0330] "POST /api/order HTTP/1.1" 201 12 "-" "curl/7.64" "-"',
    ]

    raw_candidates = [
        Path("data/raw/web_access_logs/access.log"),
    ]
    raw_path = next((p for p in raw_candidates if p.is_file()), None)

    if raw_path is None:
        print(
            "[apache_combined self-test] raw web-access log not found "
            f"(looked in {[str(p) for p in raw_candidates]}); "
            "parsing hard-coded sample lines instead."
        )
        for i, line in enumerate(sample_lines):
            rec = parse_line(line, line_index=i)
            print(json.dumps(rec, sort_keys=True, ensure_ascii=False))
    else:
        print(f"[apache_combined self-test] parsing first 3 lines of {raw_path}")
        count = 0
        for rec in parse_file(raw_path):
            print(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            count += 1
            if count >= 3:
                break

if __name__ == "__main__":
    _self_test()
