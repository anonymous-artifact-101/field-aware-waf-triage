
from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Tuple

from src.data.labels import subtype_from_tags

__all__ = [
    "parse_file",
    "parse_transaction",
    "extract_bracket_kvs",
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
    "client_ip",
    "crs_rule_ids",
    "crs_tags",
    "severities",
    "n_rules",
    "label",
    "attack_subtype",
)

_BOUNDARY_RE = re.compile(r"^--([0-9a-fA-F]+)-([A-Z])--\s*$")

_SECTION_A_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+"
    r"(?P<uid>\S+)\s+"
    r"(?P<client_ip>\S+)\s+"
    r"(?P<client_port>\S+)\s+"
    r"(?P<server_ip>\S+)\s+"
    r"(?P<server_port>\S+)\s*$"
)

_STATUS_RE = re.compile(r"^HTTP/\d(?:\.\d)?\s+(\d{3})\b")

_HTTP_VERSION_RE = re.compile(r"^HTTP/\d(?:\.\d)?$")

_MONTHS = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}

_APACHE_TS_RE = re.compile(
    r"^(?P<d>\d{2})/(?P<mon>[A-Za-z]{3})/(?P<y>\d{4})"
    r":(?P<H>\d{2}):(?P<M>\d{2}):(?P<S>\d{2})"
    r"(?:\s+(?P<tz>[+-]\d{4}))?\s*$"
)

def extract_bracket_kvs(line: str) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    n = len(line)
    i = 0
    while i < n:
        if line[i] != "[":
            i += 1
            continue

        j = i + 1
        while j < n and line[j] not in (" ", "]"):
            j += 1
        key = line[i + 1:j]

        if j < n and line[j] == "]":

            out.append((key, None))
            i = j + 1
            continue

        k = j + 1

        while k < n and line[k] == " ":
            k += 1

        if k < n and line[k] == '"':

            k += 1
            buf: List[str] = []
            while k < n:
                c = line[k]
                if c == "\\" and k + 1 < n:

                    buf.append(line[k + 1])
                    k += 2
                    continue
                if c == '"':
                    break
                buf.append(c)
                k += 1
            value: Optional[str] = "".join(buf)

            if k < n and line[k] == '"':
                k += 1
            while k < n and line[k] != "]":
                k += 1
            out.append((key, value))
            i = (k + 1) if k < n else n
            continue

        m = k
        while m < n and line[m] != "]":
            m += 1
        value = line[k:m].strip()
        out.append((key, value if value != "" else None))
        i = (m + 1) if m < n else n
    return out

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

def _split_sections(raw_txn: str) -> "Dict[str, List[str]]":
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in raw_txn.split("\n"):
        m = _BOUNDARY_RE.match(line)
        if m:
            current = m.group(2)
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections

def _parse_section_a(lines: List[str], day: str) -> Tuple[str, str, str]:
    for line in lines:
        if not line.strip():
            continue
        m = _SECTION_A_RE.match(line.strip())
        if m:
            return (
                m.group("uid"),
                _apache_ts_to_iso(m.group("ts")),
                m.group("client_ip"),
            )

        break
    return ("", "", "")

def _parse_request_line(target_line: str) -> Tuple[str, str, str]:
    line = target_line.strip()
    if not line:
        return "", "", ""
    first_sp = line.find(" ")
    if first_sp == -1:

        return line, "", ""
    method = line[:first_sp]
    remainder = line[first_sp + 1:].strip()

    last_sp = remainder.rfind(" ")
    if last_sp != -1:
        tail = remainder[last_sp + 1:]
        if _HTTP_VERSION_RE.match(tail):
            target = remainder[:last_sp].rstrip()
        else:
            target = remainder
    else:
        target = remainder
    if "?" in target:
        path, query = target.split("?", 1)
    else:
        path, query = target, ""
    return method, path, query

def _parse_section_b(lines: List[str]) -> Tuple[str, str, str, str, str]:
    method = path = query = ua = host = ""
    idx = 0
    n = len(lines)

    while idx < n:
        if lines[idx].strip():
            method, path, query = _parse_request_line(lines[idx].strip())
            idx += 1
            break
        idx += 1

    current_header: Optional[str] = None
    while idx < n:
        line = lines[idx]
        idx += 1
        if line == "":
            current_header = None
            continue

        if (line[:1] in (" ", "\t")) and current_header is not None:
            cont = line.strip()
            if current_header == "ua":
                ua = (ua + " " + cont).strip()
            elif current_header == "host":
                host = (host + " " + cont).strip()
            continue
        if ":" not in line:
            current_header = None
            continue
        name, _, value = line.partition(":")
        lname = name.strip().lower()
        value = value.strip()
        if lname == "user-agent":
            ua = value
            current_header = "ua"
        elif lname == "host":
            host = value
            current_header = "host"
        else:
            current_header = None
    return method, path, query, ua, host

def _parse_section_f(lines: List[str]) -> int:
    for line in lines:
        if not line.strip():
            continue
        m = _STATUS_RE.match(line.strip())
        if m:
            return int(m.group(1))

        break
    return 0

def _parse_stopwatch(lines: List[str]) -> int:
    combined: Optional[int] = None
    stopwatch_total: Optional[int] = None
    for line in lines:
        s = line.strip()
        if combined is None and s.startswith("Stopwatch2:"):
            m = re.search(r"combined=(\d+)", s)
            if m:
                combined = int(m.group(1))
        if stopwatch_total is None and s.startswith("Stopwatch:"):

            nums = re.findall(r"\d+", s)
            if len(nums) >= 2:
                stopwatch_total = int(nums[1])
    if combined is not None:
        return combined
    if stopwatch_total is not None:
        return stopwatch_total
    return 0

def _parse_section_h(
    lines: List[str],
) -> Tuple[List[str], List[str], List[str], int]:
    ids: List[str] = []
    tags: List[str] = []
    sevs: List[str] = []
    seen_ids = set()
    seen_tags = set()
    seen_sevs = set()

    for line in lines:
        if not line.startswith("Message:"):
            continue
        for key, value in extract_bracket_kvs(line):
            if value is None:
                continue
            if key == "id":
                if value not in seen_ids:
                    seen_ids.add(value)
                    ids.append(value)
            elif key == "tag":
                if value not in seen_tags:
                    seen_tags.add(value)
                    tags.append(value)
            elif key == "severity":
                if value not in seen_sevs:
                    seen_sevs.add(value)
                    sevs.append(value)

    timing = _parse_stopwatch(lines)
    return ids, tags, sevs, timing

def parse_transaction(
    raw_txn: str, day: str, tag_map: Optional[Dict] = None
) -> Optional[Dict]:
    sections = _split_sections(raw_txn)
    if "A" not in sections and "B" not in sections:
        return None

    unique_id, timestamp, client_ip = _parse_section_a(sections.get("A", []), day)
    method, path, query, ua, host = _parse_section_b(sections.get("B", []))
    status = _parse_section_f(sections.get("F", []))
    crs_rule_ids, crs_tags, severities, timing = _parse_section_h(
        sections.get("H", [])
    )

    if method == "" and unique_id == "":
        return None

    n_rules = len(crs_rule_ids)

    label = "attack"
    attack_subtype = subtype_from_tags(crs_tags, tag_map)

    return {
        "unique_id": unique_id,
        "timestamp": timestamp,
        "day": day,
        "method": method,
        "path": path,
        "query": query,
        "ua": ua,
        "status": status,
        "timing": timing,
        "host": host,
        "client_ip": client_ip,
        "crs_rule_ids": crs_rule_ids,
        "crs_tags": crs_tags,
        "severities": severities,
        "n_rules": n_rules,
        "label": label,
        "attack_subtype": attack_subtype,
    }

def _iter_raw_transactions(text: str) -> Iterator[str]:
    current_hexid: Optional[str] = None
    buf: List[str] = []

    def flush() -> Iterator[str]:
        if buf:
            yield "\n".join(buf)

    for line in text.split("\n"):
        m = _BOUNDARY_RE.match(line)
        if m:
            hexid, section = m.group(1), m.group(2)
            if section == "A" and current_hexid is not None and hexid != current_hexid:

                yield from flush()
                buf = []
            if current_hexid is None or hexid != current_hexid:
                current_hexid = hexid
            buf.append(line)
            if section == "Z":
                yield from flush()
                buf = []
                current_hexid = None
            continue
        buf.append(line)

    yield from flush()

def parse_file(path, day: str) -> Iterator[Dict]:
    from src.data.labels import load_tag_map

    tag_map = load_tag_map()
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        text = fh.read()
    for raw_txn in _iter_raw_transactions(text):
        record = parse_transaction(raw_txn, day, tag_map)
        if record is not None:
            yield record
