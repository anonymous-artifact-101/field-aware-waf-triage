
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "log_key_string",
    "build_key_vocab",
    "encode_log_keys",
    "group_sessions",
    "sliding_windows",
]

_HEX_OR_NUM = re.compile(r"^[0-9a-fA-F_]{6,}$|^\d+$")

def _path_shape(path: str) -> str:
    segments = [s for s in str(path).split("/") if s != ""]
    shaped = []
    for seg in segments:
        if _HEX_OR_NUM.match(seg):
            shaped.append("<id>")
        else:
            shaped.append(seg)
    return "/" + "/".join(shaped)

def _status_bucket(status: Any) -> str:
    try:
        code = int(status)
    except (TypeError, ValueError):
        return "x"
    return f"{code // 100}xx"

def log_key_string(record: Mapping[str, Any], scheme: str = "method_path_status") -> str:
    method = str(record.get("method", ""))
    if scheme == "method_path_status":
        return f"{method} {_path_shape(record.get('path', ''))} {_status_bucket(record.get('status'))}"
    if scheme == "method_path":
        return f"{method} {_path_shape(record.get('path', ''))}"
    if scheme == "path_status":
        return f"{_path_shape(record.get('path', ''))} {_status_bucket(record.get('status'))}"

    return f"{method} {_path_shape(record.get('path', ''))} {_status_bucket(record.get('status'))}"

def build_key_vocab(
    records: Sequence[Mapping[str, Any]],
    *,
    scheme: str = "method_path_status",
    num_keys: int = 512,
) -> Dict[str, int]:
    from collections import Counter

    counts: Counter = Counter(log_key_string(r, scheme) for r in records)
    vocab: Dict[str, int] = {}
    for key, _ in counts.most_common(max(int(num_keys) - 1, 1)):
        vocab[key] = len(vocab) + 1
    return vocab

def encode_log_keys(
    records: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, int],
    *,
    scheme: str = "method_path_status",
) -> List[int]:
    return [int(vocab.get(log_key_string(r, scheme), 0)) for r in records]

def group_sessions(
    records: Sequence[Mapping[str, Any]],
    key_ids: Sequence[int],
    *,
    session_by: str = "source_ip",
) -> List[List[int]]:
    field = "client_ip" if session_by in ("source_ip", "client_ip") else session_by
    sessions: Dict[str, List[int]] = {}
    order: List[str] = []
    for rec, kid in zip(records, key_ids):
        sid = str(rec.get(field, "__global__")) if field else "__global__"
        if sid not in sessions:
            sessions[sid] = []
            order.append(sid)
        sessions[sid].append(int(kid))
    return [sessions[sid] for sid in order]

def sliding_windows(
    session: Sequence[int], window_size: int
) -> List[Tuple[List[int], int]]:
    w = int(window_size)
    pairs: List[Tuple[List[int], int]] = []
    seq = list(session)
    if len(seq) < 2:
        return pairs
    for i in range(1, len(seq)):
        start = max(0, i - w)
        history = seq[start:i]
        if len(history) < w:
            history = [0] * (w - len(history)) + history
        pairs.append((history, seq[i]))
    return pairs
