
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.data.tokenizers.special_tokens import (
    CLS,
    FIELD_MASK,
    FIELD_SEP,
    SEP,
    field_sentinel,
)

__all__ = [
    "FieldSpec",
    "Schema",
    "NONFIELD_ID",
    "load_schema",
    "render_field_value",
    "encode_record",
    "decode",
    "decode_fields",
    "build_field_remap",
    "remap_field_ids",
]

NONFIELD_ID = 6

class FieldSpec:

    __slots__ = ("name", "field_id", "kind", "ftype")

    def __init__(self, name: str, field_id: int, kind: str = "text", ftype: "str | None" = None) -> None:
        self.name = name
        self.field_id = int(field_id)
        self.kind = kind
        self.ftype = ftype

    def __repr__(self) -> str:
        return (
            f"FieldSpec(name={self.name!r}, field_id={self.field_id}, "
            f"kind={self.kind!r}, ftype={self.ftype!r})"
        )

class Schema:

    __slots__ = ("fields", "nonfield_id")

    def __init__(self, fields: Iterable[FieldSpec]) -> None:
        ordered = sorted(fields, key=lambda f: f.field_id)

        for expected, spec in enumerate(ordered):
            if spec.field_id != expected:
                raise ValueError(
                    f"field_ids must be contiguous 0..N-1 in order; "
                    f"got field {spec.name!r} with id {spec.field_id} at position {expected}"
                )
        self.fields: tuple[FieldSpec, ...] = tuple(ordered)
        self.nonfield_id: int = len(self.fields)

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def __iter__(self):
        return iter(self.fields)

def load_schema(schema_obj: Mapping[str, Any]) -> Schema:
    if isinstance(schema_obj, Mapping) and "fields" in schema_obj:
        entries = schema_obj["fields"]
    else:
        entries = schema_obj
    specs = [
        FieldSpec(
            name=e["name"],
            field_id=e["field_id"],
            kind=e.get("kind", "text"),
            ftype=e.get("type"),
        )
        for e in entries
    ]
    return Schema(specs)

def render_field_value(value: Any, kind: str) -> str:
    if kind == "numeric":
        if value is None:
            return "0"
        return str(int(value))
    if value is None:
        return ""
    return str(value)

def encode_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    schema: Schema,
    *,
    mask_fields: "set[str] | frozenset[str] | None" = None,
    add_cls: bool = True,
    add_sep: bool = True,
) -> dict[str, list[int]]:
    mask_fields = mask_fields or frozenset()
    nonfield = schema.nonfield_id

    cls_id = _token_id(tokenizer, CLS)
    sep_id = _token_id(tokenizer, SEP)
    field_sep_id = _token_id(tokenizer, FIELD_SEP)
    field_mask_id = _token_id(tokenizer, FIELD_MASK)

    input_ids: list[int] = []
    field_ids: list[int] = []

    if add_cls:
        input_ids.append(cls_id)
        field_ids.append(nonfield)

    for position, spec in enumerate(schema.fields):
        fid = spec.field_id

        if position > 0:
            input_ids.append(field_sep_id)
            field_ids.append(fid)

        input_ids.append(_token_id(tokenizer, field_sentinel(spec.name)))
        field_ids.append(fid)

        if spec.name in mask_fields:

            input_ids.append(field_mask_id)
            field_ids.append(fid)
            continue

        rendered = render_field_value(record.get(spec.name), spec.kind)
        if rendered == "":

            continue
        enc = tokenizer.encode(rendered, add_special_tokens=False)
        ids = enc.ids
        input_ids.extend(ids)
        field_ids.extend([fid] * len(ids))

    if add_sep:
        input_ids.append(sep_id)
        field_ids.append(nonfield)

    assert len(input_ids) == len(field_ids), "input_ids/field_ids length mismatch"
    return {"input_ids": input_ids, "field_ids": field_ids}

def decode(tokenizer: Any, input_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
    return tokenizer.decode(list(input_ids), skip_special_tokens=skip_special_tokens)

def decode_fields(
    tokenizer: Any,
    input_ids: Iterable[int],
    field_ids: Iterable[int],
    schema: Schema,
) -> dict[str, str]:
    input_ids = list(input_ids)
    field_ids = list(field_ids)
    if len(input_ids) != len(field_ids):
        raise ValueError("input_ids and field_ids must be the same length")

    id_to_name = {f.field_id: f.name for f in schema.fields}
    buckets: dict[int, list[int]] = {}
    for tid, fid in zip(input_ids, field_ids):
        buckets.setdefault(fid, []).append(tid)

    out: dict[str, str] = {}
    for fid, ids in buckets.items():
        key = id_to_name.get(fid, "<nonfield>")
        out[key] = tokenizer.decode(ids, skip_special_tokens=True)
    return out

def build_field_remap(
    *,
    source_num_fields: int = 6,
    target_num_fields: int,
    groups: "Sequence[Sequence[int]] | None" = None,
) -> list[int]:
    source_num_fields = int(source_num_fields)
    target_num_fields = int(target_num_fields)
    if source_num_fields < 1:
        raise ValueError(f"source_num_fields must be >= 1, got {source_num_fields}")
    if not (1 <= target_num_fields <= source_num_fields):
        raise ValueError(
            f"target_num_fields must be in [1, {source_num_fields}], got {target_num_fields}"
        )

    if groups is None:
        if target_num_fields == source_num_fields:
            groups = [[i] for i in range(source_num_fields)]
        elif target_num_fields == 1:
            groups = [list(range(source_num_fields))]
        else:
            raise ValueError(
                "an explicit `groups` partition is required when "
                f"target_num_fields ({target_num_fields}) is neither 1 nor "
                f"source_num_fields ({source_num_fields})"
            )

    groups = [list(g) for g in groups]
    if len(groups) != target_num_fields:
        raise ValueError(
            f"expected {target_num_fields} groups, got {len(groups)}: {groups}"
        )

    remap: list[int] = [-1] * (source_num_fields + 1)
    remap[source_num_fields] = target_num_fields

    seen: set[int] = set()
    for target_id, members in enumerate(groups):
        for src in members:
            src = int(src)
            if not (0 <= src < source_num_fields):
                raise ValueError(
                    f"group member {src} out of range "
                    f"[0, {source_num_fields}) in groups {groups}"
                )
            if src in seen:
                raise ValueError(f"source field-id {src} appears in more than one group")
            seen.add(src)
            remap[src] = target_id

    missing = [i for i in range(source_num_fields) if i not in seen]
    if missing:
        raise ValueError(
            f"groups must cover every source field-id 0..{source_num_fields - 1}; "
            f"missing {missing}"
        )
    return remap

def remap_field_ids(field_ids: Any, remap: "Sequence[int]") -> Any:
    import torch

    if not isinstance(field_ids, torch.Tensor):
        field_ids = torch.as_tensor(field_ids, dtype=torch.long)
    table = torch.as_tensor(list(remap), dtype=torch.long, device=field_ids.device)
    flat = field_ids.reshape(-1).to(torch.long)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= table.numel()):
        raise IndexError(
            f"field id out of range for remap table of size {table.numel()}: "
            f"min={int(flat.min())} max={int(flat.max())}"
        )
    return table[flat].reshape(field_ids.shape)

def _token_id(tokenizer: Any, token: str) -> int:
    tid = tokenizer.token_to_id(token)
    if tid is None:
        raise KeyError(
            f"Special token {token!r} is not in the tokenizer vocabulary; "
            f"was the tokenizer trained with src.data.tokenizers.special_tokens.SPECIAL_TOKENS?"
        )
    return tid
