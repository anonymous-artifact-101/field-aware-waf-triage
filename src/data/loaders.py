
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.labels import SUBTYPES
from src.data.tokenizers.field_aware import build_field_remap
from src.utils.paths import SPLITS_DIR, TOKENIZED_DIR

__all__ = [
    "TokenizedDataset",
    "LabeledTokenizedDataset",
    "collate_batch",
    "make_collate_fn",
    "subtype_to_index",
    "field_remap_from_config",
]

_SUBTYPE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(SUBTYPES)}

def subtype_to_index(subtype: str) -> int:
    try:
        return _SUBTYPE_TO_INDEX[subtype]
    except KeyError as exc:
        raise KeyError(
            f"unknown attack_subtype {subtype!r}; known subtypes are {SUBTYPES}"
        ) from exc

def field_remap_from_config(
    cfg: Mapping[str, Any],
    *,
    source_num_fields: int = 6,
) -> Optional[List[int]]:
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, Mapping) else {}
    target = int(model_cfg.get("num_fields", source_num_fields))
    if target == source_num_fields:
        return None

    remap_cfg = cfg.get("field_remap") if isinstance(cfg, Mapping) else None
    groups = None
    if isinstance(remap_cfg, Mapping):
        groups = remap_cfg.get("groups")
        declared_src = remap_cfg.get("source_num_fields")
        if declared_src is not None:
            source_num_fields = int(declared_src)
    return build_field_remap(
        source_num_fields=source_num_fields,
        target_num_fields=target,
        groups=groups,
    )

def _read_meta(tokenized_dir: Path, split_name: str) -> Dict[str, Any]:
    meta_path = tokenized_dir / f"{split_name}.meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"tokenized meta not found: {meta_path}. Run scripts/04_tokenize_all.py first."
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

class TokenizedDataset(Dataset):

    def __init__(
        self,
        split_name: str,
        tokenized_dir: "str | Path | None" = None,
        *,
        max_length: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.split_name = str(split_name)
        self.tokenized_dir = Path(tokenized_dir) if tokenized_dir is not None else TOKENIZED_DIR
        self.max_length = int(max_length) if max_length is not None else None

        self.meta = _read_meta(self.tokenized_dir, self.split_name)
        self.n_records = int(self.meta["n_records"])

        self._full_n_records = self.n_records
        if limit is not None and int(limit) > 0:
            self.n_records = min(self.n_records, int(limit))

        dt = self.meta["dtypes"]
        self._input_dtype = np.dtype(dt["input_ids"])
        self._field_dtype = np.dtype(dt["field_ids"])
        self._offset_dtype = np.dtype(dt["offsets"])

        files = self.meta["files"]
        self._input_path = self.tokenized_dir / files["input_ids"]
        self._field_path = self.tokenized_dir / files["field_ids"]
        self._offset_path = self.tokenized_dir / files["offsets"]

        schema = self.meta.get("schema", {})
        self.num_fields = int(schema.get("num_fields", 6))
        self.nonfield_id = int(schema.get("nonfield_id", self.num_fields))
        self.field_names: List[str] = list(schema.get("field_names", []))

        self.pad_token_id = int(self.meta.get("special_token_ids", {}).get("[PAD]", 0))

        self._input_ids: Optional[np.memmap] = None
        self._field_ids: Optional[np.memmap] = None
        self._offsets: Optional[np.memmap] = None

    def _ensure_open(self) -> None:
        if self._offsets is None:
            self._offsets = np.memmap(self._offset_path, dtype=self._offset_dtype, mode="r")
            self._input_ids = np.memmap(self._input_path, dtype=self._input_dtype, mode="r")
            self._field_ids = np.memmap(self._field_path, dtype=self._field_dtype, mode="r")

    def __len__(self) -> int:
        return self.n_records

    def _record_span(self, index: int) -> "tuple[int, int]":
        self._ensure_open()
        assert self._offsets is not None
        if index < 0:
            index += self.n_records
        if not (0 <= index < self.n_records):
            raise IndexError(f"record index {index} out of range [0, {self.n_records})")
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return start, end

    def raw_record(self, index: int) -> "tuple[np.ndarray, np.ndarray]":
        self._ensure_open()
        assert self._input_ids is not None and self._field_ids is not None
        start, end = self._record_span(index)
        if self.max_length is not None and (end - start) > self.max_length:
            end = start + self.max_length

        ids = np.asarray(self._input_ids[start:end], dtype=np.int64)
        fids = np.asarray(self._field_ids[start:end], dtype=np.int64)
        return ids, fids

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ids, fids = self.raw_record(index)
        return {
            "input_ids": torch.from_numpy(ids),
            "field_ids": torch.from_numpy(fids),
        }

class LabeledTokenizedDataset(TokenizedDataset):

    def __init__(
        self,
        split_name: str,
        tokenized_dir: "str | Path | None" = None,
        *,
        max_length: Optional[int] = None,
        limit: Optional[int] = None,
        labels_jsonl: "str | Path | None" = None,
        label_field: str = "attack_subtype",
    ) -> None:
        super().__init__(split_name, tokenized_dir, max_length=max_length, limit=limit)
        self.label_field = str(label_field)
        if labels_jsonl is not None:
            self._labels_path = Path(labels_jsonl)
        else:
            self._labels_path = SPLITS_DIR / f"{self.split_name}.jsonl"
        self.labels: np.ndarray = self._load_labels()

    def _load_labels(self) -> np.ndarray:
        if not self._labels_path.is_file():
            raise FileNotFoundError(
                f"labels JSONL not found: {self._labels_path}. "
                "LabeledTokenizedDataset needs the matching split JSONL for attack_subtype."
            )
        labels: List[int] = []
        with self._labels_path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                rec = json.loads(stripped)
                subtype = rec.get(self.label_field)
                if subtype is None:
                    raise ValueError(
                        f"record on line {line_no} of {self._labels_path} has no "
                        f"{self.label_field!r} field"
                    )
                labels.append(subtype_to_index(str(subtype)))

        if len(labels) != self._full_n_records:
            raise ValueError(
                f"label/record alignment broken: {self._labels_path} has {len(labels)} "
                f"labeled records but the tokenized split has {self._full_n_records}. "
                "The tokenizer must preserve JSONL record order."
            )
        arr = np.asarray(labels, dtype=np.int64)
        return arr[: self.n_records]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        if index < 0:
            index += self.n_records
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item

def collate_batch(
    batch: Sequence[Mapping[str, torch.Tensor]],
    *,
    pad_token_id: int = 0,
    nonfield_id: int = 6,
    max_length: int = 512,
    field_remap: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("collate_batch received an empty batch")

    lengths = [int(item["input_ids"].shape[0]) for item in batch]
    target_len = min(max(lengths), int(max_length))

    pad_field_id = int(field_remap[int(nonfield_id)]) if field_remap is not None else int(nonfield_id)

    b = len(batch)
    input_ids = torch.full((b, target_len), int(pad_token_id), dtype=torch.long)
    field_ids = torch.full((b, target_len), pad_field_id, dtype=torch.long)
    attention_mask = torch.zeros((b, target_len), dtype=torch.long)

    table = (
        torch.as_tensor(list(field_remap), dtype=torch.long)
        if field_remap is not None
        else None
    )

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        fids = item["field_ids"].to(torch.long)
        n = min(int(ids.shape[0]), target_len)
        if table is not None and n:
            src = fids[:n]
            if int(src.min()) < 0 or int(src.max()) >= table.numel():
                raise IndexError(
                    f"field id out of range for remap table of size {table.numel()}: "
                    f"min={int(src.min())} max={int(src.max())}"
                )
            fids = table[src]
            n = min(n, int(fids.shape[0]))
        input_ids[i, :n] = ids[:n].to(torch.long)
        field_ids[i, :n] = fids[:n]
        attention_mask[i, :n] = 1

    out: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "field_ids": field_ids,
        "attention_mask": attention_mask,
    }

    if all("labels" in item for item in batch):
        out["labels"] = torch.stack([item["labels"].to(torch.long) for item in batch], dim=0)
    return out

def make_collate_fn(
    *,
    pad_token_id: int = 0,
    nonfield_id: int = 6,
    max_length: int = 512,
    field_remap: Optional[Sequence[int]] = None,
) -> Callable[[Sequence[Mapping[str, torch.Tensor]]], Dict[str, torch.Tensor]]:

    def _collate(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return collate_batch(
            batch,
            pad_token_id=pad_token_id,
            nonfield_id=nonfield_id,
            max_length=max_length,
            field_remap=field_remap,
        )

    return _collate

def _self_test() -> None:
    from torch.utils.data import DataLoader

    ds = TokenizedDataset("owasp_train")
    print(f"[loaders] TokenizedDataset(owasp_train): {len(ds)} records; "
          f"num_fields={ds.num_fields}, nonfield_id={ds.nonfield_id}, "
          f"pad_token_id={ds.pad_token_id}")
    rec = ds[0]
    print(f"[loaders] record 0: input_ids {tuple(rec['input_ids'].shape)}, "
          f"field_ids {tuple(rec['field_ids'].shape)}; "
          f"first ids {rec['input_ids'][:6].tolist()}")

    collate = make_collate_fn(
        pad_token_id=ds.pad_token_id, nonfield_id=ds.nonfield_id, max_length=512
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    print(f"[loaders] collated batch: input_ids {tuple(batch['input_ids'].shape)}, "
          f"field_ids {tuple(batch['field_ids'].shape)}, "
          f"attention_mask sum per row {batch['attention_mask'].sum(dim=1).tolist()}")

    lds = LabeledTokenizedDataset("owasp_val")
    litem = lds[0]
    print(f"[loaders] LabeledTokenizedDataset(owasp_val): label[0]={int(litem['labels'])} "
          f"({SUBTYPES[int(litem['labels'])]}); label dtype {litem['labels'].dtype}")

if __name__ == "__main__":
    _self_test()
