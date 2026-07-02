
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.baselines._common import (
    SUBTYPES,
    git_commit,
    load_split_records,
    macro_f1,
    per_class_f1,
    render_records,
    select_label_budget,
    subtype_labels,
    write_results_json,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "CharCNN", "encode_char_batch"]

_LOGGER = get_logger("baseline.char_cnn")

def encode_char_batch(
    texts: Sequence[str], vocab_size: int, max_length: int
) -> torch.Tensor:
    out = torch.zeros((len(texts), max_length), dtype=torch.long)
    for i, t in enumerate(texts):
        b = t.encode("utf-8", errors="replace")[:max_length]
        for j, byte in enumerate(b):
            out[i, j] = 1 + (byte % (vocab_size - 1))
    return out

class CharCNN(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        conv_channels: int,
        kernel_sizes: Sequence[int],
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, conv_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv_channels * len(kernel_sizes), num_classes)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(char_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            pooled.append(h.max(dim=2).values)
        feat = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(feat)

def _train_cnn(
    model: CharCNN,
    char_ids: torch.Tensor,
    labels: torch.Tensor,
    train_cfg: Mapping[str, Any],
    device: torch.device,
) -> float:
    epochs = int(train_cfg.get("epochs", 10))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    num_classes = int(model.classifier.out_features)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.where(counts > 0, counts.sum() / (counts * num_classes), torch.ones(()))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    n = char_ids.shape[0]
    final_loss = float("nan")
    for _ in range(epochs):
        perm = torch.arange(n)
        for start in range(0, n, batch_size):
            sel = perm[start : start + batch_size]
            logits = model(char_ids[sel].to(device))
            loss = criterion(logits, labels[sel].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
    return final_loss

@torch.no_grad()
def _predict(model: CharCNN, char_ids: torch.Tensor, device: torch.device, batch_size: int = 256) -> List[int]:
    model.eval()
    preds: List[int] = []
    for start in range(0, char_ids.shape[0], batch_size):
        logits = model(char_ids[start : start + batch_size].to(device))
        preds.extend(int(p) for p in logits.argmax(dim=-1).cpu().tolist())
    return preds

def run(
    cfg: Mapping[str, Any],
    seed: int = 42,
    *,
    max_records: Optional[int] = None,
    write: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")
    data = cfg.get("data", {})
    features = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    budget = float(data.get("label_budget", 0.05))

    vocab_size = int(features.get("vocab_size", 128))
    max_length = int(features.get("max_length", 256))

    train_recs = load_split_records(
        str(data.get("train_split", "owasp_train")), max_records=max_records
    )
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    idx = select_label_budget(len(train_recs), budget)
    if not idx:
        raise ValueError(f"label_budget={budget} selected 0 of {len(train_recs)} train records")
    labeled = [train_recs[i] for i in idx]

    model = CharCNN(
        vocab_size=vocab_size,
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        conv_channels=int(model_cfg.get("conv_channels", 128)),
        kernel_sizes=[int(k) for k in model_cfg.get("kernel_sizes", [3, 5, 7])],
        num_classes=int(model_cfg.get("num_classes", len(SUBTYPES))),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(dev)

    train_ids = encode_char_batch(render_records(labeled), vocab_size, max_length)
    train_labels = torch.tensor(subtype_labels(labeled), dtype=torch.long)
    final_loss = _train_cnn(model, train_ids, train_labels, train_cfg, dev)

    test_ids = encode_char_batch(render_records(test_recs), vocab_size, max_length)
    test_labels = subtype_labels(test_recs)
    preds = _predict(model, test_ids, dev)
    value = round(100.0 * macro_f1(preds, test_labels, len(SUBTYPES)), 4)
    pcf1 = dict(zip(SUBTYPES, per_class_f1(preds, test_labels, len(SUBTYPES))))

    _LOGGER.info(
        "char_cnn: budget=%.3f (%d/%d labeled), test n=%d, final_loss=%.4f macro_f1=%.4f",
        budget, len(labeled), len(train_recs), len(test_recs), final_loss, value,
    )

    col = f"{int(round(budget * 100))}%"
    extra = {
        "metric": "macro_f1",
        "label_budget": budget,
        "n_train_labeled": len(labeled),
        "n_test": len(test_recs),
        "final_train_loss": round(final_loss, 6),
        "params": int(sum(p.numel() for p in model.parameters())),
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="Char-CNN", col=col, value=value, seed=seed,
            extra_metadata=extra, per_class=pcf1,
        )
        _LOGGER.info("wrote %s", out_path)

        if max_records is None:
            from src.eval.predictions_io import dump_predictions, dump_shared_labels

            table_dir = out_path.parent
            split = str(data.get("test_split", "owasp_test"))
            dump_shared_labels(
                table_dir, split=split, labels=test_labels,
                subtypes=list(SUBTYPES), num_classes=len(SUBTYPES),
                metadata={"commit": git_commit()},
            )
            dump_predictions(
                table_dir, row="Char-CNN", col=col, seed=int(seed), split=split,
                preds=preds, labels=test_labels, num_classes=len(SUBTYPES),
                metric="macro_f1", value=value,
                metadata={"commit": git_commit(), "deterministic": False,
                          "baseline": "char_cnn"},
            )

    return {"value": value, "per_class_f1": pcf1, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records, render_records as _rr, subtype_labels as _sl

    set_seed(42)
    recs = synthetic_records(80, seed=4)
    model = CharCNN(128, 32, 32, [3, 5], len(SUBTYPES), 0.1)
    ids = encode_char_batch(_rr(recs), 128, 128)
    labels = torch.tensor(_sl(recs), dtype=torch.long)
    loss = _train_cnn(model, ids, labels, {"epochs": 3, "batch_size": 16, "lr": 1e-3}, torch.device("cpu"))
    preds = _predict(model, ids, torch.device("cpu"))
    print(f"[char_cnn] smoke: fit ok ({sum(p.numel() for p in model.parameters())} params), "
          f"final_loss={round(loss, 4)}, train macro_f1={round(macro_f1(preds, _sl(recs)), 3)}")

if __name__ == "__main__":
    _smoke()
