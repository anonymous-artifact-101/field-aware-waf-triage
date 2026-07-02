
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.baselines._common import load_split_records, write_results_json
from src.baselines._logkey import (
    build_key_vocab,
    encode_log_keys,
    group_sessions,
    sliding_windows,
)
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "DeepLogLSTM", "train_deeplog", "anomaly_rate"]

_LOGGER = get_logger("baseline.deeplog")

class DeepLogLSTM(nn.Module):

    def __init__(self, num_keys: int, hidden_size: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_keys = int(num_keys)
        self.embedding = nn.Embedding(num_keys, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.embedding(windows)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def _make_pairs(
    records: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, int],
    *,
    scheme: str,
    session_by: str,
    window_size: int,
) -> List[Tuple[List[int], int]]:
    key_ids = encode_log_keys(records, vocab, scheme=scheme)
    sessions = group_sessions(records, key_ids, session_by=session_by)
    pairs: List[Tuple[List[int], int]] = []
    for s in sessions:
        pairs.extend(sliding_windows(s, window_size))
    return pairs

def train_deeplog(
    model: DeepLogLSTM,
    pairs: Sequence[Tuple[List[int], int]],
    train_cfg: Mapping[str, Any],
    device: torch.device,
) -> float:
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 128))
    lr = float(train_cfg.get("lr", 1e-3))

    if not pairs:
        return float("nan")
    hist = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = hist.shape[0]
    final_loss = float("nan")
    for _ in range(epochs):
        for start in range(0, n, batch_size):
            hb = hist[start : start + batch_size].to(device)
            tb = tgt[start : start + batch_size].to(device)
            logits = model(hb)
            loss = criterion(logits, tb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
    return final_loss

@torch.no_grad()
def anomaly_rate(
    model: DeepLogLSTM,
    pairs: Sequence[Tuple[List[int], int]],
    device: torch.device,
    *,
    top_k: int,
    batch_size: int = 256,
) -> float:
    if not pairs:
        return float("nan")
    model.eval()
    hist = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    anomalies = 0
    total = 0
    for start in range(0, hist.shape[0], batch_size):
        hb = hist[start : start + batch_size].to(device)
        tb = tgt[start : start + batch_size].to(device)
        logits = model(hb)
        k = min(int(top_k), logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        in_topk = (topk == tb.unsqueeze(1)).any(dim=1)
        anomalies += int((~in_topk).sum().item())
        total += int(tb.shape[0])
    return anomalies / total if total else float("nan")

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
    seq = cfg.get("sequence", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    scheme = str(seq.get("log_key", "method_path_status"))
    num_keys = int(seq.get("num_keys", 512))
    window_size = int(seq.get("window_size", 10))
    session_by = str(seq.get("session_by", "source_ip"))
    top_k = int(seq.get("top_k", 9))

    train_recs = load_split_records(
        str(data.get("train_split", "owasp_train")), max_records=max_records
    )
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    vocab = build_key_vocab(train_recs, scheme=scheme, num_keys=num_keys)
    train_pairs = _make_pairs(
        train_recs, vocab, scheme=scheme, session_by=session_by, window_size=window_size
    )
    test_pairs = _make_pairs(
        test_recs, vocab, scheme=scheme, session_by=session_by, window_size=window_size
    )

    model = DeepLogLSTM(
        num_keys=num_keys,
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(dev)

    final_loss = train_deeplog(model, train_pairs, train_cfg, dev)
    rate = anomaly_rate(model, test_pairs, dev, top_k=top_k)
    value = round(100.0 * float(rate), 4) if rate == rate else float("nan")

    _LOGGER.info(
        "deeplog (RE-TRAINED): |vocab|=%d train_pairs=%d test_pairs=%d final_loss=%.4f anomaly_rate=%.4f%%",
        len(vocab), len(train_pairs), len(test_pairs), final_loss, value,
    )

    extra = {
        "metric": "anomaly_rate_pct",
        "retrained_on_corpus": True,
        "log_key_scheme": scheme,
        "key_vocab_size": len(vocab),
        "top_k": top_k,
        "window_size": window_size,
        "n_train_windows": len(train_pairs),
        "n_test_windows": len(test_pairs),
        "final_train_loss": round(final_loss, 6) if final_loss == final_loss else None,
        "note": "Unsupervised next-key detector; cell = test anomaly rate (true key "
        "outside top-k). OWASP is all-attack so in-corpus attack/benign AUC is degenerate.",
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="DeepLog (re-trained)", col="0%", value=value, seed=seed, extra_metadata=extra
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "anomaly_rate": rate, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records

    set_seed(42)
    recs = synthetic_records(120, seed=7)
    vocab = build_key_vocab(recs, num_keys=32)
    pairs = _make_pairs(recs, vocab, scheme="method_path_status", session_by="source_ip", window_size=5)
    model = DeepLogLSTM(32, 16, 1, 0.0)
    loss = train_deeplog(model, pairs, {"epochs": 2, "batch_size": 16, "lr": 1e-2}, torch.device("cpu"))
    rate = anomaly_rate(model, pairs, torch.device("cpu"), top_k=3)
    print(f"[deeplog] smoke: |vocab|={len(vocab)}, pairs={len(pairs)}, "
          f"final_loss={round(loss, 4)}, anomaly_rate={round(rate, 4)}")

if __name__ == "__main__":
    _smoke()
