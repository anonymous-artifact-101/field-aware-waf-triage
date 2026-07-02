
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from src.baselines._common import detection_summary, load_split_records, write_results_json
from src.baselines._logkey import build_key_vocab, encode_log_keys, group_sessions
from src.utils.logging_setup import get_logger
from src.utils.seeds import set_seed

__all__ = ["run", "LogBERT", "train_logbert"]

_LOGGER = get_logger("baseline.logbert")

class LogBERT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dist_id = vocab_size - 1
        self.mask_id = vocab_size - 2
        self.token_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.mlkm_head = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, key_ids: torch.Tensor) -> Dict[str, torch.Tensor]:

        b, t = key_ids.shape
        dist = torch.full((b, 1), self.dist_id, dtype=torch.long, device=key_ids.device)
        x = torch.cat([dist, key_ids], dim=1)
        positions = torch.arange(x.shape[1], device=key_ids.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        pad_mask = x == 0
        pad_mask[:, 0] = False
        enc = self.encoder(h, src_key_padding_mask=pad_mask)
        return {"sequence": enc, "dist": enc[:, 0, :], "tokens": enc[:, 1:, :]}

def _sessions_to_windows(
    records: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, int],
    *,
    scheme: str,
    session_by: str,
    window_size: int,
) -> List[List[int]]:
    key_ids = encode_log_keys(records, vocab, scheme=scheme)
    sessions = group_sessions(records, key_ids, session_by=session_by)
    windows: List[List[int]] = []
    for s in sessions:
        if not s:
            continue
        for start in range(0, len(s), window_size):
            chunk = s[start : start + window_size]
            if len(chunk) < window_size:
                chunk = chunk + [0] * (window_size - len(chunk))
            windows.append(chunk)
    return windows

def train_logbert(
    model: LogBERT,
    windows: Sequence[Sequence[int]],
    pretrain_cfg: Mapping[str, Any],
    device: torch.device,
) -> "tuple[float, torch.Tensor]":
    steps = int(pretrain_cfg.get("steps", 50000))
    batch_size = int(pretrain_cfg.get("batch_size", 64))
    lr = float(pretrain_cfg.get("lr_peak", 1e-4))
    mask_rate = float(pretrain_cfg.get("mask_rate", 0.15))
    hs_weight = float(pretrain_cfg.get("hypersphere_weight", 0.1))

    if not windows:
        return float("nan"), torch.zeros(model.hidden_size, device=device)

    data = torch.tensor([list(w) for w in windows], dtype=torch.long)
    n = data.shape[0]

    model.eval()
    with torch.no_grad():
        center = model(data[: min(n, 512)].to(device))["dist"].mean(dim=0)

    mlkm_loss = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    final_loss = float("nan")
    step = 0
    pos = 0
    while step < steps:
        if pos >= n:
            pos = 0
        batch = data[pos : pos + batch_size].to(device)
        pos += batch_size

        inp = batch.clone()
        labels = torch.full_like(batch, -100)
        prob = torch.rand_like(batch, dtype=torch.float)
        nonpad = batch != 0
        do_mask = (prob < mask_rate) & nonpad

        need = nonpad.any(dim=1) & (~do_mask.any(dim=1))
        if bool(need.any()):
            first_nonpad = nonpad.float().argmax(dim=1)
            rows = torch.nonzero(need, as_tuple=False).squeeze(1)
            do_mask[rows, first_nonpad[rows]] = True
        labels[do_mask] = batch[do_mask]
        inp[do_mask] = model.mask_id

        out = model(inp)
        logits = model.mlkm_head(out["tokens"])
        loss_mlkm = mlkm_loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss_hs = ((out["dist"] - center) ** 2).sum(dim=1).mean()
        loss = loss_mlkm + hs_weight * loss_hs

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        step += 1

    model.eval()
    with torch.no_grad():
        center = model(data[: min(n, 512)].to(device))["dist"].mean(dim=0)
    return final_loss, center

@torch.no_grad()
def _anomaly_scores(
    model: LogBERT,
    windows: Sequence[Sequence[int]],
    center: torch.Tensor,
    device: torch.device,
    *,
    mask_rate: float,
    batch_size: int = 128,
) -> List[float]:
    if not windows:
        return []
    model.eval()
    data = torch.tensor([list(w) for w in windows], dtype=torch.long)
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    scores: List[float] = []
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size].to(device)
        inp = batch.clone()
        labels = torch.full_like(batch, -100)

        stride = max(int(round(1.0 / max(mask_rate, 1e-3))), 1)
        nonpad = batch != 0
        idx = torch.zeros_like(batch, dtype=torch.bool)
        idx[:, ::stride] = True
        do_mask = idx & nonpad
        labels[do_mask] = batch[do_mask]
        inp[do_mask] = model.mask_id

        out = model(inp)
        logits = model.mlkm_head(out["tokens"])
        per_tok = ce(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).reshape(batch.shape)
        denom = do_mask.sum(dim=1).clamp_min(1).float()
        recon = (per_tok * do_mask.float()).sum(dim=1) / denom
        dist = ((out["dist"] - center.to(device)) ** 2).sum(dim=1)
        scores.extend((recon + dist).cpu().tolist())
    return scores

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
    pre = cfg.get("pretrain", {})

    scheme = str(seq.get("log_key", "method_path_status"))
    num_keys = int(seq.get("num_keys", 512))
    window_size = int(seq.get("window_size", 128))
    session_by = str(seq.get("session_by", "source_ip"))
    mask_rate = float(pre.get("mask_rate", 0.15))

    train_recs = load_split_records(
        str(data.get("train_split", "owasp_train")), max_records=max_records
    )
    test_recs = load_split_records(
        str(data.get("test_split", "owasp_test")), max_records=max_records
    )

    vocab = build_key_vocab(train_recs, scheme=scheme, num_keys=num_keys)
    vocab_size = max(len(vocab) + 1, num_keys) + 2

    train_windows = _sessions_to_windows(
        train_recs, vocab, scheme=scheme, session_by=session_by, window_size=window_size
    )
    test_windows = _sessions_to_windows(
        test_recs, vocab, scheme=scheme, session_by=session_by, window_size=window_size
    )

    model = LogBERT(
        vocab_size=vocab_size,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_hidden_layers", 4)),
        num_heads=int(model_cfg.get("num_attention_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 1024)),
        max_len=window_size,
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(dev)

    final_loss, center = train_logbert(model, train_windows, pre, dev)
    scores = _anomaly_scores(model, test_windows, center, dev, mask_rate=mask_rate)
    summary = detection_summary(scores, [1] * len(scores))
    value = round(float(summary["mean_score"]), 6)

    _LOGGER.info(
        "logbert (RE-TRAINED): |vocab|=%d train_win=%d test_win=%d final_loss=%.4f mean_score=%.6f",
        vocab_size, len(train_windows), len(test_windows), final_loss, value,
    )

    extra = {
        "metric": "mean_anomaly_score",
        "retrained_on_corpus": True,
        "log_key_scheme": scheme,
        "key_vocab_size": len(vocab),
        "model_vocab_size": vocab_size,
        "window_size": window_size,
        "n_train_windows": len(train_windows),
        "n_test_windows": len(test_windows),
        "final_train_loss": round(final_loss, 6) if final_loss == final_loss else None,
        "score_summary": {k: round(float(v), 6) for k, v in summary.items()},
        "note": "Unsupervised MLKM + hypersphere detector; cell = mean per-window "
        "anomaly score. OWASP is all-attack so in-corpus attack/benign AUC is degenerate.",
    }

    out_path = None
    if write:
        out_path = write_results_json(
            cfg, row="LogBERT (re-trained)", col="0%", value=value, seed=seed, extra_metadata=extra
        )
        _LOGGER.info("wrote %s", out_path)

    return {"value": value, "summary": summary, "results_path": str(out_path) if out_path else None}

def _smoke() -> None:
    from src.baselines._common import synthetic_records

    set_seed(42)
    recs = synthetic_records(120, seed=8)
    vocab = build_key_vocab(recs, num_keys=24)
    vsize = len(vocab) + 1 + 2
    windows = _sessions_to_windows(recs, vocab, scheme="method_path_status", session_by="source_ip", window_size=8)
    model = LogBERT(vsize, 32, 2, 2, 64, max_len=8, dropout=0.0)
    loss, center = train_logbert(model, windows, {"steps": 5, "batch_size": 8, "lr_peak": 1e-3, "mask_rate": 0.2}, torch.device("cpu"))
    scores = _anomaly_scores(model, windows, center, torch.device("cpu"), mask_rate=0.2)
    print(f"[logbert] smoke: |vocab|={vsize}, windows={len(windows)}, "
          f"final_loss={round(loss, 4)}, n_scores={len(scores)}, mean={round(sum(scores)/max(len(scores),1), 4)}")

if __name__ == "__main__":
    _smoke()
