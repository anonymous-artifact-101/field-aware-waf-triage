
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

__all__ = ["paired_bootstrap_macro_f1", "paired_block_bootstrap_macro_f1"]

def _macro_f1_per_resample(codes, idx_chunk, num_classes):
    import numpy as np

    C = int(num_classes)
    b = idx_chunk.shape[0]
    gathered = codes[idx_chunk]

    offset = (np.arange(b, dtype=np.int64)[:, None] * (C * C)) + gathered
    conf = np.bincount(offset.ravel(), minlength=b * C * C).reshape(b, C, C)

    tp = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    support = conf.sum(axis=2).astype(np.float64)
    predicted = conf.sum(axis=1).astype(np.float64)
    fp = predicted - tp
    fn = support - tp
    denom = 2.0 * tp + fp + fn

    f1 = np.where(denom > 0.0, (2.0 * tp) / denom, 0.0)
    return f1.mean(axis=1)

def paired_bootstrap_macro_f1(
    preds_a: Sequence[int],
    preds_b: Sequence[int],
    labels: Sequence[int],
    num_classes: int = 8,
    *,
    B: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
    as_percent: bool = True,
    chunk: int = 500,
    name_a: str = "a",
    name_b: str = "b",
) -> Dict[str, Any]:
    import numpy as np

    from src.baselines._common import macro_f1 as _macro_f1_point

    ya = np.asarray([int(p) for p in preds_a], dtype=np.int64)
    yb = np.asarray([int(p) for p in preds_b], dtype=np.int64)
    t = np.asarray([int(y) for y in labels], dtype=np.int64)
    n = t.shape[0]
    if not (ya.shape[0] == yb.shape[0] == n):
        raise ValueError(
            f"length mismatch: preds_a={ya.shape[0]} preds_b={yb.shape[0]} "
            f"labels={n} must all be equal (index-aligned test records)."
        )
    if n == 0:
        raise ValueError("empty label vector; nothing to bootstrap.")
    C = int(num_classes)

    scale = 100.0 if as_percent else 1.0
    point_a = scale * float(_macro_f1_point(ya.tolist(), t.tolist(), C))
    point_b = scale * float(_macro_f1_point(yb.tolist(), t.tolist(), C))

    code_a = (t * C + ya).astype(np.int64)
    code_b = (t * C + yb).astype(np.int64)

    rng = np.random.default_rng(int(seed))
    boot_a = np.empty(int(B), dtype=np.float64)
    boot_b = np.empty(int(B), dtype=np.float64)
    done = 0
    while done < int(B):
        b = min(int(chunk), int(B) - done)
        idx_chunk = rng.integers(0, n, size=(b, n))
        boot_a[done:done + b] = _macro_f1_per_resample(code_a, idx_chunk, C)
        boot_b[done:done + b] = _macro_f1_per_resample(code_b, idx_chunk, C)
        done += b
    boot_a *= scale
    boot_b *= scale
    boot_delta = boot_a - boot_b

    alpha = (1.0 - float(confidence)) / 2.0

    def _ci(arr):
        return [
            float(np.percentile(arr, 100.0 * alpha)),
            float(np.percentile(arr, 100.0 * (1.0 - alpha))),
        ]

    ci_a = _ci(boot_a)
    ci_b = _ci(boot_b)
    delta_ci = _ci(boot_delta)
    delta = point_a - point_b

    frac_le0 = float(np.mean(boot_delta <= 0.0))
    p_two_sided = min(1.0, 2.0 * min(frac_le0, 1.0 - frac_le0))
    p_value = max(p_two_sided, 1.0 / float(B))

    significant = bool(delta_ci[0] > 0.0 or delta_ci[1] < 0.0)

    return {
        "name_a": str(name_a),
        "name_b": str(name_b),
        "n": int(n),
        "num_classes": C,
        "B": int(B),
        "seed": int(seed),
        "confidence": float(confidence),
        "macro_f1_a": round(point_a, 4),
        "macro_f1_b": round(point_b, 4),
        "ci_a": [round(ci_a[0], 4), round(ci_a[1], 4)],
        "ci_b": [round(ci_b[0], 4), round(ci_b[1], 4)],
        "delta": round(delta, 4),
        "delta_ci": [round(delta_ci[0], 4), round(delta_ci[1], 4)],
        "delta_boot_mean": round(float(boot_delta.mean()), 4),
        "p_value": round(p_value, 6),
        "significant": significant,
        "zero_support_convention": (
            "count absent/zero-TP class as F1=0 over a fixed denominator of "
            f"{C} classes (matches src.baselines._common.macro_f1)"
        ),
    }

def _confusion_for_indices(preds, labels, indices, num_classes):
    import numpy as np

    C = int(num_classes)
    conf = np.zeros((C, C), dtype=np.int64)
    for idx in indices:
        y = int(labels[int(idx)])
        p = int(preds[int(idx)])
        if 0 <= y < C and 0 <= p < C:
            conf[y, p] += 1
    return conf

def _macro_from_conf_batch(conf):
    import numpy as np

    tp = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    support = conf.sum(axis=2).astype(np.float64)
    predicted = conf.sum(axis=1).astype(np.float64)
    fp = predicted - tp
    fn = support - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=np.float64)
    np.divide(2.0 * tp, denom, out=f1, where=denom > 0.0)
    return 100.0 * f1.mean(axis=1)

def paired_block_bootstrap_macro_f1(
    preds_a: Sequence[int],
    preds_b: Sequence[int],
    labels: Sequence[int],
    blocks: Sequence[Sequence[int]],
    num_classes: int = 8,
    *,
    B: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
    chunk: int = 1000,
    name_a: str = "a",
    name_b: str = "b",
    block_description: str = "temporal blocks",
) -> Dict[str, Any]:
    import numpy as np

    from src.baselines._common import macro_f1 as _macro_f1_point

    ya = np.asarray([int(p) for p in preds_a], dtype=np.int64)
    yb = np.asarray([int(p) for p in preds_b], dtype=np.int64)
    t = np.asarray([int(y) for y in labels], dtype=np.int64)
    n = t.shape[0]
    if not (ya.shape[0] == yb.shape[0] == n):
        raise ValueError(
            f"length mismatch: preds_a={ya.shape[0]} preds_b={yb.shape[0]} "
            f"labels={n} must all be equal (index-aligned test records)."
        )
    if not blocks:
        raise ValueError("empty block list; nothing to bootstrap.")

    C = int(num_classes)
    normalized_blocks = [[int(i) for i in block] for block in blocks if block]
    if not normalized_blocks:
        raise ValueError("all blocks are empty; nothing to bootstrap.")
    flat = sorted(i for block in normalized_blocks for i in block)
    if flat != list(range(n)):
        raise ValueError(
            "blocks must form an exact, non-overlapping cover of the prediction "
            f"indices 0..{n - 1}; got {len(flat)} covered entries."
        )

    point_a = 100.0 * float(_macro_f1_point(ya.tolist(), t.tolist(), C))
    point_b = 100.0 * float(_macro_f1_point(yb.tolist(), t.tolist(), C))

    block_conf_a = np.stack([
        _confusion_for_indices(ya, t, block, C) for block in normalized_blocks
    ])
    block_conf_b = np.stack([
        _confusion_for_indices(yb, t, block, C) for block in normalized_blocks
    ])
    n_blocks = block_conf_a.shape[0]
    block_sizes = np.asarray([len(block) for block in normalized_blocks], dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    boot_a = np.empty(int(B), dtype=np.float64)
    boot_b = np.empty(int(B), dtype=np.float64)
    boot_n = np.empty(int(B), dtype=np.int64)
    done = 0
    while done < int(B):
        b = min(int(chunk), int(B) - done)
        draws = rng.integers(0, n_blocks, size=(b, n_blocks))
        conf_a = block_conf_a[draws].sum(axis=1)
        conf_b = block_conf_b[draws].sum(axis=1)
        boot_a[done:done + b] = _macro_from_conf_batch(conf_a)
        boot_b[done:done + b] = _macro_from_conf_batch(conf_b)
        boot_n[done:done + b] = block_sizes[draws].sum(axis=1)
        done += b
    boot_delta = boot_a - boot_b

    alpha = (1.0 - float(confidence)) / 2.0

    def _ci(arr):
        return [
            float(np.percentile(arr, 100.0 * alpha)),
            float(np.percentile(arr, 100.0 * (1.0 - alpha))),
        ]

    ci_a = _ci(boot_a)
    ci_b = _ci(boot_b)
    delta_ci = _ci(boot_delta)
    delta = point_a - point_b
    frac_le0 = float(np.mean(boot_delta <= 0.0))
    p_two_sided = min(1.0, 2.0 * min(frac_le0, 1.0 - frac_le0))
    p_value = max(p_two_sided, 1.0 / float(B))

    return {
        "name_a": str(name_a),
        "name_b": str(name_b),
        "n": int(n),
        "num_classes": C,
        "B": int(B),
        "seed": int(seed),
        "confidence": float(confidence),
        "bootstrap_unit": "temporal_cluster_block",
        "block_description": str(block_description),
        "n_blocks": int(n_blocks),
        "block_size_min": int(block_sizes.min()),
        "block_size_median": float(np.median(block_sizes)),
        "block_size_max": int(block_sizes.max()),
        "bootstrap_n_mean": round(float(boot_n.mean()), 2),
        "macro_f1_a": round(point_a, 4),
        "macro_f1_b": round(point_b, 4),
        "ci_a": [round(ci_a[0], 4), round(ci_a[1], 4)],
        "ci_b": [round(ci_b[0], 4), round(ci_b[1], 4)],
        "delta": round(delta, 4),
        "delta_ci": [round(delta_ci[0], 4), round(delta_ci[1], 4)],
        "delta_boot_mean": round(float(boot_delta.mean()), 4),
        "p_value": round(p_value, 6),
        "significant": bool(delta_ci[0] > 0.0 or delta_ci[1] < 0.0),
        "zero_support_convention": (
            "count absent/zero-TP class as F1=0 over a fixed denominator of "
            f"{C} classes (matches src.baselines._common.macro_f1)"
        ),
    }

def _self_test() -> None:
    import numpy as np

    from src.baselines._common import macro_f1 as _macro_f1_point

    rng = np.random.default_rng(0)
    n, C = 2000, 8
    labels = rng.integers(0, C, size=n).tolist()

    preds_a = [(y if rng.random() < 0.85 else (y + 1) % C) for y in labels]
    preds_b = [(y if rng.random() < 0.55 else (y + 2) % C) for y in labels]
    r1 = paired_bootstrap_macro_f1(preds_a, preds_b, labels, C, B=500, seed=42)
    r2 = paired_bootstrap_macro_f1(preds_a, preds_b, labels, C, B=500, seed=42)
    assert r1 == r2, "bootstrap is not deterministic given a fixed seed"

    assert abs(r1["macro_f1_a"] - round(100.0 * _macro_f1_point(preds_a, labels, C), 4)) < 1e-9
    assert abs(r1["macro_f1_b"] - round(100.0 * _macro_f1_point(preds_b, labels, C), 4)) < 1e-9

    assert r1["delta"] > 0.0 and r1["delta_ci"][0] > 0.0 and r1["significant"]

    r3 = paired_bootstrap_macro_f1(preds_a, preds_a, labels, C, B=500, seed=42)
    assert r3["delta"] == 0.0 and r3["delta_ci"][0] <= 0.0 <= r3["delta_ci"][1]
    assert not r3["significant"]

    print("[significance] self-test OK:", {
        "delta": r1["delta"], "delta_ci": r1["delta_ci"],
        "p_value": r1["p_value"], "significant": r1["significant"],
    })

if __name__ == "__main__":
    _self_test()
