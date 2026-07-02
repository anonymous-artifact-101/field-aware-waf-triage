
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from pathlib import Path
from typing import Any, List, Mapping, Optional
from xml.sax.saxutils import escape

from src.detector.field_encoder import resolve_groups
from src.utils.paths import RESULTS_DIR, ensure_dir

__all__ = ["build_architecture_svg", "write_architecture_svg"]

_C_INPUT = "#eef2f7"
_C_PARSE = "#dbe7f3"
_C_EMBED = "#cfe3d4"
_C_CONCAT = "#f6e2c3"
_C_HEAD = "#e7d6ef"
_C_ATTR = "#f3d2d2"
_C_STROKE = "#33424f"
_C_TEXT = "#1b262c"

_ESTIMATOR_LABELS = {
    "linear_svc": "LinearSVC",
    "svc": "LinearSVC",
    "logreg": "LogisticRegression",
    "logistic_regression": "LogisticRegression",
    "logistic": "LogisticRegression",
    "hist_gbdt": "HistGradientBoosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "hgb": "HistGradientBoosting",
}

def _rect(x, y, w, h, fill, label, *, sub: str = "", rx: int = 8) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{_C_STROKE}" stroke-width="1.5"/>'
    ]
    cx = x + w / 2
    if sub:
        parts.append(
            f'<text x="{cx}" y="{y + h / 2 - 4}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="13" '
            f'font-weight="bold" fill="{_C_TEXT}">{escape(label)}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{y + h / 2 + 13}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="10.5" '
            f'fill="{_C_TEXT}">{escape(sub)}</text>'
        )
    else:
        parts.append(
            f'<text x="{cx}" y="{y + h / 2 + 4}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="13" '
            f'font-weight="bold" fill="{_C_TEXT}">{escape(label)}</text>'
        )
    return "".join(parts)

def _arrow(x1, y1, x2, y2) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{_C_STROKE}" '
        f'stroke-width="1.5" marker-end="url(#arrow)"/>'
    )

def _annot(x, y, text: str) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="10.5" '
        f'fill="{_C_TEXT}">{escape(text)}</text>'
    )

def build_architecture_svg(cfg: Mapping[str, Any]) -> str:
    ft = cfg.get("fasttext", {}) if isinstance(cfg, Mapping) else {}
    vector_size = int(ft.get("vector_size", 64))
    train_split = str(ft.get("train_split", "weblog_pretrain"))

    encoder = cfg.get("encoder", {}) if isinstance(cfg, Mapping) else {}
    pooling = str(encoder.get("pooling", "mean"))
    explicit_groups = encoder.get("groups")
    granularity = cfg.get("granularity", 6) if isinstance(cfg, Mapping) else 6

    groups, block_labels = resolve_groups(granularity, groups=explicit_groups)
    num_blocks = len(block_labels)
    feature_dim = num_blocks * vector_size

    detector = cfg.get("detector", {}) if isinstance(cfg, Mapping) else {}
    estimator_key = str(detector.get("estimator", "linear_svc")).lower()
    estimator_label = _ESTIMATOR_LABELS.get(estimator_key, estimator_key)

    width, height = 760, 740
    cx = width / 2
    box_w = 460
    box_x = cx - box_w / 2
    elements: List[str] = []

    elements.append(
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L8,3 L0,6 z" fill="{_C_STROKE}"/></marker></defs>'
    )
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    elements.append(
        f'<text x="{cx}" y="28" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="16" font-weight="bold" fill="{_C_TEXT}">'
        'Field-Aware FastText Detector</text>'
    )

    y = 50

    elements.append(_rect(box_x, y, box_w, 42, _C_INPUT, "Reverse-proxy access-log record",
                          sub="method path query ua status timing (raw line)"))
    bot = y + 42

    y2 = bot + 40
    chip_gap = 8
    chip_w = (box_w - chip_gap * (num_blocks - 1)) / num_blocks
    elements.append(_arrow(cx, bot, cx, y2))
    elements.append(_annot(cx, (bot + y2) / 2 + 4, "parse into typed fields"))
    for i, label in enumerate(block_labels):
        cxx = box_x + i * (chip_w + chip_gap)
        elements.append(
            f'<rect x="{cxx}" y="{y2}" width="{chip_w}" height="34" rx="6" '
            f'fill="{_C_PARSE}" stroke="{_C_STROKE}" stroke-width="1.2"/>'
            f'<text x="{cxx + chip_w / 2}" y="{y2 + 21}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="9.5" '
            f'fill="{_C_TEXT}">{escape(label)}</text>'
        )
    bot = y2 + 34

    y3 = bot + 32
    elements.append(_arrow(cx, bot, cx, y3))
    elements.append(_annot(cx, (bot + y3) / 2 + 4, f"per-field FastText embed + {pooling}-pool"))
    elements.append(_rect(box_x, y3, box_w, 50, _C_EMBED,
                          "Per-Field FastText Embedding + Pool",
                          sub=f"{num_blocks} field block(s), each -> {vector_size}-d vector"))
    bot = y3 + 50

    note_w = 250
    note_x = box_x + box_w + 14
    if note_x + note_w > width:
        note_x = box_x - note_w - 14
    elements.append(
        f'<rect x="{note_x}" y="{y3 - 2}" width="{note_w}" height="54" rx="6" '
        f'fill="none" stroke="{_C_STROKE}" stroke-width="1" stroke-dasharray="4 3"/>'
    )
    elements.append(
        f'<text x="{note_x + note_w / 2}" y="{y3 + 16}" text-anchor="middle" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="{_C_TEXT}">'
        'FastText pre-trained self-supervised</text>'
    )
    elements.append(
        f'<text x="{note_x + note_w / 2}" y="{y3 + 30}" text-anchor="middle" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="{_C_TEXT}">'
        f'on benign corpus ({escape(train_split)});</text>'
    )
    elements.append(
        f'<text x="{note_x + note_w / 2}" y="{y3 + 44}" text-anchor="middle" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="9.5" fill="{_C_TEXT}">'
        'char n-grams embed OOV payloads</text>'
    )

    y4 = bot + 32
    elements.append(_arrow(cx, bot, cx, y4))
    elements.append(_annot(cx, (bot + y4) / 2 + 4, "concatenate per-field vectors"))
    elements.append(_rect(box_x, y4, box_w, 48, _C_CONCAT,
                          "Concatenated Field-Aware Feature",
                          sub=f"D = {num_blocks} x {vector_size} = {feature_dim}"))
    bot = y4 + 48

    y5 = bot + 34
    head_w = 360
    elements.append(_arrow(cx, bot, cx, y5))
    elements.append(_rect(cx - head_w / 2, y5, head_w, 50, _C_HEAD,
                          f"Classical Head ({estimator_label})",
                          sub="subtype prediction (un-weighted, class_weight=None)"))
    bot = y5 + 50

    y6 = bot + 40
    attr_w = 400
    elements.append(_arrow(cx, bot, cx, y6))
    elements.append(_rect(cx - attr_w / 2, y6, attr_w, 46, _C_ATTR,
                          "Per-Field Anomaly Attribution",
                          sub="block-zeroing delta -> which field explains the detection"))

    ly = height - 26
    legend = [("parse", _C_PARSE), ("FastText embed", _C_EMBED),
              ("concat", _C_CONCAT), ("classical head", _C_HEAD),
              ("attribution", _C_ATTR)]
    lx = 16
    for name, color in legend:
        elements.append(
            f'<rect x="{lx}" y="{ly}" width="14" height="14" rx="3" fill="{color}" '
            f'stroke="{_C_STROKE}" stroke-width="1"/>'
            f'<text x="{lx + 20}" y="{ly + 11}" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="10.5" fill="{_C_TEXT}">{escape(name)}</text>'
        )
        lx += 30 + 8 * len(name)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>"
    )
    return svg

def write_architecture_svg(
    cfg: Mapping[str, Any],
    out_dir: Optional[Path] = None,
) -> Path:
    target_dir = ensure_dir(out_dir if out_dir is not None else RESULTS_DIR / "figure_01_architecture")
    out_path = target_dir / "architecture.svg"
    svg = build_architecture_svg(cfg)
    out_path.write_text(svg, encoding="utf-8")
    return out_path

def _self_test() -> None:
    from src.utils.config import load_config

    cfg = load_config("configs/detector/fasttext_base.yaml")
    path = write_architecture_svg(cfg)
    size = path.stat().st_size
    print(f"[architecture_diagram] wrote {path} ({size} bytes)")
    assert size > 1000, "SVG looks suspiciously small"

if __name__ == "__main__":
    _self_test()
