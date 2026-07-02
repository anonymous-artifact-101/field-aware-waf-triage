
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from src.utils.paths import LOGS_DIR, ensure_dir

__all__ = ["get_logger", "DEFAULT_FORMAT", "DEFAULT_LEVEL"]

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO

_HANDLER_TAG = "_pecti_managed"

_INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

def _safe_filename(name: str) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", name).strip("_")
    return f"{cleaned or 'app'}.log"

def _has_managed_handler(
    logger: logging.Logger, handler_type: type[logging.Handler]
) -> bool:
    return any(
        getattr(handler, _HANDLER_TAG, False) and isinstance(handler, handler_type)
        for handler in logger.handlers
    )

def get_logger(
    name: str,
    *,
    level: int = DEFAULT_LEVEL,
    log_dir: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    if not _has_managed_handler(logger, logging.StreamHandler):
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(level)
        console.setFormatter(formatter)
        setattr(console, _HANDLER_TAG, True)
        logger.addHandler(console)

    if not _has_managed_handler(logger, logging.FileHandler):
        target_dir = ensure_dir(log_dir if log_dir is not None else LOGS_DIR)
        file_path = target_dir / _safe_filename(name)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)

    return logger
