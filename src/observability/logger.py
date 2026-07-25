"""Minimal stderr logging used during application startup."""

from __future__ import annotations

import logging
import sys


_HANDLER_MARKER = "_modular_rag_stderr_handler"


def get_logger(name: str, level: str | int = logging.INFO) -> logging.Logger:
    """Return a named logger configured with one stderr handler."""
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    handler = next(
        (
            item
            for item in logger.handlers
            if getattr(item, _HANDLER_MARKER, False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    else:
        handler.stream = sys.stderr

    handler.setLevel(logger.level)
    return logger


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown logging level: {level}")
    return resolved
