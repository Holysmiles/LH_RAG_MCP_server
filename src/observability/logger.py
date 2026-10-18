"""Application stderr logging and independent UTF-8 JSON Lines trace files."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from os.path import normcase
from pathlib import Path
import sys
from threading import RLock
from typing import Any

from core.settings import Settings


_HANDLER_MARKER = "_modular_rag_stderr_handler"
_TRACE_HANDLER_MARKER = "_modular_rag_trace_handler"
_TRACE_LOGGER_LOCK = RLock()
DEFAULT_TRACE_PATH = Path("logs/traces.jsonl")


class JSONFormatter(logging.Formatter):
    """Keep trace mappings at the top level; format normal records as JSON too."""

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, Mapping):
            payload = dict(record.msg)
        else:
            payload = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname, "logger": record.name,
                "message": record.getMessage(),
            }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


class _TraceFileHandler(logging.FileHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        # StreamHandler normally swallows write/flush failures. The collector
        # contract requires these to reach the caller so a write can be retried.
        raise


def _trace_configuration(settings: Settings | None, path: str | Path | None) -> tuple[Path, bool]:
    config = settings.observability if settings is not None else {}
    enabled = config.get("trace_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("observability.trace_enabled must be a boolean")
    value = path if path is not None else config.get("trace_file", DEFAULT_TRACE_PATH)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("observability.trace_file must be a non-empty path")
    return Path(value).expanduser().resolve(), enabled


def get_trace_logger(settings: Settings | None = None, *, path: str | Path | None = None) -> logging.Logger:
    """Return one file-only logger per resolved path; repeated calls deduplicate.

    Explicit ``path`` overrides ``observability.trace_file``. Relative paths use
    the process working directory, as do the project's other storage settings.
    Disabled tracing creates no directories or files.
    """
    target, enabled = _trace_configuration(settings, path)
    name = "observability.trace." + sha256(normcase(str(target)).encode("utf-8")).hexdigest()
    with _TRACE_LOGGER_LOCK:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.disabled = not enabled
        logger.setLevel(logging.INFO)
        if enabled:
            target.parent.mkdir(parents=True, exist_ok=True)
        if not any(getattr(handler, _TRACE_HANDLER_MARKER, False) for handler in logger.handlers):
            handler = _TraceFileHandler(target, mode="a", encoding="utf-8", delay=True)
            setattr(handler, _TRACE_HANDLER_MARKER, True)
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)
        return logger


def write_trace(
    trace_dict: Mapping[str, Any], settings: Settings | None = None, *, path: str | Path | None = None,
) -> None:
    """Append exactly one JSON line, or do nothing when tracing is disabled."""
    _, enabled = _trace_configuration(settings, path)
    if not enabled:
        return
    if not isinstance(trace_dict, Mapping):
        raise TypeError("trace_dict must be a mapping")
    trace_type = trace_dict.get("trace_type")
    if not isinstance(trace_type, str) or trace_type not in {"query", "ingestion"}:
        raise ValueError("trace_dict.trace_type must be 'query' or 'ingestion'")
    # Validate and copy before opening a file, so invalid data cannot leave a
    # partial line or invoke logging's stderr error fallback.
    snapshot = json.loads(json.dumps(dict(trace_dict), ensure_ascii=False, allow_nan=False))
    get_trace_logger(settings, path=path).info(snapshot)


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
