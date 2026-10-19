"""Finalize a trace and pass its JSON snapshot to the persistence boundary."""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from core.trace.trace_context import TraceContext

if TYPE_CHECKING:
    from core.settings import Settings


class TraceCollector:
    """The writer owns persistence (F2 supplies the JSONL implementation).

    Writer errors propagate so callers can decide how to handle logging failure;
    the finished context is preserved and can be collected again for a retry.
    """

    def __init__(self, writer: Callable[[dict[str, Any]], None]) -> None:
        if not callable(writer):
            raise TypeError("writer must be callable")
        self._writer = writer

    def collect(self, trace: TraceContext) -> None:
        if not isinstance(trace, TraceContext):
            raise TypeError("trace must be a TraceContext")
        trace.finish()
        self._writer(trace.to_dict())


@contextmanager
def request_trace(settings: "Settings", *, trace_type: str, force: bool = False):
    """Own a request's trace lifecycle; persistence failure cannot replace results.

    ``force`` enables an in-memory trace for CLI verbose output even when file
    tracing is disabled. The writer still honors the configured trace switch.
    """
    from observability.logger import get_logger, write_trace

    trace = TraceContext(trace_type) if force or settings.observability.get("trace_enabled", True) else None
    try:
        yield trace
    finally:
        if trace is not None:
            try:
                TraceCollector(lambda payload: write_trace(payload, settings)).collect(trace)
            except Exception as exc:
                get_logger(__name__).warning("Cannot persist %s trace: %s", trace_type, type(exc).__name__)


def query_trace(settings: "Settings", *, force: bool = False):
    return request_trace(settings, trace_type="query", force=force)


def ingestion_trace(settings: "Settings"):
    return request_trace(settings, trace_type="ingestion")
