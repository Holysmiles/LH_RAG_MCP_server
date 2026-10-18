"""Finalize a trace and pass its JSON snapshot to the persistence boundary."""

from collections.abc import Callable
from typing import Any

from core.trace.trace_context import TraceContext


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
