"""Apply optional reranking with a bounded wait and fusion-order fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, TimeoutError
from copy import deepcopy
import math
from numbers import Real
from threading import Lock, Thread
from time import perf_counter
from typing import ClassVar

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import RerankerFactory


class Reranker:
    """Return original candidates in backend order, or retain fusion order.

    ``rerank.timeout_seconds`` bounds backend construction and inference. A
    timed-out Python call cannot be killed; its daemon worker finishes in the
    background, and this instance falls back while that worker remains busy.
    Candidate and trace copies isolate late writes from the caller. Each call
    records ``fallback`` in the supplied trace's ``rerank`` stage.
    """

    default_timeout_seconds: ClassVar[float] = 10.0

    def __init__(
        self, settings: Settings, backend: BaseReranker | None = None
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if not isinstance(settings.rerank, Mapping):
            raise TypeError("settings.rerank must be a mapping")
        if backend is not None and not isinstance(backend, BaseReranker):
            raise TypeError("backend must implement BaseReranker or be None")
        self.settings = deepcopy(settings)
        config = self.settings.rerank
        self.enabled = config.get("enabled", True)
        if not isinstance(self.enabled, bool):
            raise ValueError("rerank.enabled must be a boolean")
        provider = config.get("provider", "none")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("rerank.provider must be a non-empty string")
        self.provider = provider.strip().lower()
        timeout = config.get("timeout_seconds", self.default_timeout_seconds)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("rerank.timeout_seconds must be a positive finite number")
        self.timeout_seconds = float(timeout)
        self._backend = backend
        self._worker_lock = Lock()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Rerank all candidates before optional Top-K truncation.

        Invalid caller inputs raise; backend construction, inference, timeout,
        or output-contract failures return the untouched fusion ranking.
        Scores and citation metadata retain their original fusion values.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError("candidates must be a non-string sequence")
        original = list(candidates)
        if any(not isinstance(item, RetrievalResult) for item in original):
            raise TypeError("candidates must contain RetrievalResult objects")
        if len({item.chunk_id for item in original}) != len(original):
            raise ValueError("candidates must have unique chunk_id values")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        if trace is not None and not isinstance(trace, TraceContext):
            raise TypeError("trace must be a TraceContext or None")

        started = perf_counter()
        ranked = original
        fallback = False
        reason: str | None = None
        error_type: str | None = None
        if not self.enabled or self.provider == "none":
            reason = "disabled"
        elif not original:
            reason = "empty_candidates"
        elif not self._worker_lock.acquire(blocking=False):
            fallback, reason = True, "backend_busy"
        else:
            try:
                try:
                    isolated = deepcopy(original)
                    worker_trace = deepcopy(trace)
                except Exception:
                    self._worker_lock.release()
                    raise
                positions = {id(item): index for index, item in enumerate(isolated)}
                trace_offset = len(worker_trace.stages) if worker_trace else 0
                future: Future[list[RetrievalResult]] = Future()

                def invoke() -> None:
                    try:
                        try:
                            if self._backend is None:
                                self._backend = RerankerFactory.create(self.settings)
                            result = self._backend.rerank(
                                query.strip(), isolated, trace=worker_trace
                            )
                        finally:
                            self._worker_lock.release()
                    except Exception as exc:
                        future.set_exception(exc)
                    else:
                        future.set_result(result)

                worker = Thread(target=invoke, name="rag-reranker", daemon=True)
                try:
                    worker.start()
                except Exception:
                    self._worker_lock.release()
                    raise
                output = future.result(timeout=self.timeout_seconds)
                if not isinstance(output, list) or len(output) != len(original):
                    raise ValueError("backend must return every candidate exactly once")
                if {id(item) for item in output} != set(positions):
                    raise ValueError("backend must return the original candidate objects")
                ranked = [original[positions[id(item)]] for item in output]
                if trace is not None and worker_trace is not None:
                    trace.stages.extend(deepcopy(worker_trace.stages[trace_offset:]))
            except TimeoutError:
                fallback, reason, error_type = True, "timeout", "TimeoutError"
            except Exception as exc:
                fallback, reason, error_type = True, "backend_error", type(exc).__name__

        selected = ranked if top_k is None else ranked[:top_k]
        if trace is not None:
            trace.record_stage(
                "rerank",
                provider=self.provider,
                method=self.provider if self.enabled else "none",
                details={"details": {"input_count": len(original), "output_count": len(selected)}},
                fallback=fallback,
                reason=reason,
                error_type=error_type,
                input_count=len(original),
                output_count=len(selected),
                input_ids=[item.chunk_id for item in original],
                output_ids=[item.chunk_id for item in ranked],
                selected_ids=[item.chunk_id for item in selected],
                elapsed_ms=(perf_counter() - started) * 1000,
            )
        return list(selected)


__all__ = ["Reranker"]
