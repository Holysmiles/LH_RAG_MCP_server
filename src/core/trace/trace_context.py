"""Request lifecycle and JSON-safe stage snapshots with monotonic timing."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from numbers import Real
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4


class TraceContext:
    """保存一次请求中按发生顺序记录的阶段数据。

    Inputs:
        无必填参数；创建时自动生成全局唯一 ``trace_id``。

    Outputs:
        ``stages`` 是按调用顺序保存的字典列表，每项至少包含 ``stage`` 名称。
    """

    def __init__(self, trace_type: str = "query") -> None:
        if not isinstance(trace_type, str) or trace_type not in {"query", "ingestion"}:
            raise ValueError("trace_type must be 'query' or 'ingestion'")
        self.trace_id = str(uuid4())
        self.trace_type = trace_type
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: str | None = None
        self._started = perf_counter()
        self._total_elapsed_ms: float | None = None
        self._lock = RLock()
        self.stages: list[dict[str, Any]] = []

    def __deepcopy__(self, memo: dict[int, Any]) -> TraceContext:
        # Reranker isolates trace writes in a daemon worker using deepcopy.
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        with self._lock:
            for key, value in self.__dict__.items():
                if key != "_lock":
                    setattr(copied, key, deepcopy(value, memo))
        copied._lock = RLock()
        return copied

    def record_stage(
        self,
        stage_name: str,
        details: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """追加一个阶段快照，不保留调用方可变字典的引用。"""
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string")
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("details must be a mapping or None")

        stage: dict[str, Any] = {}
        if details is not None:
            stage.update(deepcopy(dict(details)))
        stage.update(deepcopy(fields))
        # 调用方可以自由记录业务字段，但不能把阶段身份改成另一个名称。
        stage["stage"] = stage_name.strip()
        for field in ("elapsed_ms", "duration_ms"):
            if field in stage:
                value = stage[field]
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{field} must be a non-negative finite number")
                stage[field] = float(value)
        # Reject invalid payloads before appending, rather than losing a whole
        # completed trace later when a writer tries to serialize it.
        json.dumps(stage, ensure_ascii=False, allow_nan=False)
        with self._lock:
            if self.finished_at is not None:
                raise RuntimeError("Cannot record stages after trace.finish()")
            self.stages.append(stage)

    def finish(self) -> None:
        """Finish once; repeated calls preserve the original end and duration."""
        with self._lock:
            if self.finished_at is None:
                self._total_elapsed_ms = (perf_counter() - self._started) * 1000
                self.finished_at = datetime.now(timezone.utc).isoformat()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return total duration or sum measured durations for matching stages.

        Untimed legacy stages contribute zero; no duration is guessed from the
        order of callbacks, which may come from concurrent retrieval routes.
        """
        with self._lock:
            if stage_name is None:
                return (self._total_elapsed_ms if self._total_elapsed_ms is not None
                        else (perf_counter() - self._started) * 1000)
            if not isinstance(stage_name, str) or not stage_name.strip():
                raise ValueError("stage_name must be a non-empty string")
            return sum((float(stage.get("elapsed_ms", stage.get("duration_ms", 0)))
                        for stage in self.stages if stage["stage"] == stage_name.strip()), 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON snapshot without ending an active trace."""
        with self._lock:
            payload = {
                "trace_id": self.trace_id, "trace_type": self.trace_type,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "total_elapsed_ms": self.elapsed_ms(), "stages": deepcopy(self.stages),
            }
        # Normalize tuples to JSON arrays and ensure even legacy direct writes
        # to the public stages list cannot silently emit invalid JSON.
        return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


__all__ = ["TraceContext"]
