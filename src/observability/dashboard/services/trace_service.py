"""Read persisted trace snapshots without creating active TraceContext objects."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from core.settings import Settings


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    trace_type: str
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    stages: list[dict[str, Any]]

    @property
    def status(self) -> str:
        statuses = [stage.get("details", {}).get("status") for stage in self.stages]
        if "failed" in statuses or any(stage.get("error_type") for stage in self.stages):
            return "failed"
        if "skipped" in statuses:
            return "skipped"
        if not self.finished_at:
            return "running"
        return "completed" if "completed" in statuses else "unknown"

    def detail(self, key: str) -> str:
        for stage in self.stages:
            value = stage.get("details", {}).get(key)
            if isinstance(value, str) and value:
                return value
        return ""


class TraceService:
    def __init__(self, path: str | Path = "logs/traces.jsonl") -> None:
        self.path = Path(path).expanduser()
        self.invalid_line_count = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> "TraceService":
        return cls(settings.observability.get("trace_file", "logs/traces.jsonl"))

    @staticmethod
    def _timestamp(value: str) -> datetime:
        if not isinstance(value, str):
            raise ValueError("timestamp must be a string")
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    @staticmethod
    def _duration(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("duration must be finite and nonnegative")
        return float(value)

    def load_traces(self, trace_type: str | None = None) -> list[TraceRecord]:
        """Skip corrupt or partial lines, counting them for an actionable UI notice."""
        if trace_type is not None and trace_type not in {"query", "ingestion"}:
            raise ValueError("unsupported trace type")
        self.invalid_line_count = 0
        try:
            source = self.path.open(encoding="utf-8")
        except FileNotFoundError:
            return []
        records = []
        with source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("trace must be an object")
                    identity, kind = payload.get("trace_id"), payload.get("trace_type")
                    if not isinstance(identity, str) or not identity.strip() or kind not in {"query", "ingestion"}:
                        raise ValueError("invalid trace identity")
                    self._timestamp(payload["started_at"])
                    finished = payload.get("finished_at")
                    if finished is not None:
                        self._timestamp(finished)
                    duration = self._duration(payload.get("total_elapsed_ms", 0))
                    stages = payload.get("stages")
                    if not isinstance(stages, list):
                        raise ValueError("stages must be an array")
                    for stage in stages:
                        if (not isinstance(stage, dict) or not isinstance(stage.get("stage"), str)
                                or not isinstance(stage.get("details", {}), dict)):
                            raise ValueError("invalid stage")
                        for key in ("elapsed_ms", "duration_ms"):
                            if key in stage:
                                self._duration(stage[key])
                    record = TraceRecord(identity, kind, payload["started_at"], finished, duration, stages)
                    if trace_type is None or kind == trace_type:
                        records.append(record)
                except (ValueError, KeyError, TypeError, OverflowError):
                    self.invalid_line_count += 1
        return sorted(records, key=lambda row: self._timestamp(row.started_at), reverse=True)

    @staticmethod
    def ingestion_timings(trace: TraceRecord) -> list[dict[str, Any]]:
        """Only top-level sequential phases; nested encoder timings overlap them."""
        rows = []
        for name in ("integrity", "load", "split", "transform", "embed", "upsert"):
            stages = [stage for stage in trace.stages if stage["stage"] == name]
            if not stages:
                continue
            measured = [stage for stage in stages if "elapsed_ms" in stage or "duration_ms" in stage]
            if not measured:
                continue
            rows.append({"stage": name, "elapsed_ms": sum(stage.get("elapsed_ms", stage.get("duration_ms", 0))
                                                          for stage in measured),
                         "status": "failed" if any(stage.get("details", {}).get("status") == "failed"
                                                   for stage in stages) else stages[-1].get("details", {}).get("status", "unknown")})
        return rows
