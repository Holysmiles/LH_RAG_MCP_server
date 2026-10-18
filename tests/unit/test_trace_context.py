"""Lifecycle, deterministic timing, JSON snapshots and collector delivery."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import json

import pytest

import core.trace.trace_context as trace_module
from core.trace import TraceCollector, TraceContext


pytestmark = pytest.mark.unit


@pytest.fixture
def clock(monkeypatch):
    current = [10.0]
    monkeypatch.setattr(trace_module, "perf_counter", lambda: current[0])
    return current


def test_finish_freezes_monotonic_total_and_utc_timestamps(clock):
    trace = TraceContext("ingestion")
    clock[0] = 10.5
    assert trace.elapsed_ms() == pytest.approx(500)
    assert trace.to_dict()["finished_at"] is None
    clock[0] = 11.25
    trace.finish()
    first = trace.to_dict()
    clock[0] = 99
    trace.finish()
    assert trace.to_dict() == first
    assert first["total_elapsed_ms"] == pytest.approx(1250)
    assert first["trace_type"] == "ingestion"
    assert set(first) == {"trace_id", "trace_type", "started_at", "finished_at", "total_elapsed_ms", "stages"}
    assert datetime.fromisoformat(first["started_at"]).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(first["finished_at"]) >= datetime.fromisoformat(first["started_at"])
    assert json.loads(json.dumps(first)) == first


def test_stage_durations_sum_aliases_and_leave_untimed_stages_unchanged():
    trace = TraceContext()
    trace.record_stage("batch", {"duration_ms": 3, "stage": "ignored"})
    trace.record_stage("batch", elapsed_ms=4, duration_ms=40)
    trace.record_stage("legacy", count=1)
    assert trace.trace_type == "query"
    assert trace.elapsed_ms("batch") == 7.0
    assert trace.elapsed_ms("legacy") == 0.0
    assert trace.elapsed_ms("absent") == 0.0
    assert isinstance(trace.elapsed_ms("absent"), float)
    assert trace.stages[-1] == {"stage": "legacy", "count": 1}


def test_stage_snapshots_and_json_exports_do_not_alias_mutable_data():
    details = {"items": ["a"]}
    trace = TraceContext()
    trace.record_stage("stage", details, extra=("x", "y"))
    details["items"].append("b")
    snapshot = trace.to_dict()
    assert snapshot["stages"][0]["extra"] == ["x", "y"]
    snapshot["stages"][0]["items"].append("c")
    assert trace.stages[0]["items"] == ["a"]


def test_deepcopy_supports_reranker_worker_isolation(clock):
    trace = TraceContext()
    trace.record_stage("fusion", results=[1])
    isolated = deepcopy(trace)
    assert isolated.trace_id == trace.trace_id
    isolated.stages[0]["results"].append(2)
    isolated.record_stage("backend", elapsed_ms=5)
    isolated.finish()
    assert trace.stages == [{"stage": "fusion", "results": [1]}]
    assert trace.finished_at is None
    trace.record_stage("rerank", elapsed_ms=1)


def test_concurrent_stage_records_are_not_lost():
    trace = TraceContext()
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda index: trace.record_stage("worker", index=index), range(40)))
    trace.finish()
    assert sorted(stage["index"] for stage in trace.to_dict()["stages"]) == list(range(40))
    with pytest.raises(RuntimeError, match="after"):
        trace.record_stage("late")


@pytest.mark.parametrize("trace_type", ["other", "", None, []])
def test_rejects_unknown_trace_types(trace_type):
    with pytest.raises(ValueError, match="trace_type"):
        TraceContext(trace_type)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True, "5"])
def test_rejects_invalid_duration_without_appending(duration):
    trace = TraceContext()
    with pytest.raises(ValueError, match="elapsed_ms"):
        trace.record_stage("bad", elapsed_ms=duration)
    assert trace.stages == []


def test_rejects_non_json_stage_and_invalid_stage_name():
    trace = TraceContext()
    with pytest.raises(TypeError):
        trace.record_stage("bad", value=object())
    with pytest.raises(ValueError):
        trace.record_stage("bad", value=float("nan"))
    with pytest.raises(ValueError):
        trace.elapsed_ms(" ")
    assert trace.stages == []


def test_collector_finishes_trace_and_delivers_json_to_writer(tmp_path):
    path = tmp_path / "traces.jsonl"

    def writer(payload):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        payload["stages"].clear()

    trace = TraceContext("ingestion")
    trace.record_stage("load", document="中文.pdf", elapsed_ms=5)
    TraceCollector(writer).collect(trace)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["trace_id"] == trace.trace_id
    assert stored["trace_type"] == "ingestion"
    assert stored["finished_at"] is not None
    assert stored["stages"][0]["document"] == "中文.pdf"
    assert len(trace.stages) == 1


def test_collector_writer_failure_can_be_retried_without_changing_end_time(clock):
    trace = TraceContext()

    def fail(payload):
        raise OSError("write failed")

    with pytest.raises(OSError):
        TraceCollector(fail).collect(trace)
    snapshot = trace.to_dict()
    clock[0] = 50
    delivered = []
    TraceCollector(delivered.append).collect(trace)
    assert delivered == [snapshot]
    with pytest.raises(TypeError):
        TraceCollector(None)
    with pytest.raises(TypeError):
        TraceCollector(delivered.append).collect({})
