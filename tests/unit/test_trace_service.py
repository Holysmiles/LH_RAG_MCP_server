"""Persisted trace parsing, ordering, corruption tolerance and timing boundaries."""

import json

import pytest

from observability.dashboard.services.trace_service import TraceService


pytestmark = pytest.mark.unit


def snapshot(identity="one", kind="ingestion", started="2026-10-19T10:00:00+00:00"):
    return {"trace_id": identity, "trace_type": kind, "started_at": started,
            "finished_at": "2026-10-19T11:00:00+00:00", "total_elapsed_ms": 24,
            "stages": [{"stage": "load", "elapsed_ms": 10, "method": "pdf",
                        "details": {"status": "completed", "source_path": "docs/sample.pdf", "collection": "docs"}}]}


def test_filters_and_sorts_by_actual_time_across_timezones(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [snapshot(), snapshot("new", started="2026-10-19T13:00:00+02:00"),
               snapshot("query", kind="query"), snapshot("old", started="2026-10-19T09:00:00")]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    service = TraceService(path)
    traces = service.load_traces("ingestion")
    assert [trace.trace_id for trace in traces] == ["new", "one", "old"]
    assert traces[0].detail("source_path") == "docs/sample.pdf"
    assert traces[0].detail("absent") == ""
    assert traces[0].status == "completed"
    assert len(service.load_traces()) == 4 and service.invalid_line_count == 0
    assert service.load_traces("query")[0].trace_id == "query"


@pytest.mark.parametrize("update", [
    {"started_at": "not a date"}, {"finished_at": 3}, {"trace_id": []},
    {"trace_type": "other"}, {"total_elapsed_ms": -1}, {"total_elapsed_ms": float("nan")},
    {"total_elapsed_ms": True}, {"stages": {}}, {"stages": [None]},
    {"stages": [{"stage": "load", "elapsed_ms": float("inf")}]},
    {"stages": [{"stage": "load", "details": None}]},
])
def test_invalid_lines_do_not_hide_valid_history(tmp_path, update):
    path = tmp_path / "trace.jsonl"
    invalid = dict(snapshot(), **update)
    path.write_text(json.dumps(invalid) + "\n[]\n\n" + json.dumps(snapshot()) + '\n{"trace_id":', encoding="utf-8")
    service = TraceService(path)
    assert len(service.load_traces("ingestion")) == 1
    assert service.invalid_line_count == 3


def test_missing_file_and_refresh_read_new_data_without_creating_file(tmp_path):
    path = tmp_path / "missing" / "trace.jsonl"
    service = TraceService(path)
    assert service.load_traces() == [] and not path.parent.exists()
    path.parent.mkdir()
    path.write_text(json.dumps(snapshot()), encoding="utf-8")
    assert len(service.load_traces()) == 1
    path.write_text("", encoding="utf-8")
    assert service.load_traces() == [] and service.invalid_line_count == 0
    with pytest.raises(ValueError):
        service.load_traces("unknown")


def test_statuses_and_top_level_durations_do_not_double_count_nested_stages(tmp_path):
    record = snapshot()
    record["stages"] += [{"stage": "dense_encoder", "elapsed_ms": 500},
                         {"stage": "embed", "duration_ms": 7, "details": {"status": "failed"}},
                         {"stage": "split", "details": {"status": "completed"}}]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    service = TraceService(path)
    trace = service.load_traces()[0]
    assert trace.status == "failed"
    assert service.ingestion_timings(trace) == [
        {"stage": "load", "elapsed_ms": 10, "status": "completed"},
        {"stage": "embed", "elapsed_ms": 7, "status": "failed"}]
    for status in ("skipped", "completed"):
        record["stages"] = [{"stage": "integrity", "details": {"status": status}}]
        path.write_text(json.dumps(record), encoding="utf-8")
        assert service.load_traces()[0].status == status
    record["finished_at"] = None
    record["stages"] = []
    path.write_text(json.dumps(record), encoding="utf-8")
    assert service.load_traces()[0].status == "running"
