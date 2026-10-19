"""Ingestion trace navigation, refresh, empty/error states and phase chart."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
import yaml

from core.trace import TraceContext
from observability.dashboard.services.trace_service import TraceService


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/observability/dashboard/app.py"
pytestmark = pytest.mark.integration


@pytest.fixture
def trace_path(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    path = tmp_path / "traces.jsonl"
    config["observability"]["trace_file"] = str(path)
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(config_path))
    return path


def record(status="completed", kind="ingestion"):
    trace = TraceContext(kind)
    for name in ("load", "split", "transform", "embed", "upsert"):
        trace.record_stage(name, {"details": {"status": status, "source_path": "sample.pdf", "collection": "docs"}},
                           elapsed_ms=5, method="fake")
    trace.finish()
    return trace.to_dict()


def open_page():
    return AppTest.from_file(APP, default_timeout=30).run().switch_page("pages/ingestion_traces.py").run()


def test_empty_and_refresh_then_select_failed_trace(trace_path):
    app = open_page()
    assert not app.exception and app.info and not trace_path.exists()
    first, second = record(), record("failed")
    # Windows can assign identical wall-clock timestamps to adjacent snapshots.
    first["started_at"] = "2026-10-19T10:00:00+00:00"
    second["started_at"] = "2026-10-19T11:00:00+00:00"
    trace_path.write_text("\n".join(json.dumps(item) for item in (first, record(kind="query"), second)), encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert list(app.dataframe[0].value["状态"]) == ["失败", "成功"]
    assert len(app.json) == 5 and len(app.get("vega_lite_chart")) == 1
    assert any(second["trace_id"] in element.value for element in app.text)
    app.selectbox[0].set_value(1).run()
    assert any(first["trace_id"] in element.value for element in app.text)


def test_partial_line_warning_preserves_history_and_no_timing_fallback(trace_path):
    trace = record("skipped")
    trace["stages"] = [{"stage": "integrity", "details": {"status": "skipped"}}]
    trace_path.write_text(json.dumps(trace) + '\n{"trace_id":', encoding="utf-8")
    app = open_page()
    assert not app.exception and app.warning and app.info
    assert list(app.dataframe[0].value["状态"]) == ["跳过"]
    assert not app.get("vega_lite_chart")


def test_unreadable_file_has_generic_error_without_exception_content(trace_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("private-api-key")
    monkeypatch.setattr(TraceService, "load_traces", fail)
    app = open_page()
    assert not app.exception and app.error
    assert "无法读取追踪文件" in app.error[0].value
    assert "private-api-key" not in str(app.main)


def query_record(query, fallback=False):
    payload = record(kind="query")
    payload["stages"] = [
        {"stage": "query_processing", "elapsed_ms": 1, "details": {"query": query}},
        {"stage": "dense_retrieval", "elapsed_ms": 5,
         "results": [{"chunk_id": "a", "text": "dense text", "score": 0.9, "metadata": {"page": 1}}]},
        {"stage": "sparse_retrieval", "elapsed_ms": 3, "fallback": fallback,
         "error_type": "OSError" if fallback else None,
         "results": [] if fallback else [{"chunk_id": "b", "text": "sparse text", "score": 0.1}]},
        {"stage": "fusion", "elapsed_ms": 1},
        {"stage": "rerank", "elapsed_ms": 2, "input_ids": ["a", "b"], "output_ids": ["b", "a"],
         "selected_ids": ["b"], "fallback": fallback, "reason": "timeout" if fallback else None},
    ]
    if fallback:
        payload["stages"][-1]["output_ids"] = ["a", "b"]
    return payload


def open_queries():
    return AppTest.from_file(APP, default_timeout=30).run().switch_page("pages/query_traces.py").run()


def test_query_search_select_comparisons_and_rank_changes(trace_path):
    first, second = query_record("Azure setup"), query_record("Python retrieval")
    trace_path.write_text("\n".join(json.dumps(item) for item in (first, record(), second)), encoding="utf-8")
    app = open_queries()
    assert not app.exception and len(app.dataframe[0].value) == 2
    app.text_input[0].set_value("AZURE").run()
    assert not app.exception
    assert list(app.dataframe[0].value["查询"]) == ["Azure setup"]
    assert len(app.get("vega_lite_chart")) == 1
    assert "dense text" in [element.value for element in app.text]
    assert "sparse text" in [element.value for element in app.text]
    assert list(app.dataframe[1].value["上升名次"]) == [1, -1]
    app.text_input[0].set_value("absent").run()
    assert app.info and not app.dataframe
    app.text_input[0].set_value("").run()
    app.selectbox[0].set_value(first["trace_id"]).run()
    assert "Azure setup" in [element.value for element in app.text]


def test_query_degraded_routes_and_legacy_rank_data(trace_path):
    trace = query_record("degraded", fallback=True)
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    app = open_queries()
    assert not app.exception and app.warning
    assert list(app.dataframe[0].value["状态"]) == ["降级"]
    assert list(app.dataframe[1].value["上升名次"]) == [0, 0]
    for name in ("input_ids", "output_ids", "selected_ids"):
        trace["stages"][-1].pop(name)
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception and len(app.dataframe) == 1
    assert any("排名快照" in element.value for element in app.info)


def test_query_empty_partial_and_permission_error(trace_path, monkeypatch):
    app = open_queries()
    assert not app.exception and app.info and not trace_path.exists()
    trace_path.write_text(json.dumps(query_record("hello")) + '\n{"trace_id":', encoding="utf-8")
    app.button[0].click().run()
    assert app.warning and not app.exception and len(app.dataframe[0].value) == 1
    def fail(*args, **kwargs):
        raise PermissionError("private-api-key")
    monkeypatch.setattr(TraceService, "load_traces", fail)
    app.run()
    assert not app.exception and app.error
    assert "private-api-key" not in str(app.main)
