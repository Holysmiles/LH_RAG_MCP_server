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
