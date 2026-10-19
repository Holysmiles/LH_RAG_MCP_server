"""Render the real Streamlit navigation and overview without a browser."""

from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from streamlit.testing.v1 import AppTest
import yaml

from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/observability/dashboard/app.py"


@pytest.fixture
def dashboard_config(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["llm"]["api_key"] = "dashboard-private-key"
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(path))
    return path, config


def test_empty_overview_displays_configuration_without_creating_a_database(dashboard_config):
    path, config = dashboard_config
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "系统总览"
    assert [metric.value for metric in app.metric] == ["0", "0", "0"]
    assert "openai" in [element.value for element in app.text]
    assert not Path(config["vector_store"]["persist_directory"]).exists()
    rendered = str(app.main) + str(app.sidebar)
    assert "dashboard-private-key" not in rendered
    assert "api_key" not in str(ConfigService(path).component_configs())


@pytest.mark.parametrize("file,title", [
    ("data_browser", "数据浏览器"), ("ingestion_manager", "摄取管理"),
    ("ingestion_traces", "摄取追踪"), ("query_traces", "查询追踪"), ("evaluation_panel", "评估面板"),
])
def test_all_placeholder_pages_are_reachable(dashboard_config, file, title):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.switch_page(f"pages/{file}.py").run()
    assert not app.exception
    assert app.title[0].value == title
    assert app.info


def test_overview_counts_real_persisted_records(dashboard_config):
    path, config = dashboard_config
    store = ChromaStore(config["vector_store"])
    store.upsert([
        {"id": key, "text": key, "dense_vector": [1.0, 0.0],
         "metadata": {"document_id": document, "collection": collection}}
        for key, document, collection in (("a", "doc-1", "alpha"), ("b", "doc-1", "alpha"),
                                          ("c", "doc-2", "beta"))
    ])
    assert store.get_collection_stats() == {"document_count": 2, "chunk_count": 3, "collection_count": 2}
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["2", "3", "2"]


def test_missing_physical_collection_is_reported_without_creating_it(dashboard_config):
    path, config = dashboard_config
    store = ChromaStore(config["vector_store"])
    config["vector_store"]["collection_name"] = "does_not_exist"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception and app.warning
    assert [metric.value for metric in app.metric] == ["—", "—", "—"]
    assert store._client.get_collection("knowledge_hub") is not None
    with pytest.raises(Exception):
        store._client.get_collection("does_not_exist")


def test_bad_settings_show_actionable_error_without_traceback_or_secrets(dashboard_config):
    path, config = dashboard_config
    path.write_text("llm: [dashboard-private-key", encoding="utf-8")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception and app.error
    assert "无法加载配置" in app.error[0].value
    assert "dashboard-private-key" not in str(app.main)


def test_launcher_starts_a_healthy_streamlit_server(dashboard_config, tmp_path):
    path, _ = dashboard_config
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log_path = tmp_path / "streamlit.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/start_dashboard.py"),
             "--settings", str(path), "--port", str(port)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text(encoding="utf-8")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        assert response.read() == b"ok"
                        return
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            pytest.fail("Streamlit did not become healthy: " + log_path.read_text(encoding="utf-8"))
        finally:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
