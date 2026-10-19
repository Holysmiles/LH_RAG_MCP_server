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
from PIL import Image

from ingestion.storage import ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService
from observability.dashboard.services.data_service import DataService


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/observability/dashboard/app.py"


@pytest.fixture
def dashboard_config(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["llm"]["api_key"] = "dashboard-private-key"
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    config["ingestion"] = dict(config.get("ingestion", {}), image_root=str(tmp_path / "images"),
                               image_index_db_path=str(tmp_path / "images.db"),
                               integrity_db_path=str(tmp_path / "history.db"))
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
def test_all_navigation_pages_are_reachable(dashboard_config, file, title):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.switch_page(f"pages/{file}.py").run()
    assert not app.exception
    assert app.title[0].value == title
    assert app.info


@pytest.fixture
def browser_data(dashboard_config, tmp_path):
    path, config = dashboard_config
    store = ChromaStore(config["vector_store"])
    images = ImageStorage(config["ingestion"]["image_root"], config["ingestion"]["image_index_db_path"])
    history = SQLiteIntegrityChecker(config["ingestion"]["integrity_db_path"])
    image = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    # The same content hash in two collections must remain independently selectable.
    for collection in ("alpha", "beta"):
        source = str(tmp_path / f"{collection}.pdf")
        history.mark_success("a" * 64, source, chunk_count=2)
        images.store(f"{collection}-image", image, collection, doc_hash="a" * 64)
        store.upsert([
            {"id": f"{collection}-{index}", "text": f"{collection} chunk {index}", "dense_vector": [1., 0.],
             "metadata": {"document_id": "a" * 64, "file_hash": "a" * 64,
                          "source_path": source, "collection": collection, "chunk_index": index,
                          "image_refs": [f"{collection}-image"] if index == 0 else []}}
            for index in range(2)
        ])
    return DataService(store, images, history)


def open_browser():
    return AppTest.from_file(APP, default_timeout=30).run().switch_page("pages/data_browser.py").run()


def test_browser_empty_state_does_not_create_storage(dashboard_config):
    _, config = dashboard_config
    app = open_browser()
    assert not app.exception and app.info
    for path in (config["vector_store"]["persist_directory"],
                 *(config["ingestion"][key] for key in ("image_root", "image_index_db_path", "integrity_db_path"))):
        assert not Path(path).exists()


def test_browser_lists_filters_and_displays_chunks_metadata_and_images(browser_data):
    app = open_browser()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 2
    assert set(app.dataframe[0].value.columns) == {"来源路径", "集合", "文本块数", "图片数", "摄入时间"}
    app.selectbox[0].set_value("beta").run()
    assert not app.exception and not app.error
    assert list(app.dataframe[0].value["集合"]) == ["beta"]
    assert list(app.dataframe[0].value["文本块数"]) == [2]
    texts = [element.value for element in app.text]
    assert "beta chunk 0" in texts and "beta chunk 1" in texts
    assert "alpha chunk 0" not in texts
    assert len(app.json) == 2
    assert len(app.get("image")) == 2  # chunk association and document gallery
    assert not app.warning
    app.selectbox[0].set_value(None).run()
    assert len(app.dataframe[0].value) == 2
    app.selectbox[1].set_value(1).run()
    assert "beta chunk 0" in [element.value for element in app.text]


def test_browser_missing_and_corrupt_images_do_not_break_text(browser_data):
    image_path = browser_data.get_image_path("alpha-image")
    image_path.unlink()
    app = open_browser()
    assert not app.exception and app.warning
    assert "alpha chunk 0" in [element.value for element in app.text]
    image_path.write_bytes(b"not an image")
    app.run()
    assert not app.exception and app.warning
    assert "无法预览" in app.warning[0].value


def test_browser_rejects_image_paths_outside_managed_root(browser_data, tmp_path):
    import sqlite3

    external = tmp_path / "external.png"
    Image.new("RGB", (8, 8), "red").save(external)
    with sqlite3.connect(browser_data.image_storage.db_path) as connection:
        connection.execute("UPDATE image_index SET file_path = ? WHERE image_id = ?",
                           (str(external), "alpha-image"))
    assert browser_data.get_image_path("alpha-image") is None
    app = open_browser()
    assert not app.exception and app.warning
    assert not app.get("image")


def test_browser_storage_error_is_actionable_and_hides_exception_details(browser_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("dashboard-private-key")
    monkeypatch.setattr(DataService, "list_documents", fail)
    app = open_browser()
    assert not app.exception and app.error
    assert "无法读取文档索引" in app.error[0].value
    assert "dashboard-private-key" not in str(app.main)


def test_browser_missing_document_and_refresh(browser_data, monkeypatch):
    original = DataService.get_document_detail
    def missing(*args, **kwargs):
        raise KeyError("gone")
    monkeypatch.setattr(DataService, "get_document_detail", missing)
    app = open_browser()
    assert not app.exception and app.warning
    monkeypatch.setattr(DataService, "get_document_detail", original)
    app.button[0].click().run()
    assert not app.exception and not app.warning
    assert len(app.json) == 2


def test_browser_service_without_optional_indexes(dashboard_config, tmp_path):
    path, config = dashboard_config
    store = ChromaStore(config["vector_store"])
    store.upsert([{"id": "one", "text": "plain text", "dense_vector": [1., 0.],
                   "metadata": {"source_path": "docs/plain.pdf", "document_id": "plain",
                                "collection": "docs"}}])
    service = DataService.from_settings(ConfigService(path).settings)
    document = service.list_documents("docs")[0]
    assert document.ingested_at is None and document.image_count == 0
    assert service.get_document_detail(document).chunks[0]["text"] == "plain text"
    app = open_browser()
    assert not app.exception and not app.error
    assert list(app.dataframe[0].value["摄入时间"]) == ["未知"]
    assert not (tmp_path / "history.db").exists() and not (tmp_path / "images.db").exists()


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
