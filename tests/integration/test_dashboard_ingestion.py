"""Exercise upload, progress and deletion through the real Streamlit page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
import yaml

from core.settings import load_settings
from ingestion import IngestionPipeline
from ingestion.document_manager import DeleteResult
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from libs.embedding.base_embedding import BaseEmbedding
from observability.dashboard.services.data_service import DataService
from observability.dashboard.services import ingestion_service


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/observability/dashboard/app.py"
PDF = ROOT / "tests/fixtures/sample_documents/with_images.pdf"
pytestmark = pytest.mark.integration


class OfflineEmbedding(BaseEmbedding):
    def embed(self, texts, trace=None):
        return [[float(len(text) + 1), 1.] for text in texts]


@pytest.fixture
def context(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    config["ingestion"].update({key: str(tmp_path / name) for key, name in {
        "documents_path": "documents", "image_root": "images", "image_index_db_path": "images.db",
        "integrity_db_path": "history.db", "bm25_path": "bm25"}.items()})
    config["observability"]["trace_file"] = str(tmp_path / "traces.jsonl")
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(path))
    events = []

    def factory(settings):
        pipeline = IngestionPipeline(settings, batch_processor=BatchProcessor(
            settings, dense_encoder=DenseEncoder(settings, embedding=OfflineEmbedding()),
            sparse_encoder=SparseEncoder()))
        original = pipeline.run
        def run(*args, on_progress=None, **kwargs):
            def observe(*event):
                events.append(event)
                on_progress(*event)
            return original(*args, on_progress=observe, **kwargs)
        pipeline.run = run
        return pipeline

    monkeypatch.setattr(ingestion_service, "IngestionPipeline", factory)
    return load_settings(path), events


def open_page():
    return AppTest.from_file(APP, default_timeout=40).run().switch_page("pages/ingestion_manager.py").run()


def upload(app, name="sample.pdf", content=None):
    app.file_uploader[0].set_value((name, PDF.read_bytes() if content is None else content, "application/pdf")).run()
    app.button[0].click().run()
    return app


def test_real_upload_progress_skip_force_and_delete(context):
    settings, events = context
    app = open_page()
    assert not app.exception and app.button[0].disabled
    upload(app)
    assert not app.exception and not app.error and app.success
    assert [event[1:] for event in events] == [(index, 6) for index in range(1, 7)]
    assert app.get("progress")[0].proto.value == 100
    documents = DataService.from_settings(settings).list_documents()
    assert len(documents) == 1 and documents[0].image_count == 1
    source = Path(documents[0].source_path)
    assert source.is_relative_to(Path(settings.ingestion["documents_path"]))
    assert source.read_bytes() == PDF.read_bytes()
    events.clear()
    app.button[0].click().run()
    assert not app.exception and any("跳过" in info.value for info in app.info)
    assert events == [("integrity", 6, 6)]
    events.clear()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and app.success and len(events) == 6
    delete_button = next(button for button in app.button if button.label == "删除文档")
    delete_button.click().run()
    assert not app.exception and app.success
    assert DataService.from_settings(settings).list_documents() == []
    assert source.exists()
    assert not any(button.label == "删除文档" for button in app.button)


def test_invalid_collection_and_pdf_are_rejected_without_storage(context):
    settings, events = context
    app = open_page()
    app.text_input[0].set_value("../escape").run()
    upload(app)
    assert app.error and not app.exception and events == []
    assert not Path(settings.ingestion["documents_path"]).exists()
    app.text_input[0].set_value("docs").run()
    upload(app, content=b"not a PDF")
    assert app.error and not app.exception and events == []
    assert not Path(settings.ingestion["documents_path"]).exists()


def test_filename_is_contained_and_existing_collection_is_selectable(context):
    settings, _ = context
    app = upload(open_page(), name="../../outside.pdf")
    assert not app.exception and not app.error
    doc = DataService.from_settings(settings).list_documents()[0]
    assert Path(doc.source_path).parent == Path(settings.ingestion["documents_path"]) / "default"
    assert ".." not in Path(doc.source_path).name
    app.run()
    app.selectbox[0].set_value("default").run()
    assert not app.text_input


def test_ingestion_failure_keeps_partial_progress_and_hides_secrets(context, monkeypatch):
    def fail(self, *args, on_progress=None, **kwargs):
        on_progress("load", 2, 6)
        raise RuntimeError("private-api-key")
    monkeypatch.setattr(ingestion_service.IngestionService, "ingest_upload", fail)
    app = upload(open_page())
    assert not app.exception and app.error and not app.success
    assert app.get("progress")[0].proto.value == 33
    assert "private-api-key" not in str(app.main)


def test_failed_delete_keeps_document_and_can_retry(context, monkeypatch):
    settings, _ = context
    app = upload(open_page())
    original = ingestion_service.IngestionService.delete_document
    monkeypatch.setattr(ingestion_service.IngestionService, "delete_document",
                        lambda self, source, collection: DeleteResult(source, collection, errors={"bm25": "OSError"}))
    next(button for button in app.button if button.label == "删除文档").click().run()
    assert not app.exception and app.error and "bm25" in app.error[0].value
    assert len(DataService.from_settings(settings).list_documents()) == 1
    monkeypatch.setattr(ingestion_service.IngestionService, "delete_document", original)
    next(button for button in app.button if button.label == "删除文档").click().run()
    assert not app.exception and app.success
    assert DataService.from_settings(settings).list_documents() == []


@pytest.mark.parametrize("name,content,collection", [
    ("file.exe", b"%PDF-1.7", "docs"), ("", b"%PDF-1.7", "docs"),
    ("file.pdf", b"", "docs"), ("file.pdf", b"invalid", "docs"),
    ("file.pdf", b"%PDF-1.7", "../escape"), ("file.pdf", b"%PDF-1.7", ""),
])
def test_upload_validation_precedes_writes(context, name, content, collection):
    settings, events = context
    with pytest.raises(ValueError):
        ingestion_service.IngestionService(settings).ingest_upload(name, content, collection)
    assert not Path(settings.ingestion["documents_path"]).exists()
    assert events == []


def test_existing_upload_is_never_overwritten(context):
    settings, _ = context
    app = upload(open_page())
    source = Path(DataService.from_settings(settings).list_documents()[0].source_path)
    source.write_bytes(b"existing edited source")
    app.button[0].click().run()
    assert not app.exception and app.error
    assert source.read_bytes() == b"existing edited source"
