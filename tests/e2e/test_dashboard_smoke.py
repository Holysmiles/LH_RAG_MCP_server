"""Render all six Dashboard pages against a real populated temporary corpus.

PDF ingestion, storage, query traces and evaluation are production code. Only
embedding calls use an offline double; no UI service or page is replaced.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
import yaml

from core.settings import load_settings
from core.query_engine import DenseRetriever, SparseRetriever, HybridSearch, QueryProcessor, Fusion, Reranker
from core.trace.trace_collector import query_trace
from ingestion import IngestionPipeline
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.chroma_store import ChromaStore


pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/observability/dashboard/app.py"
FIXTURES = ROOT / "tests/fixtures"
PAGES = [
    ("overview", "系统总览"), ("data_browser", "数据浏览器"), ("ingestion_manager", "摄取管理"),
    ("ingestion_traces", "摄取追踪"), ("query_traces", "查询追踪"), ("evaluation_panel", "评估面板"),
]


class OfflineEmbedding(BaseEmbedding):
    def embed(self, texts, trace=None):
        return [[1., float(len(text) + 1) / 100] for text in texts]


@pytest.fixture(scope="module")
def populated_corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dashboard-smoke")
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["llm"]["api_key"] = "dashboard-smoke-private-key"
    config["vision_llm"]["enabled"] = False
    config["vector_store"]["persist_directory"] = str(folder / "chroma")
    config["splitter"].update(chunk_size=300, chunk_overlap=30)
    config["ingestion"].update({key: str(folder / name) for key, name in {
        "documents_path": "documents", "integrity_db_path": "history.db", "bm25_path": "bm25",
        "image_root": "images", "image_index_db_path": "images.db"}.items()})
    config["observability"]["trace_file"] = str(folder / "traces.jsonl")
    config["evaluation"] = {"provider": "custom", "top_k": 3,
                            "test_sets": [str(FIXTURES / "golden_test_set.json")]}
    config["rerank"] = {"enabled": False, "provider": "none"}
    path = folder / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = load_settings(path)
    embedding = OfflineEmbedding()
    pipeline = IngestionPipeline(settings, batch_processor=BatchProcessor(
        settings, dense_encoder=DenseEncoder(settings, embedding=embedding), sparse_encoder=SparseEncoder()))
    for source in ("simple.pdf", "with_images.pdf"):
        result = pipeline.run(FIXTURES / "sample_documents" / source, collection="smoke")
        assert result.status == "success" and not result.skipped
    store = ChromaStore(settings.vector_store, create_if_missing=False)
    records = store.list_records()
    hybrid = HybridSearch(settings, QueryProcessor(),
                          DenseRetriever(settings, embedding_client=embedding, vector_store=store),
                          SparseRetriever(settings, vector_store=store), Fusion(settings))
    with query_trace(settings) as trace:
        hits = hybrid.search("Retrieval augmented generation", top_k=3, trace=trace)
        assert Reranker(settings).rerank("Retrieval augmented generation", hits, top_k=2, trace=trace)
    snapshots = [json.loads(line) for line in Path(settings.observability["trace_file"]).read_text(encoding="utf-8").splitlines()]
    assert {trace["trace_type"] for trace in snapshots} == {"query", "ingestion"}
    return path, len(records), embedding


@pytest.fixture
def app(populated_corpus, monkeypatch):
    path, _, embedding = populated_corpus
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(path))
    monkeypatch.setattr(EmbeddingFactory, "create", lambda settings: embedding)
    return AppTest.from_file(APP, default_timeout=40).run()


def assert_healthy(app, title):
    assert not app.exception, str(app.exception)
    assert not app.error, str(app.error)
    assert not app.warning, str(app.warning)
    assert app.title[0].value == title
    assert "dashboard-smoke-private-key" not in str(app.main) + str(app.sidebar)


@pytest.mark.parametrize("page,title", PAGES)
def test_populated_dashboard_page_smoke(app, populated_corpus, page, title):
    _, chunk_count, _ = populated_corpus
    app.switch_page(f"pages/{page}.py").run()
    assert_healthy(app, title)
    if page == "overview":
        assert [metric.value for metric in app.metric] == ["2", str(chunk_count), "1"]
    elif page == "data_browser":
        table = app.dataframe[0].value
        assert len(table) == 2 and sum(table["文本块数"]) == chunk_count
        image_document = next(index for index, source in enumerate(table["来源路径"])
                              if Path(source).name == "with_images.pdf")
        app.selectbox[1].set_value(image_document).run()
        assert_healthy(app, title)
        assert app.json and app.get("image")
    elif page == "ingestion_manager":
        assert app.file_uploader and app.button[0].disabled
        assert len([button for button in app.button if button.label == "删除文档"]) == 2
    elif page == "ingestion_traces":
        assert len(app.dataframe[0].value) == 2
        assert set(app.dataframe[0].value["状态"]) == {"成功"}
        assert app.get("vega_lite_chart") and app.json
    elif page == "query_traces":
        assert len(app.dataframe[0].value) >= 1
        assert app.get("vega_lite_chart") and app.expander
        assert any("重排前" in table.value.columns for table in app.dataframe)
    else:
        app.button[0].click().run()
        assert_healthy(app, title)
        assert app.success and len(app.metric) == 2
        assert len(app.dataframe[1].value) == 6 and len(app.json) == 6


def test_all_pages_share_a_session_and_can_return_to_overview(app, populated_corpus):
    for page, title in PAGES:
        app.switch_page(f"pages/{page}.py").run()
        assert_healthy(app, title)
    app.switch_page("pages/overview.py").run()
    assert_healthy(app, "系统总览")
    assert [metric.value for metric in app.metric] == ["2", str(populated_corpus[1]), "1"]
