"""Run real retrieval evaluations through the Dashboard with offline embeddings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
import yaml

from core import Chunk
from core.settings import load_settings
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.evaluation_service import EvaluationService
from observability.evaluation import ragas_evaluator


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/observability/dashboard/app.py"
pytestmark = pytest.mark.integration


@pytest.fixture
def context(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    config["ingestion"]["bm25_path"] = str(tmp_path / "bm25")
    config["observability"]["trace_file"] = str(tmp_path / "traces.jsonl")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps({"test_cases": [
        {"query": "Azure alpha", "collection": "alpha", "expected_chunk_ids": ["a", "b"]},
        {"query": "Azure beta", "collection": "beta", "expected_chunk_ids": ["c"]}]}), encoding="utf-8")
    second.write_text(json.dumps({"test_cases": [
        {"query": "Azure", "expected_chunk_ids": ["absent"]}]}), encoding="utf-8")
    config["evaluation"] = {"provider": "custom", "test_sets": [str(first), str(second)], "top_k": 2}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(path))
    calls = []
    class Embedding(BaseEmbedding):
        def embed(self, texts, trace=None):
            calls.append(texts)
            return [[1., 0.] for _ in texts]
    monkeypatch.setattr(EmbeddingFactory, "create", lambda settings: Embedding())
    return path, config, calls


def seed(config):
    store = ChromaStore(config["vector_store"])
    chunks = [Chunk(key, f"Azure setup {key}", {"source_path": f"{key}.pdf", "collection": owner},
                    start_offset=0, end_offset=13)
              for key, owner in (("a", "alpha"), ("b", "alpha"), ("c", "beta"))]
    store.upsert([{"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata, "dense_vector": [1., 0.]}
                  for chunk in chunks])
    BM25Indexer(config["ingestion"]["bm25_path"]).build(SparseEncoder().encode(chunks))


def open_page():
    return AppTest.from_file(APP, default_timeout=30).run().switch_page("pages/evaluation_panel.py").run()


def test_empty_page_does_not_create_index_or_model(context):
    _, config, calls = context
    app = open_page()
    assert not app.exception and app.info and not app.metric
    assert not calls and not Path(config["vector_store"]["persist_directory"]).exists()
    app.button[0].click().run()
    assert not app.exception and app.error
    assert "源文档" in app.error[0].value
    assert not calls and not Path(config["vector_store"]["persist_directory"]).exists()


def test_panel_runs_real_retrieval_retains_reports_and_selects_history(context):
    _, config, calls = context
    seed(config)
    app = open_page()
    assert not calls and not app.exception
    app.button[0].click().run()
    assert not app.exception and not app.error and app.success
    assert [metric.value for metric in app.metric] == ["1.0000", "1.0000"]
    assert len(app.dataframe[1].value) == 2 and len(app.json) == 2
    assert len(calls) == 2
    first_id = app.selectbox[2].value
    app.number_input[0].set_value(1).run()
    assert len(calls) == 2  # Widget reruns do not repeat a benchmark.
    app.selectbox[0].set_value("custom").run()
    app.selectbox[1].set_value(Path(config["evaluation"]["test_sets"][1])).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert [metric.value for metric in app.metric] == ["0.0000", "0.0000"]
    assert len(app.dataframe[0].value) == 2 and len(calls) == 3
    app.selectbox[2].set_value(first_id).run()
    assert [metric.value for metric in app.metric] == ["1.0000", "1.0000"]
    assert len(calls) == 3


@pytest.mark.parametrize("backend", ["ragas", "all"])
def test_semantic_backends_require_answers_before_model_calls(context, backend):
    _, config, calls = context
    seed(config)
    app = open_page()
    app.selectbox[0].set_value(backend).run()
    app.button[0].click().run()
    assert not app.exception and app.error and "answer" in app.error[0].value
    assert not calls and not app.metric


def test_invalid_golden_set_and_failed_run_do_not_replace_previous_report(context, monkeypatch):
    _, config, _ = context
    seed(config)
    app = open_page()
    app.button[0].click().run()
    Path(config["evaluation"]["test_sets"][0]).write_text("not json", encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception and app.error and "黄金集" in app.error[0].value
    assert len(app.dataframe[0].value) == 1
    def fail(*args, **kwargs):
        raise RuntimeError("private-api-key")
    monkeypatch.setattr(EvaluationService, "run", fail)
    app.button[0].click().run()
    assert not app.exception and app.error and "private-api-key" not in str(app.main)
    assert len(app.dataframe[0].value) == 1


def test_service_config_is_not_mutated_by_backend_overrides(context):
    path, config, _ = context
    seed(config)
    settings = load_settings(path)
    service = EvaluationService(settings)
    report = service.run(config["evaluation"]["test_sets"][0], "custom", 1)
    assert report.top_k == 1 and settings.evaluation == config["evaluation"]


def test_invalid_panel_configuration_is_actionable(context):
    path, config, calls = context
    config["evaluation"]["test_sets"] = []
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    app = open_page()
    assert not app.exception and app.error and not calls
    config["evaluation"]["test_sets"] = [str(ROOT / "tests/fixtures/golden_test_set.json")]
    config["evaluation"]["top_k"] = 0
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    app = open_page()
    assert not app.exception and "top_k" in app.error[0].value


def test_all_backends_combine_semantic_and_retrieval_metrics(context, monkeypatch):
    _, config, calls = context
    seed(config)
    path = Path(config["evaluation"]["test_sets"][0])
    golden = json.loads(path.read_text(encoding="utf-8"))
    for case in golden["test_cases"]:
        case["answer"] = "Azure setup answer"
    path.write_text(json.dumps(golden), encoding="utf-8")
    metrics = SimpleNamespace(Faithfulness=lambda: SimpleNamespace(name="faithfulness"),
                               AnswerRelevancy=lambda: SimpleNamespace(name="answer_relevancy"))
    observed = []
    def evaluate(**kwargs):
        observed.append(kwargs["dataset"][0])
        return {metric.name: [.8] for metric in kwargs["metrics"]}
    backend = SimpleNamespace(EvaluationDataset=SimpleNamespace(from_list=lambda rows: rows), evaluate=evaluate)
    monkeypatch.setattr(ragas_evaluator, "import_module", lambda name: backend if name == "ragas" else metrics)
    app = open_page()
    app.selectbox[0].set_value("all").run()
    app.button[0].click().run()
    assert not app.exception and not app.error and app.success
    assert [metric.label for metric in app.metric] == ["answer_relevancy", "faithfulness", "hit_rate", "mrr"]
    assert [metric.value for metric in app.metric] == ["0.8000", "0.8000", "1.0000", "1.0000"]
    assert len(calls) == len(observed) == 2
    assert all(sample["retrieved_contexts"] and sample["response"] == "Azure setup answer" for sample in observed)
