"""Real CLI subprocesses, Chroma/BM25 persistence, and local fake HTTP models."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.chroma_store import ChromaStore

pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "query.py"


@pytest.fixture
def cli_data(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            if self.path.endswith("/embeddings"):
                response = {"data": [{"index": i, "embedding": [1.0, 0.0]}
                                     for i, _ in enumerate(payload["input"])]}
            elif self.path.endswith("/chat/completions"):
                request = json.loads(payload["messages"][-1]["content"])
                ranked_ids = [item["chunk_id"] for item in reversed(request["candidates"])]
                response = {"choices": [{"message": {"content": json.dumps({"ranked_ids": ranked_ids})}}]}
            else:
                self.send_error(404)
                return
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    config = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    for section in ("embedding", "llm"):
        config[section] = {"provider": "openai", "model": "offline-test", "api_key": "fake-key", "base_url": endpoint}
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    config["ingestion"]["bm25_path"] = str(tmp_path / "bm25")
    config["rerank"] = {"enabled": True, "provider": "llm", "timeout_seconds": 5,
                         "top_m": 3, "prompt_path": str(PROJECT_ROOT / "config" / "prompts" / "rerank.txt")}
    # Production retrievers load the exact persisted stores created here.
    store = ChromaStore(config["vector_store"])
    chunks = [Chunk(key, f"Azure setup guide {key}", {"source_path": f"{key}.pdf", "collection": collection, "page": 2},
                    start_offset=0, end_offset=len(f"Azure setup guide {key}"))
              for key, collection in (("a", "alpha"), ("b", "alpha"), ("c", "beta"))]
    store.upsert([{"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata,
                   "dense_vector": vector}
                  for chunk, vector in zip(chunks, ([1.0, 0.0], [0.8, 0.2], [0.0, 1.0]))])
    BM25Indexer(config["ingestion"]["bm25_path"]).build(SparseEncoder().encode(chunks))
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    worker.start()
    try:
        yield settings_path, config, calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def run_cli(tmp_path, *args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp_path,
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)


def test_cli_collection_top_k_verbose_and_actual_llm_reranking(tmp_path, cli_data, monkeypatch):
    path, config, calls = cli_data
    result = run_cli(tmp_path, "--settings", str(path), "--query", "Azure", "--collection", "alpha", "--top-k", "1", "--verbose")
    assert result.returncode == 0, result.stderr
    for stage in ("Dense 召回", "Sparse 召回", "Fusion 融合", "Rerank 重排"):
        assert stage in result.stdout
    final = result.stdout.split("Top-K 检索结果")[-1]
    assert "(1)" in final and "chunk_id=b" in final
    assert "b.pdf" in final and "页码: 2" in final and "score=" in final
    assert "chunk_id=c" not in final
    chat_calls = [payload for path, payload in calls if path.endswith("/chat/completions")]
    assert len(chat_calls) == 1
    request = json.loads(chat_calls[0]["messages"][-1]["content"])
    assert len(request["candidates"]) == 2  # Final Top-1 did not prematurely truncate rerank input.
    assert "fallback=false" in result.stdout
    traces = [json.loads(line) for line in (tmp_path / "logs/traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 1
    assert traces[0]["trace_type"] == "query" and traces[0]["finished_at"] is not None
    assert {stage["stage"] for stage in traces[0]["stages"]} >= {
        "query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank",
    }
    rerank = next(stage for stage in traces[0]["stages"] if stage["stage"] == "rerank")
    assert rerank["input_ids"] == ["a", "b"] and rerank["output_ids"] == ["b", "a"]
    assert rerank["selected_ids"] == ["b"]
    config["observability"]["trace_file"] = str(tmp_path / "logs/traces.jsonl")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(path))
    app = AppTest.from_file(PROJECT_ROOT / "src/observability/dashboard/app.py", default_timeout=30).run()
    app.switch_page("pages/query_traces.py").run()
    assert not app.exception and not app.error
    assert list(app.dataframe[0].value["查询"]) == ["Azure"]
    assert list(app.dataframe[1].value["Chunk ID"]) == ["b", "a"]
    assert list(app.dataframe[1].value["上升名次"]) == [1, -1]
    assert len(app.get("vega_lite_chart")) == 1


def test_no_rerank_skips_backend_even_when_provider_is_invalid(tmp_path, cli_data):
    path, config, calls = cli_data
    config["rerank"] = {"enabled": True, "provider": "unknown", "timeout_seconds": -1}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_cli(tmp_path, "--settings", str(path), "--query", "Azure", "--top-k", "2", "--no-rerank", "--verbose")
    assert result.returncode == 0, result.stderr
    assert "Top-K 检索结果 (2)" in result.stdout
    assert "已跳过" in result.stdout
    assert not any(path.endswith("/chat/completions") for path, _ in calls)


def test_empty_database_has_friendly_message(tmp_path, cli_data):
    path, config, calls = cli_data
    config["vector_store"]["persist_directory"] = str(tmp_path / "empty-chroma")
    config["ingestion"]["bm25_path"] = str(tmp_path / "empty-bm25")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_cli(tmp_path, "--settings", str(path), "--query", "Azure")
    assert result.returncode == 0, result.stderr
    assert "未找到相关文档" in result.stdout and "ingest.py" in result.stdout
    assert not any(path.endswith("/chat/completions") for path, _ in calls)


def test_missing_collection_has_no_cross_collection_results(tmp_path, cli_data):
    path, _, _ = cli_data
    result = run_cli(tmp_path, "--settings", str(path), "--query", "Azure", "--collection", "missing", "--no-rerank")
    assert result.returncode == 0, result.stderr
    assert "未找到相关文档" in result.stdout


@pytest.mark.parametrize("args", [[], ["--query", " "], ["--query", "Azure", "--top-k", "0"],
                                  ["--query", "Azure", "--top-k", "abc"], ["--query", "Azure", "--collection", " "]])
def test_invalid_arguments_fail_before_component_initialization(tmp_path, args):
    result = run_cli(tmp_path, *args)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert not (tmp_path / "data").exists()


def test_help_does_not_require_model_configuration(tmp_path):
    result = run_cli(tmp_path, "--help")
    assert result.returncode == 0
    for flag in ("--query", "--top-k", "--collection", "--verbose", "--no-rerank"):
        assert flag in result.stdout


def test_missing_settings_returns_configuration_error(tmp_path):
    result = run_cli(tmp_path, "--query", "Azure", "--settings", str(tmp_path / "missing.yaml"))
    assert result.returncode == 2
    assert "Cannot start query" in result.stderr
