"""Native MCP client -> subprocess server -> real ingested stores -> citations.

Only model HTTP endpoints are local doubles. Client negotiation, tool schemas,
stdio transport, PDF ingestion, retrieval, reranking and citation assembly use
the production implementations. All state lives in the temporary directory.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import sys
from threading import Thread

import anyio
from mcp import ClientSession, MCPError
from mcp.client.stdio import StdioServerParameters, stdio_client
import pytest
import yaml

from core.settings import load_settings
from ingestion import IngestionPipeline
from libs.vector_store.chroma_store import ChromaStore


pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "tests/fixtures/sample_documents"


@pytest.fixture
def knowledge(tmp_path):
    calls = []

    class Models(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            if self.path.endswith("/embeddings"):
                response = {"data": [{"index": index, "embedding": [1., (len(text) % 19) / 100, .1]}
                                     for index, text in enumerate(payload["input"])]}
            elif self.path.endswith("/chat/completions"):
                candidates = json.loads(payload["messages"][-1]["content"])["candidates"]
                response = {"choices": [{"message": {"content": json.dumps({
                    "ranked_ids": [item["chunk_id"] for item in reversed(candidates)]})}}]}
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

    http = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    worker = Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
        for section in ("llm", "embedding"):
            config[section] = {"provider": "openai", "model": "local-e2e-model", "api_key": "private-test-key",
                               "base_url": f"http://127.0.0.1:{http.server_port}/v1", "timeout": 5}
        config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
        config["splitter"].update(chunk_size=300, chunk_overlap=30)
        config["observability"]["trace_file"] = str(tmp_path / "traces.jsonl")
        config["retrieval"].update(dense_top_k=10, sparse_top_k=10, fusion_top_k=2)
        config["ingestion"].update({key: str(tmp_path / name) for key, name in {
            "documents_path": "documents", "integrity_db_path": "history.db", "bm25_path": "bm25",
            "image_root": "images", "image_index_db_path": "images.db"}.items()})
        config["rerank"] = {"enabled": True, "provider": "llm", "timeout_seconds": 5, "top_m": 10,
                             "prompt_path": str(ROOT / "config/prompts/rerank.txt")}
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        primary, archive = tmp_path / "documents/primary", tmp_path / "documents/archive"
        primary.mkdir(parents=True)
        archive.mkdir()
        pipeline = IngestionPipeline(load_settings(path))
        results = []
        for sample in ("simple.pdf", "with_images.pdf"):
            source = primary / sample
            shutil.copyfile(SAMPLES / sample, source)
            results.append(pipeline.run(source, collection="primary"))
        archive_source = archive / "simple.pdf"
        shutil.copyfile(SAMPLES / "simple.pdf", archive_source)
        pipeline.run(archive_source, collection="archive", force=True)
        records = ChromaStore(config["vector_store"], create_if_missing=False).list_records()
        assert len(records) > 2 and all(result.chunk_count > 0 for result in results)
        yield path, config, calls, results, {record["id"]: record for record in records}
    finally:
        http.shutdown()
        http.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


@pytest.mark.parametrize("rerank_enabled", [False, True])
def test_native_client_discovers_tools_queries_citations_and_shuts_down(knowledge, tmp_path, rerank_enabled):
    path, config, calls, ingested, records = knowledge
    config["rerank"]["enabled"] = rerank_enabled
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    parameters = StdioServerParameters(
        command=sys.executable, args=["-u", str(ROOT / "main.py"), "--settings", str(path)],
        cwd=tmp_path, env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    log_path = tmp_path / "server-stderr.log"
    unexpected = []
    returned_ids = []

    async def on_message(message):
        unexpected.append(message)

    async def scenario():
        with anyio.fail_after(90):
            with log_path.open("w", encoding="utf-8") as stderr:
                async with stdio_client(parameters, errlog=stderr) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=45,
                                             message_handler=on_message) as client:
                        initialized = await client.initialize()
                        assert initialized.server_info.name == "modular-rag-mcp-server"
                        assert initialized.capabilities.tools is not None
                        before_query = len(calls)
                        tools = (await client.list_tools()).tools
                        schemas = {tool.name: tool.input_schema for tool in tools}
                        assert set(schemas) == {"query_knowledge_hub", "list_collections", "get_document_summary"}
                        assert schemas["query_knowledge_hub"]["required"] == ["query"]
                        collections = await client.call_tool("list_collections", {})
                        assert not collections.is_error
                        assert collections.structured_content == {"collections": ["archive", "primary"], "total": 2}
                        assert len(calls) == before_query  # Handshake/discovery use no model.

                        with pytest.raises(MCPError) as invalid:
                            await client.call_tool("query_knowledge_hub", {"query": "query", "top_k": 0})
                        assert invalid.value.error.code == -32602
                        assert len(calls) == before_query

                        result = await client.call_tool("query_knowledge_hub", {
                            "query": "Retrieval augmented generation", "collection": "primary", "top_k": 2})
                        assert not result.is_error
                        structured = result.structured_content
                        assert structured["query"] == "Retrieval augmented generation"
                        assert structured["total"] == len(structured["citations"]) == 2
                        returned_ids.extend(citation["chunk_id"] for citation in structured["citations"])
                        markdown = result.content[0].text
                        for number, citation in enumerate(structured["citations"], start=1):
                            record = records[citation["chunk_id"]]
                            assert record["metadata"]["collection"] == "primary"
                            assert citation["source"] == record["metadata"]["source_path"]
                            expected_page = record["metadata"].get("page", record["metadata"].get("page_number"))
                            assert citation["page"] == expected_page
                            assert citation["index"] == number and math.isfinite(citation["score"])
                            assert f"[{number}]" in markdown and record["text"] in markdown
                        assert "archive" not in markdown

                        summary = await client.call_tool("get_document_summary", {"doc_id": ingested[0].document_id})
                        assert not summary.is_error and summary.structured_content["doc_id"] == ingested[0].document_id
                        missing = await client.call_tool("query_knowledge_hub", {
                            "query": "embedded image", "collection": "absent", "top_k": 1})
                        assert not missing.is_error and missing.structured_content["citations"] == []
                        assert missing.structured_content["total"] == 0
                        again = await client.call_tool("query_knowledge_hub", {
                            "query": "图像 embedded image", "collection": "primary", "top_k": 1})
                        assert not again.is_error and len(again.structured_content["citations"]) == 1
                        assert "图像" in again.structured_content["query"]
                        assert await client.send_ping() is not None

    anyio.run(scenario)
    assert not any(isinstance(message, Exception) for message in unexpected)
    stderr = log_path.read_text(encoding="utf-8")
    assert "MCP stdio server stopped" in stderr and "Traceback" not in stderr
    assert "private-test-key" not in stderr
    traces = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = [trace for trace in traces if trace["trace_type"] == "query"]
    assert len(queries) == 3 and all(trace["finished_at"] for trace in queries)
    rerank = next(stage for stage in queries[0]["stages"] if stage["stage"] == "rerank")
    assert not rerank["fallback"]
    assert returned_ids == rerank["selected_ids"]
    assert rerank["output_ids"] == (list(reversed(rerank["input_ids"])) if rerank_enabled else rerank["input_ids"])
    chat_calls = [payload for route, payload in calls if route.endswith("/chat/completions")]
    assert bool(chat_calls) is rerank_enabled
