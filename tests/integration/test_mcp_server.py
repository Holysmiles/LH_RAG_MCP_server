"""Exercise the real stdio transport, including a live handshake and EOF."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread

import pytest


pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = PROJECT_ROOT / "main.py"


@contextmanager
def running_server(tmp_path, *, launcher=None, arguments=()):
    command = ([sys.executable, "-u", "-c", launcher] if launcher else
               [sys.executable, "-u", str(ENTRY_POINT)])
    process = subprocess.Popen(
        [*command, *arguments], cwd=tmp_path,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    messages = Queue()
    errors = []

    def read_stdout():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    def read_stderr():
        errors.extend(process.stderr)

    readers = [Thread(target=read_stdout, daemon=True),
               Thread(target=read_stderr, daemon=True)]
    for reader in readers:
        reader.start()
    try:
        yield process, messages, errors
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def send(process, message):
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()


def receive(messages, timeout=20):
    try:
        line = messages.get(timeout=timeout)
    except Empty:
        pytest.fail(f"MCP server did not respond within {timeout} seconds")
    assert line is not None, "MCP server closed stdout before responding"
    message = json.loads(line)  # Any non-protocol stdout fails the test.
    assert message["jsonrpc"] == "2.0"
    return message


def assert_clean_shutdown(process, messages):
    process.stdin.close()
    assert process.wait(timeout=20) == 0
    assert messages.get(timeout=5) is None  # No trailing stdout noise.


@pytest.mark.parametrize("noisy", [False, True])
def test_initialize_ping_and_eof_keep_stdout_protocol_only(tmp_path, noisy):
    launcher = None
    if noisy:
        launcher = f'''
import runpy
from mcp_server import server
original_create = server.create_server
def noisy_create(*args, **kwargs):
    print("factory diagnostic")
    instance = original_create(*args, **kwargs)
    original_run = instance.run
    async def noisy_run(*args, **kwargs):
        print("handler diagnostic")
        return await original_run(*args, **kwargs)
    instance.run = noisy_run
    return instance
server.create_server = noisy_create
runpy.run_path({str(ENTRY_POINT)!r}, run_name="__main__")
'''
    with running_server(tmp_path, launcher=launcher) as (process, messages, errors):
        send(process, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "stdio-integration-test", "version": "1.0"},
            },
        })
        response = receive(messages)
        assert response["id"] == 1
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"
        assert result["serverInfo"]["version"]
        assert isinstance(result["capabilities"], dict)
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(process, {"jsonrpc": "2.0", "id": "ping-2", "method": "ping"})
        assert receive(messages) == {"jsonrpc": "2.0", "id": "ping-2", "result": {}}
        assert_clean_shutdown(process, messages)
    stderr = "".join(errors)
    assert "Configuration loaded" in stderr
    assert "MCP stdio server starting" in stderr
    assert "MCP stdio server stopped" in stderr
    if noisy:
        assert "factory diagnostic" in stderr and "handler diagnostic" in stderr
    assert not (tmp_path / "data").exists()


def test_eof_without_initialize_exits_cleanly(tmp_path):
    with running_server(tmp_path) as (process, messages, errors):
        assert_clean_shutdown(process, messages)
    assert "MCP stdio server stopped" in "".join(errors)


@pytest.mark.parametrize("config_text", [None, "llm: {}\n"])
def test_invalid_settings_fail_on_stderr_without_protocol_output(tmp_path, config_text):
    settings = tmp_path / "settings.yaml"
    if config_text is not None:
        settings.write_text(config_text, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ENTRY_POINT), "--settings", str(settings)],
        cwd=tmp_path, input="", capture_output=True, text=True,
        encoding="utf-8", timeout=20,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Failed to load settings" in result.stderr
    assert "MCP stdio server starting" not in result.stderr


def test_live_tool_registry_and_jsonrpc_errors(tmp_path):
    launcher = f'''
import runpy
from mcp_server import server
from mcp_server.protocol_handler import ProtocolHandler
handler = ProtocolHandler("test-tools", "1.0")
schema = {{"type": "object", "properties": {{"query": {{"type": "string"}}}},
          "required": ["query"], "additionalProperties": False}}
def echo(arguments):
    print("tool diagnostic")
    return {{"content": [{{"type": "text", "text": arguments["query"]}}]}}
async def fail(arguments):
    raise RuntimeError("private-provider-secret")
handler.register_tool("echo", "Echo a query", schema, echo)
handler.register_tool("fail", "Fail a query", schema, fail)
original_create = server.create_server
server.create_server = lambda **kwargs: original_create(handler)
runpy.run_path({str(ENTRY_POINT)!r}, run_name="__main__")
'''
    with running_server(tmp_path, launcher=launcher) as (process, messages, errors):
        send(process, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "1.0"}},
        })
        initialization = receive(messages)["result"]
        assert initialization["serverInfo"]["name"] == "test-tools"
        assert initialization["capabilities"]["tools"] == {"listChanged": False}
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = receive(messages)["result"]["tools"]
        assert [tool["name"] for tool in tools] == ["echo", "fail"]
        assert tools[0]["inputSchema"]["required"] == ["query"]
        send(process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "echo", "arguments": {"query": "中文查询"}}})
        result = receive(messages)
        assert result["id"] == 3
        assert result["result"]["content"] == [{"type": "text", "text": "中文查询"}]

        for request_id, method, params, code in [
            (4, "unknown/method", {}, -32601),
            (5, "tools/call", {}, -32602),
            (6, "tools/call", {"name": "echo", "arguments": []}, -32602),
            (7, "tools/call", {"name": "echo", "arguments": {"query": 42}}, -32602),
            (8, "tools/call", {"name": "absent"}, -32602),
            (9, "tools/call", {"name": "fail", "arguments": {"query": "x"}}, -32603),
            (10, "tools/list", {"cursor": "unsupported"}, -32602),
        ]:
            send(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            response = receive(messages)
            assert response["id"] == request_id
            assert response["error"]["code"] == code
            assert "private-provider-secret" not in json.dumps(response)
            assert "Traceback" not in json.dumps(response)

        send(process, {"jsonrpc": "1.0", "id": 11, "method": "ping"})
        assert receive(messages) == {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32600, "message": "Invalid Request"}}
        process.stdin.write("{invalid-json\n")
        process.stdin.flush()
        assert receive(messages) == {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "Parse error"}}
        send(process, {"jsonrpc": "2.0", "id": 12, "method": "ping"})
        assert receive(messages) == {"jsonrpc": "2.0", "id": 12, "result": {}}
        assert_clean_shutdown(process, messages)
    stderr = "".join(errors)
    assert "tool diagnostic" in stderr
    assert "private-provider-secret" not in stderr


def test_default_server_lists_tools_without_opening_database(tmp_path):
    with running_server(tmp_path) as (process, messages, errors):
        send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "test", "version": "1.0"}}})
        assert "tools" in receive(messages)["result"]["capabilities"]
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        response = receive(messages)
        assert response["id"] == 2
        assert [tool["name"] for tool in response["result"]["tools"]] == ["query_knowledge_hub", "list_collections"]
        assert_clean_shutdown(process, messages)
    assert not (tmp_path / "data").exists()


def test_list_collections_live_stdio_reads_current_directories(tmp_path):
    with running_server(tmp_path) as (process, messages, errors):
        send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "collections-test", "version": "1.0"}}})
        receive(messages)
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                       "params": {"name": "list_collections"}})
        empty = receive(messages)
        assert empty["id"] == 2
        assert empty["result"]["structuredContent"] == {"collections": [], "total": 0}
        assert not (tmp_path / "data").exists()
        for name in ("beta", "alpha"):
            (tmp_path / "data/documents" / name).mkdir(parents=True)
        send(process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "list_collections", "arguments": {}}})
        result = receive(messages)
        assert result["id"] == 3
        assert result["result"]["structuredContent"] == {"collections": ["alpha", "beta"], "total": 2}
        assert "alpha" in result["result"]["content"][0]["text"]
        assert_clean_shutdown(process, messages)
    assert not (tmp_path / "data/db").exists()


@pytest.fixture
def knowledge_data(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import yaml

    from core.types import Chunk
    from ingestion.embedding.sparse_encoder import SparseEncoder
    from ingestion.storage.bm25_indexer import BM25Indexer
    from libs.vector_store.chroma_store import ChromaStore

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            if self.path.endswith("/embeddings"):
                response = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
            elif self.path.endswith("/chat/completions"):
                candidates = json.loads(payload["messages"][-1]["content"])["candidates"]
                ranked = [item["chunk_id"] for item in reversed(candidates)]
                response = {"choices": [{"message": {"content": json.dumps({"ranked_ids": ranked})}}]}
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

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    config = yaml.safe_load((PROJECT_ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    for section in ("embedding", "llm"):
        config[section] = {"provider": "openai", "model": "offline-test", "api_key": "fake-key",
                           "base_url": f"http://127.0.0.1:{http.server_port}/v1"}
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    config["ingestion"] = {"bm25_path": str(tmp_path / "bm25")}
    config["retrieval"]["fusion_top_k"] = 2
    config["rerank"] = {"enabled": True, "provider": "llm", "timeout_seconds": 5, "top_m": 3,
                         "prompt_path": str(PROJECT_ROOT / "config/prompts/rerank.txt")}
    store = ChromaStore(config["vector_store"])
    chunks = [Chunk(key, f"Azure setup guide {key}",
                    {"source_path": f"{key}.pdf", "collection": collection, "page": 2},
                    start_offset=0, end_offset=len(f"Azure setup guide {key}"))
              for key, collection in (("a", "alpha"), ("b", "alpha"), ("c", "beta"))]
    store.upsert([{"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata,
                   "dense_vector": vector}
                  for chunk, vector in zip(chunks, ([1.0, 0.0], [0.8, 0.2], [0.0, 1.0]))])
    BM25Indexer(config["ingestion"]["bm25_path"]).build(SparseEncoder().encode(chunks))
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    thread.start()
    try:
        yield settings_path, config, calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("rerank_enabled", [True, False])
def test_query_knowledge_hub_real_retrieval_citations_and_reranking(tmp_path, knowledge_data, rerank_enabled):
    import yaml

    settings_path, config, calls = knowledge_data
    config["rerank"]["enabled"] = rerank_enabled
    settings_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with running_server(tmp_path, arguments=("--settings", str(settings_path))) as (process, messages, errors):
        send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "query-test", "version": "1.0"}}})
        receive(messages)
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                       "params": {"name": "query_knowledge_hub",
                                  "arguments": {"query": "Azure", "top_k": 1, "collection": "alpha"}}})
        # Cold PDF/database dependency imports happen on this first query,
        # after the lightweight handshake has already completed.
        response = receive(messages, timeout=60)
        assert response["id"] == 2
        result = response["result"]
        citation = result["structuredContent"]["citations"][0]
        expected = "b" if rerank_enabled else "a"
        assert citation["chunk_id"] == expected
        assert citation["source"] == f"{expected}.pdf" and citation["page"] == 2
        assert citation["score"] > 0
        assert result["structuredContent"]["total"] == 1
        assert "[1]" in result["content"][0]["text"]
        assert "c.pdf" not in result["content"][0]["text"]
        chats = [payload for path, payload in calls if path.endswith("/chat/completions")]
        if rerank_enabled:
            assert len(json.loads(chats[0]["messages"][-1]["content"])["candidates"]) == 2
        else:
            assert not chats

        send(process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "query_knowledge_hub", "arguments": {"query": "Azure"}}})
        assert receive(messages)["result"]["structuredContent"]["total"] == 2
        send(process, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "query_knowledge_hub",
                                  "arguments": {"query": "Azure", "collection": "missing"}}})
        empty = receive(messages)["result"]
        assert empty["structuredContent"]["citations"] == []
        assert "未找到相关文档" in empty["content"][0]["text"]
        assert_clean_shutdown(process, messages)


def test_query_knowledge_hub_rejects_invalid_arguments_before_backend_initialization(tmp_path):
    with running_server(tmp_path) as (process, messages, errors):
        send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "query-test", "version": "1.0"}}})
        receive(messages)
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        for request_id, arguments in enumerate([
            {}, {"query": " "}, {"query": "!!!"}, {"query": "Azure", "top_k": 0},
            {"query": "Azure", "top_k": True}, {"query": "Azure", "collection": " "},
        ], start=2):
            send(process, {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                           "params": {"name": "query_knowledge_hub", "arguments": arguments}})
            response = receive(messages)
            assert response["id"] == request_id
            assert response["error"]["code"] == -32602
        assert_clean_shutdown(process, messages)
    assert not (tmp_path / "data").exists()
