"""C15 CLI 端到端测试：真实子进程、真实存储、本地 Embedding 协议桩。"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).parents[2]
INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "ingest.py"
SIMPLE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"


class EmbeddingStubHandler(BaseHTTPRequestHandler):
    """实现测试所需的最小 OpenAI ``/embeddings`` 协议。"""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler contract
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(payload)
        texts = payload["input"]
        response = {
            "data": [
                {
                    "index": index,
                    "embedding": [
                        float(len(text.encode("utf-8")) + 1),
                        float(sum(ord(char) for char in text) % 997 + 1),
                        1.0,
                    ],
                }
                for index, text in enumerate(texts)
            ]
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def embedding_server() -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    EmbeddingStubHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingStubHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1", EmbeddingStubHandler.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def write_settings(tmp_path: Path, embedding_base_url: str) -> Path:
    settings_path = tmp_path / "settings.yaml"
    config = {
        "project": {"name": "c15-e2e", "environment": "test"},
        "llm": {"provider": "openai", "model": "unused", "api_key": "test"},
        "vision_llm": {"enabled": False},
        "embedding": {
            "provider": "openai",
            "model": "local-stub",
            "api_key": "test",
            "base_url": embedding_base_url,
            "timeout": 5,
        },
        "splitter": {
            "provider": "recursive",
            "chunk_size": 300,
            "chunk_overlap": 30,
        },
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "data" / "db" / "chroma"),
            "collection_name": "cli_e2e",
        },
        "retrieval": {
            "dense_top_k": 20,
            "sparse_top_k": 20,
            "fusion_top_k": 10,
        },
        "rerank": {"provider": "none"},
        "evaluation": {"provider": "custom"},
        "observability": {"log_level": "INFO"},
        "ingestion": {
            "batch_size": 2,
            "integrity_db_path": str(tmp_path / "data" / "db" / "history.db"),
            "bm25_path": str(tmp_path / "data" / "db" / "bm25"),
            "image_root": str(tmp_path / "data" / "images"),
            "image_index_db_path": str(
                tmp_path / "data" / "db" / "image_index.db"
            ),
        },
    }
    settings_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return settings_path


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, str(INGEST_SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )


def parse_single_result(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert process.returncode == 0, process.stderr
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, process.stdout
    return json.loads(lines[0])


@pytest.mark.e2e
def test_cli_creates_indexes_skips_unchanged_and_force_reprocesses(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    shutil.copy2(SIMPLE_PDF, documents / "guide.pdf")

    with embedding_server() as (base_url, requests):
        settings_path = write_settings(tmp_path, base_url)
        arguments = (
            "--path",
            str(documents),
            "--collection",
            "cli_docs",
            "--settings",
            str(settings_path),
        )

        first = parse_single_result(run_cli(*arguments))
        request_count = len(requests)
        second = parse_single_result(run_cli(*arguments))
        forced = parse_single_result(run_cli(*arguments, "--force"))

    assert first["status"] == "success"
    assert first["collection"] == "cli_docs"
    assert first["chunk_count"] > 0
    assert request_count > 0
    assert second["status"] == "skipped"
    assert len(requests) > request_count
    assert forced["status"] == "success"
    assert forced["record_ids"] == first["record_ids"]

    assert (tmp_path / "data" / "db" / "history.db").is_file()
    assert (tmp_path / "data" / "db" / "bm25" / "index.json").is_file()
    assert any((tmp_path / "data" / "db" / "chroma").iterdir())


@pytest.mark.e2e
def test_cli_reports_invalid_path_with_usage_exit_code(tmp_path: Path) -> None:
    with embedding_server() as (base_url, _):
        settings_path = write_settings(tmp_path, base_url)
        process = run_cli(
            "--path",
            str(tmp_path / "missing"),
            "--settings",
            str(settings_path),
        )

    assert process.returncode == 2
    assert "ingestion path does not exist" in process.stderr
