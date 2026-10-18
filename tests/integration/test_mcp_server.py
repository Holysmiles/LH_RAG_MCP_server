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


def receive(messages):
    try:
        line = messages.get(timeout=20)
    except Empty:
        pytest.fail("MCP server did not respond within 20 seconds")
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
def noisy_create():
    print("factory diagnostic")
    instance = original_create()
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
