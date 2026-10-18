"""Trace persistence remains one JSON object per line and never uses stdout."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging

import pytest

from core.settings import Settings
from core.trace import TraceCollector, TraceContext
from observability.logger import JSONFormatter, get_logger, get_trace_logger, write_trace


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def close_trace_files():
    yield
    # Release Windows file handles before pytest removes its temporary folders.
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("observability.trace.") and isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


def payload(trace_type="query"):
    trace = TraceContext(trace_type)
    trace.record_stage("load", document="中文.pdf", note="first\nsecond")
    trace.finish()
    return trace.to_dict()


def test_default_path_appends_utf8_json_lines_without_console_output(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    first, second = payload(), payload("ingestion")
    write_trace(first)
    write_trace(second)
    text = (tmp_path / "logs/traces.jsonl").read_text(encoding="utf-8")
    assert "中文.pdf" in text
    assert len(text.splitlines()) == 2
    assert [json.loads(line) for line in text.splitlines()] == [first, second]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_settings_path_and_explicit_override_are_independent(tmp_path):
    first, second = tmp_path / "a/traces.jsonl", tmp_path / "b/traces.jsonl"
    settings = Settings(observability={"trace_file": str(first), "trace_enabled": True})
    write_trace(payload(), settings)
    write_trace(payload("ingestion"), settings, path=second)
    assert json.loads(first.read_text(encoding="utf-8"))["trace_type"] == "query"
    assert json.loads(second.read_text(encoding="utf-8"))["trace_type"] == "ingestion"
    assert get_trace_logger(settings) is not get_trace_logger(path=second)


def test_repeated_configuration_adds_only_one_handler(tmp_path):
    path = tmp_path / "traces.jsonl"
    first = get_trace_logger(path=path)
    assert get_trace_logger(path=path.parent / "." / path.name) is first
    assert len(first.handlers) == 1 and first.propagate is False
    write_trace(payload(), path=path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_disabled_tracing_creates_no_files_and_can_be_reenabled(tmp_path):
    path = tmp_path / "nested/traces.jsonl"
    settings = Settings(observability={"trace_file": str(path), "trace_enabled": False})
    write_trace(payload(), settings)
    get_trace_logger(settings).info(payload())
    assert not path.parent.exists()
    settings.observability["trace_enabled"] = True
    write_trace(payload(), settings)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_concurrent_writes_produce_complete_unique_lines(tmp_path):
    path = tmp_path / "concurrent.jsonl"
    traces = [payload() for _ in range(30)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda trace: write_trace(trace, path=path), traces))
    stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == 30
    assert {item["trace_id"] for item in stored} == {item["trace_id"] for item in traces}


@pytest.mark.parametrize("bad", [None, {}, {"trace_type": "other"},
                                 {"trace_type": "query", "data": object()},
                                 {"trace_type": "query", "data": float("nan")}])
def test_bad_trace_does_not_open_a_file(tmp_path, bad):
    path = tmp_path / "nested/traces.jsonl"
    with pytest.raises((TypeError, ValueError)):
        write_trace(bad, path=path)
    assert not path.parent.exists()


def test_formatter_formats_regular_messages_and_escaped_exception_text():
    record = logging.LogRecord("test", logging.WARNING, __file__, 1, "hello %s\nnext", ("world",), None)
    formatted = JSONFormatter().format(record)
    assert len(formatted.splitlines()) == 1
    assert json.loads(formatted)["message"] == "hello world\nnext"
    try:
        raise ValueError("failure\nsecond line")
    except ValueError:
        import sys
        record.exc_info = sys.exc_info()
    encoded = JSONFormatter().format(record)
    assert len(encoded.splitlines()) == 1
    assert "ValueError" in json.loads(encoded)["exception"]


def test_collector_uses_real_writer_and_propagates_write_failures(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    trace = TraceContext("ingestion")
    collector = TraceCollector(partial(write_trace, path=path))
    collector.collect(trace)
    assert json.loads(path.read_text(encoding="utf-8")) == trace.to_dict()
    handler = get_trace_logger(path=path).handlers[0]

    class BrokenStream:
        def write(self, message):
            raise OSError("disk full")

        def flush(self):
            pass

    original_stream = handler.stream
    monkeypatch.setattr(handler, "stream", BrokenStream())
    try:
        with pytest.raises(OSError, match="disk full"):
            collector.collect(trace)
    finally:
        handler.stream = original_stream


def test_regular_logger_keeps_stderr_behavior(capsys):
    logger = get_logger("test-jsonl-stderr")
    logger.info("regular diagnostic")
    captured = capsys.readouterr()
    assert captured.out == "" and "regular diagnostic" in captured.err


@pytest.mark.parametrize("config", [{"trace_file": " "}, {"trace_file": 3}, {"trace_enabled": "false"}])
def test_invalid_trace_settings_fail_clearly(config):
    with pytest.raises(ValueError, match="observability.trace_"):
        get_trace_logger(Settings(observability=config))
