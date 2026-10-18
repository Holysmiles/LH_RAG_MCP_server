"""Document ID lookup, metadata aggregation, and MCP error contracts."""

from copy import deepcopy
from types import SimpleNamespace

from mcp import types
from mcp.shared.exceptions import MCPError
import pytest

from core.settings import Settings
from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.tools.get_document_summary import get_document_summary, register_document_summary_tool


pytestmark = pytest.mark.unit


def record(key, index, **metadata):
    return {"id": key, "text": "Raw text must not be substituted for saved summaries.",
            "metadata": {"document_id": "document-1", "chunk_index": index, **metadata}}


def test_aggregates_saved_metadata_in_chunk_order_without_mutating_records():
    records = [record("a", 2, title="Later", summary="Second part", tags=["RAG", "python"]),
               record("z", 0, title="标题", summary="First part", tags=["RAG", "中文"]),
               record("b", 1, title="标题", summary="First part", tags=["RAG"])]
    before = deepcopy(records)
    filters = []
    store = SimpleNamespace(get_by_metadata=lambda value: filters.append(value) or records)

    result = get_document_summary(" document-1 ", settings=Settings(), vector_store=store)

    assert filters == [{"document_id": "document-1"}]
    assert result["structuredContent"] == {
        "doc_id": "document-1", "title": "标题", "summary": "First part\n\nSecond part",
        "tags": ["RAG", "中文", "python"], "chunk_count": 3, "summary_source": "chunk_metadata",
    }
    assert types.CallToolResult.model_validate(result).is_error is False
    assert "First part" in result["content"][0]["text"]
    assert records == before


def test_absent_document_returns_standard_mcp_tool_error():
    result = get_document_summary("absent", settings=Settings(),
                                  vector_store=SimpleNamespace(get_by_metadata=lambda filters: []))
    assert types.CallToolResult.model_validate(result).is_error is True
    assert result["structuredContent"]["error"]["code"] == "document_not_found"
    assert "未找到该文档" in result["content"][0]["text"]


def test_missing_metadata_is_explicit_and_malformed_tags_are_ignored():
    records = [record("b", None, title=7, summary=None, tags="not-a-tag-list"),
               record("a", None, tags=[" tag ", None, "", "tag"])]
    result = get_document_summary("document-1", settings=Settings(),
                                  vector_store=SimpleNamespace(get_by_metadata=lambda filters: records))
    assert result["structuredContent"]["title"] == "未提供标题"
    assert result["structuredContent"]["summary"] == ""
    assert result["structuredContent"]["tags"] == ["tag"]
    assert "暂无已保存的摘要" in result["content"][0]["text"]
    assert "Raw text" not in result["content"][0]["text"]


def test_backend_cannot_return_another_documents_metadata():
    store = SimpleNamespace(get_by_metadata=lambda filters: [record("a", 0, document_id="other")])
    with pytest.raises(RuntimeError, match="another document"):
        get_document_summary("document-1", settings=Settings(), vector_store=store)


@pytest.mark.parametrize("doc_id", [None, "", " ", 42])
def test_invalid_id_fails_before_opening_storage(doc_id):
    with pytest.raises(ValueError, match="doc_id"):
        get_document_summary(doc_id, settings=Settings())


def test_backend_failure_is_not_reported_as_missing_document():
    def fail(filters):
        raise OSError("database unavailable")

    with pytest.raises(OSError):
        get_document_summary("document-1", settings=Settings(),
                             vector_store=SimpleNamespace(get_by_metadata=fail))


@pytest.mark.anyio
async def test_registration_rejects_invalid_arguments_before_storage_access():
    handler = ProtocolHandler("test", "1.0")
    register_document_summary_tool(handler, Settings())
    assert handler.handle_tools_list().tools[0].name == "get_document_summary"
    for arguments in ({}, {"doc_id": " "}, {"doc_id": 1}, {"doc_id": "x", "other": 1}):
        with pytest.raises(MCPError) as caught:
            await handler.handle_tools_call("get_document_summary", arguments)
        assert caught.value.code == -32602


@pytest.fixture
def anyio_backend():
    return "asyncio"
