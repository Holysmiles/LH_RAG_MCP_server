"""Filesystem collection discovery and its MCP registration contract."""

import pytest
from mcp import types
from mcp.shared.exceptions import MCPError

from core.settings import Settings
from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.tools.list_collections import list_collections, register_list_collections_tool


pytestmark = pytest.mark.unit


def test_lists_immediate_directories_including_empty_collections(tmp_path):
    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha" / "nested").mkdir(parents=True)
    (tmp_path / "中文资料").mkdir()
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "guide.pdf").write_bytes(b"placeholder")

    response = list_collections(tmp_path)

    assert response["structuredContent"] == {"collections": ["alpha", "zeta", "中文资料"], "total": 3}
    assert "- alpha" in response["content"][0]["text"]
    assert "中文资料" in response["content"][0]["text"]
    assert types.CallToolResult.model_validate(response).is_error is False


@pytest.mark.parametrize("create_root", [True, False])
def test_missing_or_empty_root_returns_friendly_empty_list_without_creating_it(tmp_path, create_root):
    root = tmp_path / "documents"
    if create_root:
        root.mkdir()

    response = list_collections(root)

    assert response["structuredContent"] == {"collections": [], "total": 0}
    assert "暂无文档集合" in response["content"][0]["text"]
    assert root.exists() is create_root


def test_default_path_uses_working_directory_and_does_not_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/documents"
    assert list_collections()["structuredContent"]["collections"] == []
    (root / "new").mkdir(parents=True)
    assert list_collections()["structuredContent"]["collections"] == ["new"]
    (root / "new").rmdir()
    assert list_collections()["structuredContent"]["collections"] == []


def test_file_as_root_is_an_error_rather_than_an_empty_library(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("file", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        list_collections(root)


@pytest.mark.anyio
async def test_registration_honors_configured_root_and_rejects_parameters(tmp_path):
    (tmp_path / "configured").mkdir()
    handler = ProtocolHandler("test", "1.0")
    register_list_collections_tool(handler, Settings(ingestion={"documents_path": str(tmp_path)}))
    assert handler.handle_tools_list().tools[0].name == "list_collections"
    response = await handler.handle_tools_call("list_collections")
    assert response.structured_content == {"collections": ["configured"], "total": 1}
    with pytest.raises(MCPError) as caught:
        await handler.handle_tools_call("list_collections", {"documents_path": "elsewhere"})
    assert caught.value.code == -32602


@pytest.fixture
def anyio_backend():
    return "asyncio"
