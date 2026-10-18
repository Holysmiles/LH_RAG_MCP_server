"""List source-document collection directories without opening retrieval stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.response.response_builder import MCPResponse
from core.settings import Settings

if TYPE_CHECKING:
    from mcp_server.protocol_handler import ProtocolHandler


DEFAULT_DOCUMENTS_PATH = Path("data/documents")


def list_collections(documents_path: str | Path = DEFAULT_DOCUMENTS_PATH) -> MCPResponse:
    """List immediate child directories in stable order, including empty ones.

    Directory presence describes source organization, not ingestion success.
    Each call reads the filesystem again so new collections appear immediately.
    """
    root = Path(documents_path)
    try:
        names = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    except FileNotFoundError:
        names = []
    text = ("## 文档集合\n\n" + "\n".join(f"- {name}" for name in names)
            if names else "暂无文档集合，请先在文档目录下按集合整理原始文档。")
    return MCPResponse(
        content=[{"type": "text", "text": text}],
        structuredContent={"collections": names, "total": len(names)},
        isError=False,
    )


def register_list_collections_tool(handler: ProtocolHandler, settings: Settings) -> None:
    documents_path = settings.ingestion.get("documents_path", DEFAULT_DOCUMENTS_PATH)

    def call(arguments):
        return list_collections(documents_path)

    handler.register_tool(
        "list_collections",
        "List collection directories under the configured source documents path; does not imply ingestion status.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        call,
    )
