"""Summarize persisted chunk metadata for an ingestion document ID."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.response.response_builder import MCPResponse
from core.settings import Settings

if TYPE_CHECKING:
    from libs.vector_store.base_vector_store import BaseVectorStore
    from mcp_server.protocol_handler import ProtocolHandler


def get_document_summary(
    doc_id: str, *, settings: Settings, vector_store: BaseVectorStore | None = None,
) -> MCPResponse:
    """Merge saved summaries and tags in chunk order, without generating facts.

    ``doc_id`` is the ``document_id`` reported by ingestion, not a chunk ID.
    Missing summary metadata remains explicitly unavailable. No model is used.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    doc_id = doc_id.strip()
    if vector_store is None:
        from libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(settings)
    records = vector_store.get_by_metadata({"document_id": doc_id})
    if not records:
        return MCPResponse(
            content=[{"type": "text", "text": "未找到该文档，请检查摄取返回的 document_id。"}],
            structuredContent={"doc_id": doc_id, "error": {"code": "document_not_found", "message": "Document not found"}},
            isError=True,
        )
    # Defend the tool boundary even for third-party provider implementations.
    if any(record["metadata"].get("document_id") != doc_id for record in records):
        raise RuntimeError("Document metadata lookup returned another document")

    def position(record):
        index = record["metadata"].get("chunk_index")
        valid = isinstance(index, int) and not isinstance(index, bool) and index >= 0
        return (index if valid else float("inf"), record["id"])

    records = sorted(records, key=position)
    titles, summaries, tags = [], [], []
    for record in records:
        metadata = record["metadata"]
        for key, target in (("title", titles), ("summary", summaries)):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in target:
                target.append(value.strip())
        values = metadata.get("tags", [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip() and value.strip() not in tags:
                    tags.append(value.strip())
    title = titles[0] if titles else "未提供标题"
    summary = "\n\n".join(summaries)
    text = f"## {title}\n\n{summary or '暂无已保存的摘要。'}\n\n标签：{', '.join(tags) or '无'}"
    return MCPResponse(
        content=[{"type": "text", "text": text}],
        structuredContent={"doc_id": doc_id, "title": title, "summary": summary,
                           "tags": tags, "chunk_count": len(records), "summary_source": "chunk_metadata"},
        isError=False,
    )


def register_document_summary_tool(handler: ProtocolHandler, settings: Settings) -> None:
    def call(arguments):
        return get_document_summary(**arguments, settings=settings)

    handler.register_tool(
        "get_document_summary",
        "Get saved title, summary and tags by the document_id returned by ingestion (not a chunk ID).",
        {"type": "object", "properties": {"doc_id": {"type": "string", "minLength": 1, "pattern": r"\S"}},
         "required": ["doc_id"], "additionalProperties": False},
        call,
    )
