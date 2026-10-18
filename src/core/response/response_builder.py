"""Build a text-first MCP tool result using Core's retrieval contracts."""

from collections.abc import Sequence
from typing import Any, TypedDict

from core.response.citation_generator import CitationGenerator
from core.response.multimodal_assembler import MultimodalAssembler
from core.types import RetrievalResult


class MCPResponse(TypedDict):
    content: list[dict[str, str]]
    structuredContent: dict[str, Any]
    isError: bool


class ResponseBuilder:
    def __init__(self, multimodal_assembler: MultimodalAssembler | None = None) -> None:
        self.multimodal_assembler = multimodal_assembler or MultimodalAssembler()

    def build(self, retrieval_results: Sequence[RetrievalResult], query: str) -> MCPResponse:
        results = list(retrieval_results)
        citations = CitationGenerator().generate(results)
        if not results:
            markdown = "未找到相关文档，请先运行 ingest.py 摄取数据，或检查集合筛选条件。"
        else:
            sections = ["## 检索结果", f"查询：{' '.join(query.split())}"]
            for item, citation in zip(results, citations):
                source = " ".join(citation["source"].split())
                page = citation["page"] if citation["page"] is not None else "未知"
                sections.append(
                    f"### [{citation['index']}]\n\n{item.text}\n\n"
                    f"来源：{source} | 页码：{page} | 分数：{item.score:.6f}"
                )
            markdown = "\n\n".join(sections)
        return MCPResponse(
            content=[{"type": "text", "text": markdown}, *self.multimodal_assembler.assemble(results)],
            structuredContent={"query": query, "citations": citations, "total": len(results)},
            isError=False,
        )
