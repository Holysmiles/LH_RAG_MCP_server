"""Generate citations from the final ranking without inventing source metadata."""

from collections.abc import Sequence
from typing import TypedDict

from core.types import RetrievalResult


class Citation(TypedDict):
    index: int
    source: str
    page: int | str | None
    chunk_id: str
    score: float


class CitationGenerator:
    def generate(self, retrieval_results: Sequence[RetrievalResult]) -> list[Citation]:
        citations = []
        for index, result in enumerate(retrieval_results, start=1):
            source = result.metadata.get("source_path") or result.metadata.get("source")
            page = result.metadata.get("page", result.metadata.get("page_number"))
            if isinstance(page, bool) or not isinstance(page, (int, str)):
                page = None
            citations.append(Citation(
                index=index,
                source=source if isinstance(source, str) and source.strip() else "未知来源",
                page=page,
                chunk_id=result.chunk_id,
                score=result.score,
            ))
        return citations
