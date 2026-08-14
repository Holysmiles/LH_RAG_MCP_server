"""D1 QueryProcessor 与 ProcessedQuery 的行为契约。"""

from __future__ import annotations

import math
from typing import Any

import pytest

from core import ProcessedQuery
from core.query_engine import QueryProcessor


@pytest.mark.unit
def test_mixed_language_query_keeps_semantics_and_extracts_sparse_terms() -> None:
    processor = QueryProcessor()

    result = processor.process("  How to configure Azure OpenAI 和 向量数据库？  ")

    assert isinstance(result, ProcessedQuery)
    assert result.query == "How to configure Azure OpenAI 和 向量数据库？"
    assert result.original_query == result.query
    assert result.semantic_query == result.query
    assert result.keywords == ["configure", "azure", "openai", "向量数据库"]
    assert result.filters == {}


@pytest.mark.unit
def test_keywords_are_casefolded_deduplicated_and_keep_numbers() -> None:
    result = QueryProcessor().process("RAG rag RAG 2026 3.14")

    assert result.keywords == ["rag", "2026", "3.14"]


@pytest.mark.unit
def test_stopword_only_query_falls_back_to_non_empty_keywords() -> None:
    result = QueryProcessor().process("the and")

    assert result.keywords == ["the", "and"]


@pytest.mark.unit
def test_custom_stopwords_extend_defaults() -> None:
    result = QueryProcessor(stopwords=["Azure"]).process(
        "How Azure retrieval works"
    )

    assert result.keywords == ["retrieval", "works"]


@pytest.mark.unit
def test_filters_are_deep_copied_and_serializable() -> None:
    filters: dict[str, Any] = {
        "collection": "guides",
        "time_range": {"from": "2026-01-01"},
        "tags": ["rag", "azure"],
    }

    result = QueryProcessor().process("hybrid retrieval", filters)
    filters["time_range"]["from"] = "changed"
    filters["tags"].append("changed")

    assert result.filters == {
        "collection": "guides",
        "time_range": {"from": "2026-01-01"},
        "tags": ["rag", "azure"],
    }
    assert result.to_dict() == {
        "query": "hybrid retrieval",
        "keywords": ["hybrid", "retrieval"],
        "filters": result.filters,
    }


@pytest.mark.unit
@pytest.mark.parametrize("query", ["", "   ", "?!。"])
def test_empty_or_non_searchable_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query must"):
        QueryProcessor().process(query)


@pytest.mark.unit
def test_invalid_public_inputs_are_rejected() -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        QueryProcessor().process(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="filters must be"):
        QueryProcessor().process("query", ["not", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"stopwords\[1\]"):
        QueryProcessor(stopwords=["valid", 2])  # type: ignore[list-item]


@pytest.mark.unit
@pytest.mark.parametrize(
    "filters",
    [
        {"": "bad-key"},
        {1: "bad-key-type"},
        {"score": math.nan},
        {"value": object()},
    ],
)
def test_invalid_filter_contract_is_rejected(filters: dict[Any, Any]) -> None:
    with pytest.raises(ValueError, match="ProcessedQuery.filters"):
        QueryProcessor().process("valid query", filters)
