"""Final ranking, citation metadata, and MCP response shape stay aligned."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from mcp import types
import pytest

from core.response.citation_generator import CitationGenerator
from core.response.response_builder import ResponseBuilder
from core.settings import load_settings
from core.types import RetrievalResult
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub


pytestmark = pytest.mark.unit


def result(chunk_id="b", **metadata):
    return RetrievalResult(chunk_id, 0.125, f"正文 {chunk_id}\n\n```python\nx = 1\n```", metadata)


def test_markdown_and_structured_citations_follow_final_ranking():
    results = [result("b", source_path="指南.pdf", page=3),
               result("a", source_path="manual.pdf", page_number=9)]
    original = deepcopy(results)
    response = ResponseBuilder().build(results, "配置查询")
    validated = types.CallToolResult.model_validate(response)
    text = validated.content[0].text
    assert "[1]" in text and "[2]" in text
    assert text.index("正文 b") < text.index("正文 a")
    assert "指南.pdf" in text and "页码：3" in text
    assert "```python\nx = 1\n```" in text
    assert response["structuredContent"]["citations"] == [
        {"index": 1, "source": "指南.pdf", "page": 3, "chunk_id": "b", "score": 0.125},
        {"index": 2, "source": "manual.pdf", "page": 9, "chunk_id": "a", "score": 0.125},
    ]
    assert response["structuredContent"]["total"] == 2
    assert results == original


def test_empty_results_return_a_friendly_text_block():
    response = ResponseBuilder().build([], "missing")
    assert response["content"][0]["type"] == "text"
    assert "未找到相关文档" in response["content"][0]["text"]
    assert response["structuredContent"] == {"query": "missing", "citations": [], "total": 0}
    assert response["isError"] is False


def test_missing_citation_metadata_is_explicit_without_inventing_a_page():
    citations = CitationGenerator().generate([result(), result("a", source="legacy.pdf")])
    assert citations[0]["source"] == "未知来源"
    assert citations[0]["page"] is None
    assert citations[1]["source"] == "legacy.pdf"
    assert citations[1]["page"] is None


@pytest.fixture
def settings():
    settings = load_settings(Path(__file__).resolve().parents[2] / "config/settings.yaml")
    settings.observability["trace_enabled"] = False
    return settings


def test_query_preserves_candidates_until_after_reranking(settings):
    settings.retrieval["fusion_top_k"] = 2
    settings.rerank["top_m"] = 4
    candidates = [result("a"), result("b"), result("c")]
    calls = []

    def search(query, **kwargs):
        calls.append(("search", query, kwargs))
        return candidates

    def rerank(query, items, top_k, trace=None):
        calls.append(("rerank", query, items, top_k))
        return list(reversed(items))[:top_k]

    response = query_knowledge_hub(
        " Azure ", top_k=1, collection=" alpha ", settings=settings,
        hybrid_search=SimpleNamespace(search=search),
        reranker=SimpleNamespace(enabled=True, provider="llm", rerank=rerank),
    )
    assert calls[0] == ("search", "Azure", {"top_k": 4, "filters": {"collection": "alpha"}, "trace": None})
    assert calls[1] == ("rerank", "Azure", candidates, 1)
    assert response["structuredContent"]["citations"][0]["chunk_id"] == "c"


def test_query_uses_configured_default_count(settings):
    settings.retrieval["fusion_top_k"] = 1
    calls = []
    hybrid = SimpleNamespace(search=lambda query, **kwargs: [result("a"), result("b")])
    reranker = SimpleNamespace(enabled=False, provider="none",
                              rerank=lambda query, items, top_k, trace=None: calls.append(top_k) or items[:top_k])
    response = query_knowledge_hub("Azure", settings=settings, hybrid_search=hybrid, reranker=reranker)
    assert calls == [1]
    assert response["structuredContent"]["total"] == 1


@pytest.mark.parametrize("arguments", [
    {"query": " "}, {"query": "Azure", "top_k": 0},
    {"query": "Azure", "top_k": True}, {"query": "Azure", "collection": " "},
])
def test_invalid_direct_calls_fail_before_opening_backends(settings, arguments):
    with pytest.raises(ValueError):
        query_knowledge_hub(**arguments, settings=settings)
