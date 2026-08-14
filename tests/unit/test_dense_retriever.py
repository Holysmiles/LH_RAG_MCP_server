"""D2 DenseRetriever 的依赖编排与结果契约测试。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from core import RetrievalResult
from core.query_engine import DenseRetriever
from core.settings import Settings
from core.trace import TraceContext
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    StoredVectorRecord,
    VectorRecord,
    VectorSearchResult,
)


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: object = None) -> None:
        self.vectors = [[0.25, 0.75]] if vectors is None else vectors
        self.calls: list[tuple[list[str], object | None]] = []

    def embed(
        self,
        texts: list[str],
        trace: object | None = None,
    ) -> list[list[float]]:
        self.calls.append((list(texts), trace))
        return self.vectors  # type: ignore[return-value]


class FakeVectorStore(BaseVectorStore):
    def __init__(self, results: object = None) -> None:
        self.results = [] if results is None else results
        self.calls: list[dict[str, Any]] = []

    def upsert(
        self,
        records: Sequence[VectorRecord],
        trace: object | None = None,
    ) -> None:
        del records, trace

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        self.calls.append(
            {
                "vector": list(vector),
                "top_k": top_k,
                "filters": filters,
                "trace": trace,
            }
        )
        return self.results  # type: ignore[return-value]

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: object | None = None,
    ) -> list[StoredVectorRecord]:
        del ids, trace
        return []


def settings(dense_top_k: int = 20) -> Settings:
    return Settings(retrieval={"dense_top_k": dense_top_k})


@pytest.mark.unit
def test_retrieve_embeds_once_calls_store_and_preserves_result_order() -> None:
    embedding = FakeEmbedding()
    raw_metadata = {"source_path": "guide.pdf", "tags": ["rag"]}
    store = FakeVectorStore(
        [
            {
                "chunk_id": "chunk-2",
                "score": 0.9,
                "text": "second",
                "metadata": raw_metadata,
            },
            {
                "chunk_id": "chunk-1",
                "score": 0.8,
                "text": "first",
                "metadata": {"source_path": "other.pdf"},
            },
        ]
    )
    trace = TraceContext()
    filters = {"collection": "guides"}
    retriever = DenseRetriever(settings(), embedding, store)

    results = retriever.retrieve("  Azure vector search  ", 2, filters, trace)
    raw_metadata["tags"].append("changed")

    assert embedding.calls == [(["Azure vector search"], trace)]
    assert store.calls == [
        {
            "vector": [0.25, 0.75],
            "top_k": 2,
            "filters": filters,
            "trace": trace,
        }
    ]
    assert [item.chunk_id for item in results] == ["chunk-2", "chunk-1"]
    assert all(isinstance(item, RetrievalResult) for item in results)
    assert results[0].metadata["tags"] == ["rag"]


@pytest.mark.unit
def test_default_top_k_comes_from_settings_and_empty_results_are_valid() -> None:
    embedding = FakeEmbedding()
    store = FakeVectorStore()

    results = DenseRetriever(settings(7), embedding, store).retrieve("query")

    assert results == []
    assert store.calls[0]["top_k"] == 7
    assert store.calls[0]["filters"] is None


@pytest.mark.unit
def test_retrieval_result_is_finite_serializable_and_copies_metadata() -> None:
    metadata = {"collection": "docs", "nested": {"language": "zh"}}
    result = RetrievalResult("chunk-1", 0.75, "content", metadata)
    metadata["nested"]["language"] = "changed"

    assert result.to_dict() == {
        "chunk_id": "chunk-1",
        "score": 0.75,
        "text": "content",
        "metadata": {"collection": "docs", "nested": {"language": "zh"}},
    }


@pytest.mark.unit
@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_fails_before_provider_calls(query: str) -> None:
    embedding = FakeEmbedding()
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="query must not be empty"):
        DenseRetriever(settings(), embedding, store).retrieve(query)

    assert embedding.calls == []
    assert store.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_fails_before_provider_calls(top_k: object) -> None:
    embedding = FakeEmbedding()
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="top_k"):
        DenseRetriever(settings(), embedding, store).retrieve(
            "query", top_k=top_k  # type: ignore[arg-type]
        )

    assert embedding.calls == []
    assert store.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "returned 0 vectors"),
        ([[1.0], [2.0]], "returned 2 vectors"),
        ([[]], "must not be empty"),
        ([[1.0, float("nan")]], "must be finite"),
        ([[1.0, "bad"]], "must be numeric"),
    ],
)
def test_bad_embedding_response_never_reaches_store(
    vectors: object,
    message: str,
) -> None:
    store = FakeVectorStore()

    with pytest.raises((TypeError, ValueError), match=message):
        DenseRetriever(settings(), FakeEmbedding(vectors), store).retrieve("query")

    assert store.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([{"chunk_id": "missing"}], "missing fields"),
        (
            [
                {
                    "chunk_id": "chunk-1",
                    "score": float("inf"),
                    "text": "text",
                    "metadata": {},
                }
            ],
            r"result\[0\].*score.*finite",
        ),
        ([object()], r"result\[0\].*mapping"),
    ],
)
def test_bad_store_response_is_rejected_with_result_index(
    results: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DenseRetriever(
            settings(),
            FakeEmbedding(),
            FakeVectorStore(results),
        ).retrieve("query")


@pytest.mark.unit
def test_constructor_and_filter_types_are_validated() -> None:
    embedding = FakeEmbedding()
    store = FakeVectorStore()
    with pytest.raises(TypeError, match="settings"):
        DenseRetriever(object(), embedding, store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="embedding_client"):
        DenseRetriever(settings(), object(), store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="vector_store"):
        DenseRetriever(settings(), embedding, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="filters"):
        DenseRetriever(settings(), embedding, store).retrieve(
            "query", filters=[]  # type: ignore[arg-type]
        )
