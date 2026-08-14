"""D3 SparseRetriever 的 BM25 排名与正文补全契约测试。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from core import Chunk, RetrievalResult
from core.query_engine import SparseRetriever
from core.settings import Settings
from core.trace import TraceContext
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    StoredVectorRecord,
    VectorRecord,
    VectorSearchResult,
)


class FakeVectorStore(BaseVectorStore):
    def __init__(self, records: object = None) -> None:
        self.records = [] if records is None else records
        self.calls: list[tuple[list[str], object | None]] = []

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
        del vector, top_k, filters, trace
        return []

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: object | None = None,
    ) -> list[StoredVectorRecord]:
        self.calls.append((list(ids), trace))
        return self.records  # type: ignore[return-value]


def settings(sparse_top_k: int = 20) -> Settings:
    return Settings(retrieval={"sparse_top_k": sparse_top_k})


def build_index(tmp_path: Any) -> BM25Indexer:
    chunks = [
        Chunk(
            id="chunk-rag",
            text="hybrid retrieval combines dense sparse signals",
            metadata={"source_path": "rag.pdf"},
            start_offset=0,
            end_offset=45,
        ),
        Chunk(
            id="chunk-vector",
            text="vector database stores embeddings",
            metadata={"source_path": "vector.pdf"},
            start_offset=0,
            end_offset=33,
        ),
        Chunk(
            id="chunk-other",
            text="unrelated cooking notes",
            metadata={"source_path": "other.pdf"},
            start_offset=0,
            end_offset=23,
        ),
    ]
    indexer = BM25Indexer(tmp_path / "bm25")
    indexer.build(SparseEncoder().encode(chunks))
    return indexer


@pytest.mark.unit
def test_real_bm25_hits_expected_id_and_restores_ranked_content(tmp_path: Any) -> None:
    indexer = build_index(tmp_path)
    trace = TraceContext()
    keywords = ["hybrid", "vector"]
    expected_hits = indexer.query(" ".join(keywords), top_k=2)
    records = {
        "chunk-vector": {
            "id": "chunk-vector",
            "text": "vector text",
            "metadata": {"source_path": "vector.pdf"},
        },
        "chunk-rag": {
            "id": "chunk-rag",
            "text": "rag text",
            "metadata": {"source_path": "rag.pdf", "tags": ["rag"]},
        },
    }
    # 故意按 BM25 命中的反序返回，只验证“保留索引排名”这个核心不变量。
    store = FakeVectorStore(
        [records[hit["chunk_id"]] for hit in reversed(expected_hits)]
    )

    results = SparseRetriever(settings(), indexer, store).retrieve(
        keywords,
        top_k=2,
        trace=trace,
    )

    expected_ids = [hit["chunk_id"] for hit in expected_hits]
    assert [item.chunk_id for item in results] == expected_ids
    assert [item.text for item in results] == [
        records[chunk_id]["text"] for chunk_id in expected_ids
    ]
    assert all(isinstance(item, RetrievalResult) for item in results)
    assert store.calls == [(expected_ids, trace)]


@pytest.mark.unit
def test_missing_vector_record_is_skipped_without_changing_other_scores(
    tmp_path: Any,
) -> None:
    indexer = build_index(tmp_path)
    expected = indexer.query("hybrid vector", top_k=2)
    store = FakeVectorStore(
        [
            {
                "id": "chunk-vector",
                "text": "vector text",
                "metadata": {"source_path": "vector.pdf"},
            }
        ]
    )

    results = SparseRetriever(settings(), indexer, store).retrieve(
        ["hybrid", "vector"], top_k=2
    )

    assert [item.chunk_id for item in results] == ["chunk-vector"]
    assert results[0].score == next(
        item["score"] for item in expected if item["chunk_id"] == "chunk-vector"
    )


@pytest.mark.unit
def test_empty_keywords_or_no_hits_do_not_call_vector_store(tmp_path: Any) -> None:
    store = FakeVectorStore()
    retriever = SparseRetriever(settings(), build_index(tmp_path), store)

    assert retriever.retrieve([]) == []
    assert retriever.retrieve(["not-in-index"]) == []
    assert store.calls == []


@pytest.mark.unit
def test_default_top_k_comes_from_settings(tmp_path: Any) -> None:
    indexer = build_index(tmp_path)
    store = FakeVectorStore([])
    retriever = SparseRetriever(settings(1), indexer, store)

    retriever.retrieve(["hybrid", "vector"])

    assert len(store.calls[0][0]) == 1


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected_before_queries(tmp_path: Any, top_k: object) -> None:
    store = FakeVectorStore()
    with pytest.raises(ValueError, match="top_k"):
        SparseRetriever(settings(), build_index(tmp_path), store).retrieve(
            ["hybrid"], top_k=top_k  # type: ignore[arg-type]
        )
    assert store.calls == []


@pytest.mark.unit
def test_invalid_keyword_and_constructor_inputs_are_rejected(tmp_path: Any) -> None:
    indexer = build_index(tmp_path)
    store = FakeVectorStore()
    retriever = SparseRetriever(settings(), indexer, store)

    with pytest.raises(TypeError, match="keywords"):
        retriever.retrieve("hybrid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"keywords\[0\]"):
        retriever.retrieve([1])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="settings"):
        SparseRetriever(object(), indexer, store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bm25_indexer"):
        SparseRetriever(settings(), object(), store)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"id": "chunk-rag"}], "missing fields"),
        (
            [
                {"id": "unknown", "text": "x", "metadata": {}},
            ],
            "unrequested",
        ),
        (
            [
                {"id": "chunk-rag", "text": "x", "metadata": {}},
                {"id": "chunk-rag", "text": "x", "metadata": {}},
            ],
            "duplicate",
        ),
    ],
)
def test_bad_vector_store_records_are_rejected(
    tmp_path: Any,
    records: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SparseRetriever(
            settings(),
            build_index(tmp_path),
            FakeVectorStore(records),
        ).retrieve(["hybrid"], top_k=2)
