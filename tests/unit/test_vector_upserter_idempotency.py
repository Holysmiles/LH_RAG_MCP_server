"""Contract tests for stable vector IDs and aligned upserts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from core import Chunk
from core.settings import Settings
from ingestion.storage import VectorUpserter
from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    StoredVectorRecord,
    VectorRecord,
    VectorSearchResult,
)


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.calls: list[list[VectorRecord]] = []

    def upsert(self, records: Sequence[VectorRecord], trace: object | None = None) -> None:
        self.calls.append(list(records))

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, object] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        return []

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: object | None = None,
    ) -> list[StoredVectorRecord]:
        del ids, trace
        return []


def settings() -> Settings:
    return Settings(vector_store={"provider": "chroma", "persist_directory": "./data/db/test"})


def make_chunk(text: str = "same content", index: int = 0) -> Chunk:
    return Chunk(
        id=f"source-id-{index}",
        text=text,
        metadata={"source_path": "reports/source.pdf", "chunk_index": index},
        start_offset=index * 10,
        end_offset=index * 10 + len(text),
        source_ref="document-1",
    )


@pytest.mark.unit
def test_same_chunk_is_idempotent_and_content_change_changes_id() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(settings(), store=store)
    first = upserter.upsert([make_chunk()], [[0.1, 0.2]])
    second = upserter.upsert([make_chunk()], [[0.3, 0.4]])
    changed = upserter.upsert([make_chunk("changed content")], [[0.3, 0.4]])

    assert first[0].id == second[0].id
    assert first[0].id != changed[0].id
    assert len(store.calls) == 3
    assert store.calls[0][0]["id"] == store.calls[1][0]["id"]


@pytest.mark.unit
def test_batch_upsert_preserves_order_and_metadata_contract() -> None:
    store = FakeVectorStore()
    records = VectorUpserter(settings(), store=store).upsert(
        [make_chunk("one", 0), make_chunk("two", 1)], [[1.0, 0.0], [0.0, 1.0]]
    )

    assert [record.metadata["chunk_index"] for record in records] == [0, 1]
    assert [record["id"] for record in store.calls[0]] == [record.id for record in records]
    assert records[0].metadata["source_ref"] == "document-1"
    assert records[0].dense_vector == [1.0, 0.0]


@pytest.mark.unit
def test_alignment_and_vector_validation_happen_before_store_call() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(settings(), store=store)
    with pytest.raises(ValueError, match="same length"):
        upserter.upsert([make_chunk()], [])
    with pytest.raises(ValueError, match="same dimension"):
        upserter.upsert([make_chunk("one", 0), make_chunk("two", 1)], [[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        upserter.upsert([make_chunk()], [[float("nan")]])
    assert store.calls == []


@pytest.mark.unit
def test_missing_chunk_identity_fields_are_rejected() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(settings(), store=store)
    chunk = make_chunk()
    chunk.metadata.pop("chunk_index")
    with pytest.raises(ValueError, match="chunk_index"):
        upserter.upsert([chunk], [[1.0]])


@pytest.mark.unit
def test_source_path_separator_does_not_change_stable_id() -> None:
    forward = make_chunk()
    backward = make_chunk()
    backward.metadata["source_path"] = "reports\\source.pdf"

    assert VectorUpserter.generate_chunk_id(forward) == (
        VectorUpserter.generate_chunk_id(backward)
    )
    assert len(VectorUpserter.generate_chunk_id(forward)) == len("chunk_") + 64


@pytest.mark.unit
def test_empty_batch_does_not_call_vector_store() -> None:
    store = FakeVectorStore()

    records = VectorUpserter(settings(), store=store).upsert([], [])

    assert records == []
    assert store.calls == []
