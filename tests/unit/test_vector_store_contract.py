"""Contract tests for provider-independent vector stores and their factory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from core.settings import Settings
from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    VectorSearchResult,
)
from libs.vector_store.vector_store_factory import VectorStoreFactory


class InMemoryVectorStore(BaseVectorStore):
    """Small deterministic backend used only to verify the shared contract."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        records: Sequence[VectorRecord],
        trace: object | None = None,
    ) -> None:
        required_fields = {"id", "text", "metadata", "dense_vector"}
        for record in records:
            missing = required_fields.difference(record)
            if missing:
                raise ValueError(f"Vector record is missing fields: {sorted(missing)}")
            self.records[str(record["id"])] = dict(record)

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidates = [
            record
            for record in self.records.values()
            if self._matches_filters(record["metadata"], filters)
        ]
        ranked = sorted(
            candidates,
            key=lambda record: self._dot_product(vector, record["dense_vector"]),
            reverse=True,
        )
        return [
            {
                "chunk_id": str(record["id"]),
                "score": self._dot_product(vector, record["dense_vector"]),
                "text": str(record["text"]),
                "metadata": dict(record["metadata"]),
            }
            for record in ranked[:top_k]
        ]

    @staticmethod
    def _matches_filters(
        metadata: Mapping[str, Any], filters: Mapping[str, Any] | None
    ) -> bool:
        return filters is None or all(
            metadata.get(key) == value for key, value in filters.items()
        )

    @staticmethod
    def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right, strict=True)))


@pytest.fixture
def registered_memory_provider() -> Iterator[None]:
    VectorStoreFactory.register("memory", InMemoryVectorStore)
    yield
    VectorStoreFactory.unregister("memory")


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    return [
        {
            "id": "chunk-1",
            "text": "alpha document",
            "metadata": {"collection": "docs", "source_path": "alpha.pdf"},
            "dense_vector": [1.0, 0.0],
        },
        {
            "id": "chunk-2",
            "text": "beta document",
            "metadata": {"collection": "notes", "source_path": "beta.pdf"},
            "dense_vector": [0.0, 1.0],
        },
    ]


@pytest.mark.unit
def test_base_vector_store_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseVectorStore()


@pytest.mark.unit
def test_upsert_accepts_the_required_record_shape(
    sample_records: list[dict[str, Any]],
) -> None:
    store = InMemoryVectorStore({})

    store.upsert(sample_records)

    assert set(store.records) == {"chunk-1", "chunk-2"}
    assert store.records["chunk-1"]["text"] == "alpha document"


@pytest.mark.unit
def test_upsert_rejects_records_with_missing_fields() -> None:
    store = InMemoryVectorStore({})

    with pytest.raises(ValueError, match="dense_vector"):
        store.upsert(
            [{"id": "chunk-1", "text": "missing vector", "metadata": {}}]
        )


@pytest.mark.unit
def test_query_returns_the_required_result_shape(
    sample_records: list[dict[str, Any]],
) -> None:
    store = InMemoryVectorStore({})
    store.upsert(sample_records)

    results = store.query([1.0, 0.0], top_k=1)

    assert results == [
        {
            "chunk_id": "chunk-1",
            "score": 1.0,
            "text": "alpha document",
            "metadata": {"collection": "docs", "source_path": "alpha.pdf"},
        }
    ]


@pytest.mark.unit
def test_query_applies_metadata_filters(
    sample_records: list[dict[str, Any]],
) -> None:
    store = InMemoryVectorStore({})
    store.upsert(sample_records)

    results = store.query(
        [1.0, 0.0], top_k=2, filters={"collection": "notes"}
    )

    assert [result["chunk_id"] for result in results] == ["chunk-2"]


@pytest.mark.unit
def test_query_rejects_non_positive_top_k() -> None:
    store = InMemoryVectorStore({})

    with pytest.raises(ValueError, match="top_k"):
        store.query([1.0, 0.0], top_k=0)


@pytest.mark.unit
def test_factory_routes_to_registered_backend(
    registered_memory_provider: None,
) -> None:
    settings = Settings(
        vector_store={"provider": "memory", "collection_name": "test"}
    )

    store = VectorStoreFactory.create(settings)

    assert isinstance(store, InMemoryVectorStore)
    assert store.config["collection_name"] == "test"
    assert store.config is not settings.vector_store


@pytest.mark.unit
def test_factory_normalizes_provider_name(
    registered_memory_provider: None,
) -> None:
    settings = Settings(vector_store={"provider": "  MEMORY  "})

    assert isinstance(VectorStoreFactory.create(settings), InMemoryVectorStore)


@pytest.mark.unit
def test_factory_reports_missing_provider() -> None:
    with pytest.raises(ValueError, match=r"vector_store\.provider"):
        VectorStoreFactory.create(Settings(vector_store={}))


@pytest.mark.unit
def test_factory_rejects_non_mapping_config() -> None:
    settings = Settings(vector_store=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"section 'vector_store'.*mapping"):
        VectorStoreFactory.create(settings)


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    settings = Settings(vector_store={"provider": "missing"})

    with pytest.raises(ValueError, match="Unsupported vector store provider 'missing'"):
        VectorStoreFactory.create(settings)


@pytest.mark.unit
def test_duplicate_registration_requires_explicit_replace(
    registered_memory_provider: None,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        VectorStoreFactory.register("MEMORY", InMemoryVectorStore)


@pytest.mark.unit
def test_factory_rejects_builder_with_wrong_return_type() -> None:
    VectorStoreFactory.register(
        "invalid", lambda config: object()  # type: ignore[return-value]
    )
    try:
        settings = Settings(vector_store={"provider": "invalid"})

        with pytest.raises(TypeError, match="must return BaseVectorStore"):
            VectorStoreFactory.create(settings)
    finally:
        VectorStoreFactory.unregister("invalid")
