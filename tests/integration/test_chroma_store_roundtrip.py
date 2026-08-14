"""Real ChromaDB roundtrip tests for the B7.6 storage adapter.

Every test uses pytest's ``tmp_path`` directory.  The tests therefore exercise
the real persistent database without reading or modifying ``data/db/chroma``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    """Return three small records whose nearest-neighbour order is obvious."""
    return [
        {
            "id": "chunk-alpha",
            "text": "Alpha explains vector retrieval.",
            "metadata": {
                "collection": "guides",
                "language": "en",
            },
            "dense_vector": [1.0, 0.0],
        },
        {
            "id": "chunk-beta",
            "text": "Beta is also close to the query.",
            "metadata": {
                "collection": "guides",
                "language": "zh",
            },
            "dense_vector": [0.8, 0.2],
        },
        {
            "id": "chunk-gamma",
            "text": "Gamma points in another direction.",
            "metadata": {
                "collection": "notes",
                "language": "en",
            },
            "dense_vector": [0.0, 1.0],
        },
    ]


def make_config(directory: Path) -> dict[str, Any]:
    """Build a unique Chroma configuration for one temporary test database."""
    return {
        "provider": "chroma",
        "persist_directory": str(directory),
        "collection_name": "test_knowledge_hub",
    }


@pytest.mark.integration
def test_factory_creates_builtin_chroma_store(tmp_path: Path) -> None:
    """provider=chroma should work without manual runtime registration."""
    config = make_config(tmp_path / "factory-db")

    store = VectorStoreFactory.create(Settings(vector_store=config))

    assert isinstance(store, ChromaStore)
    assert store.config == config
    assert store.config is not config


@pytest.mark.integration
def test_upsert_then_query_returns_ranked_text_and_metadata(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    """The complete write/read path must preserve every public result field."""
    store = ChromaStore(make_config(tmp_path / "roundtrip-db"))
    store.upsert(sample_records)

    results = store.query([1.0, 0.0], top_k=2)

    assert [result["chunk_id"] for result in results] == [
        "chunk-alpha",
        "chunk-beta",
    ]
    assert results[0] == {
        "chunk_id": "chunk-alpha",
        "score": pytest.approx(1.0),
        "text": "Alpha explains vector retrieval.",
        "metadata": {"collection": "guides", "language": "en"},
    }
    assert results[0]["score"] > results[1]["score"]


@pytest.mark.integration
def test_query_honours_top_k_and_multiple_metadata_filters(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    """The adapter hides Chroma's special $and syntax from its callers."""
    store = ChromaStore(make_config(tmp_path / "filter-db"))
    store.upsert(sample_records)

    results = store.query(
        [1.0, 0.0],
        top_k=1,
        filters={"collection": "guides", "language": "zh"},
    )

    assert [result["chunk_id"] for result in results] == ["chunk-beta"]


@pytest.mark.integration
def test_upsert_replaces_an_existing_record_with_the_same_id(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    """Upsert means update-or-insert, so a stable chunk ID cannot duplicate."""
    store = ChromaStore(make_config(tmp_path / "update-db"))
    store.upsert(sample_records)
    store.upsert(
        [
            {
                "id": "chunk-alpha",
                "text": "Alpha was updated.",
                "metadata": {"collection": "updated"},
                "dense_vector": [0.0, 1.0],
            }
        ]
    )

    results = store.query(
        [0.0, 1.0],
        top_k=1,
        filters={"collection": "updated"},
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-alpha"
    assert results[0]["text"] == "Alpha was updated."


@pytest.mark.integration
def test_same_directory_and_collection_reopen_persisted_records(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    """A second adapter instance must see data written by the first instance."""
    config = make_config(tmp_path / "persistent-db")
    first_store = ChromaStore(config)
    first_store.upsert(sample_records)

    reopened_store = ChromaStore(config)
    results = reopened_store.query([1.0, 0.0], top_k=1)

    assert results[0]["chunk_id"] == "chunk-alpha"


@pytest.mark.integration
def test_empty_metadata_roundtrips_as_an_empty_dictionary(
    tmp_path: Path,
) -> None:
    """The project accepts {}, even though Chroma internally requires None."""
    store = ChromaStore(make_config(tmp_path / "empty-metadata-db"))
    store.upsert(
        [
            {
                "id": "chunk-empty",
                "text": "No metadata is still valid.",
                "metadata": {},
                "dense_vector": [1.0, 0.0],
            }
        ]
    )

    results = store.query([1.0, 0.0], top_k=1)

    assert results[0]["metadata"] == {}


@pytest.mark.integration
def test_nested_metadata_roundtrips_without_losing_filterable_scalars(
    tmp_path: Path,
) -> None:
    store = ChromaStore(make_config(tmp_path / "nested-metadata-db"))
    metadata = {
        "collection": "guides",
        "tags": ["rag", "azure"],
        "images": [{"id": "image-1", "path": "data/image-1.png"}],
        "image_captions": {"image-1": "architecture chart"},
    }
    store.upsert(
        [
            {
                "id": "chunk-nested",
                "text": "Nested metadata must survive Chroma.",
                "metadata": metadata,
                "dense_vector": [1.0, 0.0],
            }
        ]
    )

    results = store.query(
        [1.0, 0.0], top_k=1, filters={"collection": "guides"}
    )

    assert results[0]["metadata"] == metadata


@pytest.mark.integration
@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"id": "missing-fields"}], "missing fields"),
        (
            [
                {
                    "id": "bad-vector",
                    "text": "text",
                    "metadata": {},
                    "dense_vector": [],
                }
            ],
            "must not be empty",
        ),
        (
            [
                {
                    "id": "duplicate",
                    "text": "first",
                    "metadata": {},
                    "dense_vector": [1.0],
                },
                {
                    "id": "duplicate",
                    "text": "second",
                    "metadata": {},
                    "dense_vector": [1.0],
                },
            ],
            "duplicate id",
        ),
    ],
)
def test_upsert_rejects_invalid_records_before_writing(
    tmp_path: Path,
    records: list[dict[str, Any]],
    message: str,
) -> None:
    """Invalid input should produce a clear project error, not a Chroma trace."""
    store = ChromaStore(make_config(tmp_path / f"invalid-{message}"))

    with pytest.raises(ValueError, match=message):
        store.upsert(records)


@pytest.mark.integration
@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_query_rejects_invalid_top_k(tmp_path: Path, top_k: Any) -> None:
    """A result count must be a positive whole number."""
    store = ChromaStore(make_config(tmp_path / f"top-k-{top_k}"))

    with pytest.raises(ValueError, match="top_k"):
        store.query([1.0, 0.0], top_k=top_k)
