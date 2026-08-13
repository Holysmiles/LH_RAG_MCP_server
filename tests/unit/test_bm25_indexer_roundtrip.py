"""Round-trip and update tests for the persistent BM25 index."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from core import Chunk


def chunk(index: int, text: str) -> Chunk:
    return Chunk(
        id=f"doc-{index}",
        text=text,
        metadata={"source_path": "docs/source.pdf", "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc",
    )


def encoded() -> object:
    return SparseEncoder().encode(
        [chunk(0, "azure endpoint endpoint"), chunk(1, "vector endpoint"), chunk(2, "python")]
    )


@pytest.mark.unit
def test_build_query_idf_and_postings_are_deterministic(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "bm25")
    indexer.build(encoded(), persist=False)  # type: ignore[arg-type]

    assert indexer.document_count == 3
    assert math.isclose(indexer.idf["endpoint"], math.log((3 - 2 + 0.5) / (2 + 0.5)))
    results = indexer.query("endpoint", top_k=10)
    assert [item["chunk_id"] for item in results] == ["doc-1", "doc-0"]
    assert results[0]["score"] > results[1]["score"]
    assert indexer.postings["endpoint"] == [
        {"chunk_id": "doc-0", "tf": 2.0, "doc_length": 3},
        {"chunk_id": "doc-1", "tf": 1.0, "doc_length": 2},
    ]


@pytest.mark.unit
def test_save_load_roundtrip_returns_same_results(tmp_path: Path) -> None:
    path = tmp_path / "bm25"
    indexer = BM25Indexer(path)
    indexer.build(encoded())  # type: ignore[arg-type]
    before = indexer.query("azure endpoint", top_k=3)

    loaded = BM25Indexer.load(path)

    assert loaded.query("azure endpoint", top_k=3) == before
    assert (path / "index.json").exists()


@pytest.mark.unit
def test_incremental_upsert_replaces_record_and_remove_document_rebuilds(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "bm25")
    indexer.build(SparseEncoder().encode([chunk(0, "old term"), chunk(1, "keep")]), source="doc-a")
    indexer.upsert(SparseEncoder().encode([chunk(0, "new term")]), source="doc-a", persist=False)

    assert indexer.query("old", 5) == []
    assert indexer.query("new", 5)[0]["chunk_id"] == "doc-0"
    assert indexer.remove_document("doc-a", persist=False) == 2
    assert indexer.query("new", 5) == []


@pytest.mark.unit
def test_empty_query_and_input_validation(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "bm25")
    indexer.build([], persist=False)

    assert indexer.query("", 1) == []
    with pytest.raises(ValueError, match="top_k"):
        indexer.query("term", 0)
    with pytest.raises(TypeError, match="encoded"):
        indexer.build("bad", persist=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        indexer.remove_document("")
