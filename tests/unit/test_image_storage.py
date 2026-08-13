"""Contract tests for managed image files and the SQLite image index."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from ingestion.storage import ImageStorage


def make_source(tmp_path: Path, name: str = "source.png") -> Path:
    source = tmp_path / name
    source.write_bytes(b"fake-png-content")
    return source


@pytest.mark.unit
def test_store_persists_file_and_mapping_roundtrip(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "image_index.db")
    source = make_source(tmp_path)

    stored = storage.store("image-1", source, "docs", doc_hash="doc-hash", page_num=2)

    assert stored.exists()
    assert stored.parent == tmp_path / "images" / "docs"
    assert stored.read_bytes() == source.read_bytes()
    assert storage.get_path("image-1") == stored
    rows = storage.list_images(collection="docs", doc_hash="doc-hash")
    assert rows[0]["image_id"] == "image-1"
    assert rows[0]["page_num"] == 2
    assert rows[0]["exists"] is True

    with sqlite3.connect(tmp_path / "db" / "image_index.db") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


@pytest.mark.unit
def test_store_many_and_upsert_are_idempotent(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "index.db")
    first = make_source(tmp_path, "first.png")
    second = make_source(tmp_path, "second.png")

    paths = storage.store_many(
        [{"id": "a", "path": str(first), "page": 1}, {"id": "b", "path": str(second), "page": 2}],
        "collection-a",
        doc_hash="doc-a",
    )
    replacement = make_source(tmp_path, "replacement.png")
    replaced = storage.store("a", replacement, "collection-b", doc_hash="doc-b")

    assert len(paths) == 2
    assert storage.get_path("a") == replaced
    assert len(storage.list_images()) == 2
    assert storage.list_images(collection="collection-a")[0]["image_id"] == "b"
    assert storage.list_images(doc_hash="doc-b")[0]["image_id"] == "a"


@pytest.mark.unit
def test_delete_document_removes_rows_and_files(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "index.db")
    source = make_source(tmp_path)
    stored = storage.store("image-1", source, "docs", doc_hash="doc-hash")

    assert storage.delete_document("doc-hash") == 1
    assert not stored.exists()
    assert storage.get_path("image-1") is None
    assert storage.list_images() == []


@pytest.mark.unit
def test_validation_prevents_unsafe_ids_and_bad_sources(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "index.db")
    source = make_source(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        storage.store("../escape", source, "docs")
    with pytest.raises(FileNotFoundError):
        storage.store("image-1", tmp_path / "missing.png", "docs")
    with pytest.raises(ValueError, match="collection"):
        storage.list_images(collection="")
