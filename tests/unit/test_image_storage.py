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
    assert not paths[0].exists()


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
    unsafe = make_source(tmp_path, "active.svg")
    with pytest.raises(ValueError, match="extension"):
        storage.store("image-svg", unsafe, "docs")


@pytest.mark.unit
def test_tampered_index_cannot_read_or_delete_outside_managed_root(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db" / "index.db"
    storage = ImageStorage(tmp_path / "images", db_path)
    external = make_source(tmp_path, "external.png")
    storage.store("image-1", external, "docs", doc_hash="doc-hash")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE image_index SET file_path = ? WHERE image_id = ?",
            (str(external), "image-1"),
        )

    assert storage.get_path("image-1") is None
    with pytest.raises(ValueError, match="managed root"):
        storage.delete_document("doc-hash")
    assert external.exists()
    assert len(storage.list_images(doc_hash="doc-hash")) == 1


@pytest.mark.unit
def test_failed_database_update_restores_previous_managed_file(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db" / "index.db"
    storage = ImageStorage(tmp_path / "images", db_path)
    first = make_source(tmp_path, "first.png")
    stored = storage.store("image-1", first, "docs", doc_hash="doc-hash")
    original_content = stored.read_bytes()
    replacement = make_source(tmp_path, "replacement.png")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_image_update
            BEFORE UPDATE ON image_index
            BEGIN
                SELECT RAISE(FAIL, 'forced update failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced update failure"):
        storage.store("image-1", replacement, "docs", doc_hash="doc-hash")

    assert stored.read_bytes() == original_content
    assert storage.get_path("image-1") == stored


@pytest.mark.unit
def test_failed_database_delete_restores_files_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "index.db"
    storage = ImageStorage(tmp_path / "images", db_path)
    source = make_source(tmp_path)
    stored = storage.store("image-1", source, "docs", doc_hash="doc-hash")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_image_delete
            BEFORE DELETE ON image_index
            BEGIN
                SELECT RAISE(FAIL, 'forced delete failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced delete failure"):
        storage.delete_document("doc-hash")

    assert stored.exists()
    assert storage.get_path("image-1") == stored
