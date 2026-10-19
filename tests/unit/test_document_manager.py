"""Document lifecycle with isolated persisted stores and injected failures."""

from pathlib import Path

import pytest

from core import Chunk
from ingestion.document_manager import DocumentManager
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


pytestmark = pytest.mark.unit


@pytest.fixture
def manager(tmp_path):
    return DocumentManager(
        ChromaStore({"persist_directory": str(tmp_path / "chroma")}),
        BM25Indexer(tmp_path / "bm25"),
        ImageStorage(tmp_path / "images", tmp_path / "images.db"),
        SQLiteIntegrityChecker(tmp_path / "history.db"),
    )


def seed(manager, tmp_path, key="a", collection="alpha", source=None, file_hash=None):
    source = source or str(tmp_path / f"{key}.pdf")
    file_hash = file_hash or key * 64
    chunks = [Chunk(id=f"{collection}-{key}-{index}", text="searchable document text",
                    metadata={"source_path": source, "document_id": file_hash,
                              "file_hash": file_hash, "collection": collection},
                    start_offset=0, end_offset=24, source_ref=file_hash)
              for index in range(2)]
    manager.chroma_store.upsert([
        {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata, "dense_vector": [1., 0.]}
        for chunk in chunks
    ])
    manager.bm25_indexer.upsert(SparseEncoder().encode(chunks), source=source)
    image = tmp_path / "input.png"
    image.write_bytes(b"image bytes")
    manager.image_storage.store(f"{collection}-{key}", image, collection, doc_hash=file_hash)
    manager.file_integrity.mark_success(file_hash, source, chunk_count=2)
    return source, file_hash


def test_list_detail_and_filtered_stats_survive_reopening(manager, tmp_path):
    source, doc_id = seed(manager, tmp_path)
    seed(manager, tmp_path, "b", "beta")
    documents = manager.list_documents("alpha")
    assert len(documents) == 1
    assert documents[0].source_path == source
    assert documents[0].chunk_count == 2 and documents[0].image_count == 1
    assert documents[0].ingested_at
    detail = manager.get_document_detail(doc_id)
    assert detail.document == documents[0]
    assert len(detail.chunks) == 2 and len(detail.images) == 1
    assert detail.images[0]["exists"]
    stats = manager.get_collection_stats()
    assert (stats.document_count, stats.chunk_count, stats.image_count, stats.collection_count) == (2, 4, 2, 2)
    assert manager.get_collection_stats("alpha").chunk_count == 2
    reopened = DocumentManager(manager.chroma_store, BM25Indexer(tmp_path / "bm25"),
                               ImageStorage(tmp_path / "images", tmp_path / "images.db"),
                               SQLiteIntegrityChecker(tmp_path / "history.db"))
    assert reopened.list_documents() == manager.list_documents()


def test_delete_cleans_all_four_stores_preserves_source_and_other_document(manager, tmp_path):
    source, doc_id = seed(manager, tmp_path)
    Path(source).write_bytes(b"original source")
    seed(manager, tmp_path, "b", "beta")
    image_path = manager.image_storage.get_path("alpha-a")
    result = manager.delete_document(source, "alpha")
    assert result.success and not result.errors
    assert (result.deleted_chunks, result.deleted_sparse_chunks, result.deleted_images,
            result.removed_integrity_records) == (2, 2, 1, 1)
    assert Path(source).read_bytes() == b"original source"
    assert not image_path.exists()
    assert not manager.file_integrity.should_skip(doc_id)
    assert [doc.collection for doc in manager.list_documents()] == ["beta"]
    assert manager.chroma_store.get_by_metadata({"collection": "alpha"}) == []
    persisted = BM25Indexer(tmp_path / "bm25")
    assert persisted.document_count == 2
    assert all(row["chunk_id"].startswith("beta") for row in persisted.query("searchable"))
    assert len(manager.image_storage.list_images()) == 1
    assert manager.delete_document(source, "alpha").deleted_chunks == 0


def test_collection_scoped_delete_preserves_shared_source_and_hash(manager, tmp_path):
    source, doc_id = seed(manager, tmp_path)
    seed(manager, tmp_path, "a", "beta", source=source, file_hash=doc_id)
    with pytest.raises(ValueError, match="ambiguous"):
        manager.get_document_detail(doc_id)
    assert manager.get_document_detail(doc_id, "alpha").document.collection == "alpha"
    assert manager.delete_document(source, "alpha").success
    assert manager.file_integrity.should_skip(doc_id)
    assert manager.image_storage.get_path("beta-a").exists()
    assert manager.bm25_indexer.document_count == 2
    assert manager.list_documents()[0].collection == "beta"
    assert manager.delete_document(source, "beta").success
    assert not manager.file_integrity.should_skip(doc_id)


@pytest.mark.parametrize("stage,attribute,method", [
    ("bm25", "bm25_indexer", "remove_document"),
    ("images", "image_storage", "delete_document"),
    ("integrity", "file_integrity", "remove_record"),
    ("chroma", "chroma_store", "delete_by_metadata"),
])
def test_partial_failure_retains_identity_and_can_be_retried(manager, tmp_path, monkeypatch,
                                                            stage, attribute, method):
    source, doc_id = seed(manager, tmp_path)
    store = getattr(manager, attribute)
    original = getattr(store, method)
    def fail(*args, **kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(store, method, fail)
    result = manager.delete_document(source, "alpha")
    assert not result.success and result.errors == {stage: "OSError"}
    assert manager.list_documents()[0].document_id == doc_id
    monkeypatch.setattr(store, method, original)
    assert manager.delete_document(source, "alpha").success
    assert manager.list_documents() == []
    assert manager.bm25_indexer.document_count == 0
    assert manager.image_storage.list_images() == []
    assert manager.file_integrity.list_processed() == []


def test_history_lists_only_success_and_exact_hash_removal(manager, tmp_path):
    checker = manager.file_integrity
    checker.mark_success("a" * 64, str(tmp_path / "a.pdf"))
    checker.mark_processing("b" * 64, str(tmp_path / "b.pdf"))
    checker.mark_failed("c" * 64, "error", file_path=str(tmp_path / "c.pdf"))
    assert [row["file_hash"] for row in checker.list_processed()] == ["a" * 64]
    assert checker.remove_record("a" * 64)
    assert not checker.remove_record("a" * 64)
    assert not checker.should_skip("a" * 64)
    with pytest.raises(ValueError):
        checker.remove_record("' OR 1=1")


def test_chroma_delete_requires_explicit_nonempty_filters(manager, tmp_path):
    seed(manager, tmp_path)
    with pytest.raises(ValueError):
        manager.chroma_store.delete_by_metadata({})
    assert len(manager.chroma_store.list_records()) == 2
    assert manager.chroma_store.delete_by_metadata({"collection": "missing"}) == 0


def test_sparse_failed_save_restores_memory_and_disk(manager, tmp_path, monkeypatch):
    source, _ = seed(manager, tmp_path)
    before = manager.bm25_indexer.query("searchable")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(manager.bm25_indexer, "save", fail)
    with pytest.raises(OSError):
        manager.bm25_indexer.remove_document(source)
    assert manager.bm25_indexer.query("searchable") == before
    assert BM25Indexer(tmp_path / "bm25").query("searchable") == before


def test_empty_missing_and_invalid_requests(manager, tmp_path):
    assert manager.list_documents() == []
    assert manager.get_collection_stats().document_count == 0
    assert manager.delete_document(tmp_path / "absent.pdf", "alpha").success
    with pytest.raises(KeyError):
        manager.get_document_detail("absent")
    for method in (manager.list_documents, manager.get_collection_stats):
        with pytest.raises(ValueError):
            method("")
    with pytest.raises(ValueError):
        manager.delete_document("", "alpha")
    with pytest.raises(ValueError):
        manager.delete_document("file.pdf", "")
