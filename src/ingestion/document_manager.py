"""Read document assets and coordinate retryable deletion across local stores.

The stores cannot share a transaction. Deletion stops on the first failure and
removes Chroma records last, retaining document identity for another attempt.
Callers must serialize ingestion and deletion for the same document.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


@dataclass(frozen=True)
class DocumentInfo:
    document_id: str
    source_path: str
    collection: str
    chunk_count: int
    image_count: int
    file_hash: str | None = None
    ingested_at: str | None = None


@dataclass(frozen=True)
class DocumentDetail:
    document: DocumentInfo
    chunks: list[dict[str, Any]]
    images: list[dict[str, Any]]


@dataclass(frozen=True)
class CollectionStats:
    document_count: int
    chunk_count: int
    image_count: int
    collection_count: int


@dataclass
class DeleteResult:
    source_path: str
    collection: str
    success: bool = False
    deleted_chunks: int = 0
    deleted_sparse_chunks: int = 0
    deleted_images: int = 0
    removed_integrity_records: int = 0
    errors: dict[str, str] = field(default_factory=dict)


class DocumentManager:
    def __init__(self, chroma_store: ChromaStore, bm25_indexer: BM25Indexer,
                 image_storage: ImageStorage, file_integrity: SQLiteIntegrityChecker) -> None:
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    @staticmethod
    def _text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        if collection is not None:
            collection = self._text(collection, "collection")
        records = (self.chroma_store.list_records() if collection is None else
                   self.chroma_store.get_by_metadata({"collection": collection}))
        history = {row["file_hash"]: row for row in self.file_integrity.list_processed()}
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for record in records:
            metadata = record["metadata"]
            source = metadata.get("source_path")
            doc_id = metadata.get("document_id") or metadata.get("file_hash") or metadata.get("source_ref")
            owner = metadata.get("collection", "default")
            if not all(isinstance(item, str) and item for item in (source, doc_id, owner)):
                continue
            grouped.setdefault((doc_id, source, owner), []).append(record)
        documents = []
        for (doc_id, source, owner), chunks in sorted(grouped.items()):
            file_hash = chunks[0]["metadata"].get("file_hash")
            processed = history.get(file_hash, {})
            images = self.image_storage.list_images(collection=owner, doc_hash=doc_id)
            documents.append(DocumentInfo(doc_id, source, owner, len(chunks), len(images),
                                          file_hash, processed.get("processed_at")))
        return documents

    def get_document_detail(self, doc_id: str, collection: str | None = None) -> DocumentDetail:
        """Raise KeyError if absent; require a collection for ambiguous IDs."""
        doc_id = self._text(doc_id, "doc_id")
        documents = [doc for doc in self.list_documents(collection) if doc.document_id == doc_id]
        if not documents:
            raise KeyError(doc_id)
        if len(documents) != 1:
            raise ValueError("document ID is ambiguous; select a collection and unique source")
        document = documents[0]
        chunks = self.chroma_store.get_by_metadata({"source_path": document.source_path,
                                                    "collection": document.collection})
        chunks = [row for row in chunks if (row["metadata"].get("document_id") or
                  row["metadata"].get("file_hash") or row["metadata"].get("source_ref")) == doc_id]
        images = self.image_storage.list_images(collection=document.collection, doc_hash=doc_id)
        return DocumentDetail(document, chunks, images)

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        if collection is not None:
            collection = self._text(collection, "collection")
        documents = self.list_documents(collection)
        images = self.image_storage.list_images(collection=collection)
        return CollectionStats(len(documents), sum(doc.chunk_count for doc in documents),
                               len(images), len({doc.collection for doc in documents}))

    def delete_document(self, source_path: str | Path, collection: str) -> DeleteResult:
        """Delete indexed assets only; the original source file is retained.

        Shared hashes keep their integrity record until their final indexed
        document is removed. Errors identify the incomplete store; retrying is
        safe after correcting that store's failure.
        """
        if not isinstance(source_path, (str, Path)) or not str(source_path).strip():
            raise ValueError("source_path must be a non-empty string or Path")
        source = Path(source_path).expanduser().resolve()
        collection = self._text(collection, "collection")
        result = DeleteResult(str(source), collection)
        stage = "read"
        try:
            all_documents = self.list_documents()
            targets = [doc for doc in all_documents if doc.collection == collection and
                       Path(doc.source_path).expanduser().resolve() == source]
            others = [doc for doc in all_documents if doc not in targets]
            for doc in targets:
                filters = {"source_path": doc.source_path, "collection": collection}
                records = self.chroma_store.get_by_metadata(filters)
                stage = "bm25"
                result.deleted_sparse_chunks += self.bm25_indexer.remove_document(
                    doc.source_path, chunk_ids=[record["id"] for record in records])
            stage = "images"
            for doc_id in sorted({doc.document_id for doc in targets}):
                if not any(doc.document_id == doc_id and doc.collection == collection for doc in others):
                    result.deleted_images += self.image_storage.delete_document(doc_id, collection=collection)
            stage = "integrity"
            for file_hash in sorted({doc.file_hash for doc in targets if doc.file_hash}):
                if not any(doc.file_hash == file_hash for doc in others):
                    result.removed_integrity_records += int(self.file_integrity.remove_record(file_hash))
            stage = "chroma"
            for source_name in sorted({doc.source_path for doc in targets}):
                result.deleted_chunks += self.chroma_store.delete_by_metadata(
                    {"source_path": source_name, "collection": collection})
            result.success = True
        except Exception as exc:
            result.errors[stage] = type(exc).__name__
        return result


__all__ = ["DocumentManager", "DocumentInfo", "DocumentDetail", "DeleteResult", "CollectionStats"]
