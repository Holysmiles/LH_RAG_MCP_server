"""Read existing document indexes for the dashboard without running ingestion."""

from pathlib import Path

from core.settings import Settings
from ingestion.document_manager import DocumentDetail, DocumentInfo
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import DEFAULT_INTEGRITY_DB_PATH, SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


class DataService:
    def __init__(self, chroma_store: ChromaStore | None,
                 image_storage: ImageStorage | None = None,
                 file_integrity: SQLiteIntegrityChecker | None = None) -> None:
        self.chroma_store = chroma_store
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataService":
        if settings.vector_store.get("provider") != "chroma":
            raise ValueError("data browser requires the chroma provider")
        directory = Path(settings.vector_store.get("persist_directory", "data/db/chroma")).expanduser()
        if not (directory / "chroma.sqlite3").is_file():
            return cls(None)
        store = ChromaStore(settings.vector_store, create_if_missing=False)
        image_db = Path(settings.ingestion.get("image_index_db_path", "data/db/image_index.db")).expanduser()
        image_root = Path(settings.ingestion.get("image_root", "data/images")).expanduser()
        history_db = Path(settings.ingestion.get("integrity_db_path", DEFAULT_INTEGRITY_DB_PATH)).expanduser()
        images = ImageStorage(image_root, image_db) if image_db.is_file() else None
        history = SQLiteIntegrityChecker(history_db) if history_db.is_file() else None
        return cls(store, images, history)

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        if collection is not None and (not isinstance(collection, str) or not collection.strip()):
            raise ValueError("collection must be a non-empty string")
        if self.chroma_store is None:
            return []
        records = (self.chroma_store.list_records() if collection is None else
                   self.chroma_store.get_by_metadata({"collection": collection}))
        history = ({row["file_hash"]: row for row in self.file_integrity.list_processed()}
                   if self.file_integrity is not None else {})
        grouped = {}
        for record in records:
            metadata = record["metadata"]
            identity = metadata.get("document_id") or metadata.get("file_hash") or metadata.get("source_ref")
            source, owner = metadata.get("source_path"), metadata.get("collection", "default")
            if all(isinstance(value, str) and value for value in (identity, source, owner)):
                grouped.setdefault((identity, source, owner), []).append(record)
        documents = []
        for (identity, source, owner), chunks in sorted(grouped.items()):
            metadata = chunks[0]["metadata"]
            file_hash = metadata.get("file_hash")
            timestamp = history.get(file_hash, {}).get("processed_at") or metadata.get("ingested_at")
            images = (self.image_storage.list_images(collection=owner, doc_hash=identity)
                      if self.image_storage is not None else [])
            documents.append(DocumentInfo(identity, source, owner, len(chunks), len(images), file_hash, timestamp))
        return documents

    def get_document_detail(self, document: DocumentInfo) -> DocumentDetail:
        if self.chroma_store is None:
            raise KeyError(document.document_id)
        records = self.chroma_store.get_by_metadata({"source_path": document.source_path,
                                                    "collection": document.collection})
        chunks = [row for row in records if (row["metadata"].get("document_id") or
                  row["metadata"].get("file_hash") or row["metadata"].get("source_ref")) == document.document_id]
        if not chunks:
            raise KeyError(document.document_id)
        chunks.sort(key=lambda row: (row["metadata"].get("chunk_index", 0), row["id"]))
        images = (self.image_storage.list_images(collection=document.collection, doc_hash=document.document_id)
                  if self.image_storage is not None else [])
        return DocumentDetail(document, chunks, images)

    def get_image_path(self, image_id: str) -> Path | None:
        """Use the storage boundary's managed-root checks, never metadata paths."""
        return self.image_storage.get_path(image_id) if self.image_storage is not None else None
