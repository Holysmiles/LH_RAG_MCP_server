"""Persist uploads and coordinate dashboard mutations using existing ingestion APIs."""

import hashlib
from collections.abc import Callable
from pathlib import Path
import re
from threading import RLock

from core.settings import Settings
from ingestion import IngestionPipeline, IngestionResult
from ingestion.document_manager import DeleteResult, DocumentManager
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


_mutation_lock = RLock()


class IngestionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def normalize_collection(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value.strip()):
            raise ValueError("collection must contain letters, digits, dots, underscores or hyphens")
        return value.strip()

    def ingest_upload(
        self, filename: str, content: bytes, collection: str, *, force: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> IngestionResult:
        collection = self.normalize_collection(collection)
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("a PDF filename is required")
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if Path(name).suffix.lower() != ".pdf":
            raise ValueError("only PDF uploads are supported")
        if not isinstance(content, bytes) or not content or not content.startswith(b"%PDF-"):
            raise ValueError("upload must contain a nonempty PDF")
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")
        if self.settings.vector_store.get("provider") != "chroma":
            raise ValueError("dashboard ingestion requires the chroma provider")
        # A stable content-derived name avoids overwriting a prior upload and
        # keeps source references valid after the Streamlit session ends.
        documents_path = self.settings.ingestion.get("documents_path", "data/documents")
        if not isinstance(documents_path, (str, Path)) or not str(documents_path).strip():
            raise ValueError("ingestion.documents_path must be a nonempty path")
        root = Path(documents_path).expanduser().resolve()
        safe_name = re.sub(r"[^\w.-]", "_", Path(name).stem)[:80] or "document"
        target = (root / collection / f"{hashlib.sha256(content).hexdigest()}-{safe_name}.pdf").resolve()
        if root not in target.parents:
            raise ValueError("upload path is outside documents directory")
        with _mutation_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with target.open("xb") as destination:
                    created = True
                    destination.write(content)
            except FileExistsError:
                if target.read_bytes() != content:
                    raise ValueError("upload conflicts with an existing file")
            except Exception:
                if created:
                    target.unlink(missing_ok=True)
                raise
            return IngestionPipeline(self.settings).run(target, collection, force=force, on_progress=on_progress)

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        collection = self.normalize_collection(collection)
        with _mutation_lock:
            store = ChromaStore(self.settings.vector_store, create_if_missing=False)
            config = self.settings.ingestion
            manager = DocumentManager(
                store, BM25Indexer(config.get("bm25_path", "data/db/bm25")),
                ImageStorage(config.get("image_root", "data/images"),
                             config.get("image_index_db_path", "data/db/image_index.db")),
                SQLiteIntegrityChecker(config.get("integrity_db_path", "data/db/ingestion_history.db")),
            )
            return manager.delete_document(source_path, collection)
