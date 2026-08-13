"""Persistent image files and metadata index for the ingestion pipeline.

``Loader image refs -> ImageStorage.store -> managed file + SQLite row``

The loader is responsible for extracting an image.  This adapter takes a
source path and copies it into ``data/images/{collection}``, then records the
mapping in SQLite.  The file and row are written in that order: a database row
is never intentionally created for a path that could not be copied.  SQLite
uses WAL and a busy timeout so concurrent ingestion workers can safely share
the index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Generator


class ImageStorage:
    """Store image files and query their durable ``image_id -> path`` mapping.

    Inputs:
        ``root`` is the managed image directory, ``db_path`` is the SQLite
        index, and image records contain ``id`` and ``path`` plus optional
        ``doc_hash``/``page``.  ``collection`` groups files under one folder.
    Outputs:
        ``store`` returns the managed ``Path``.  ``get_path`` and
        ``list_images`` return JSON-compatible mapping records.
    Failure boundaries:
        Invalid identifiers/configuration fail before file or database writes;
        missing source files raise ``FileNotFoundError``.  Query methods skip
        stale rows whose managed file was removed externally and can be used
        to diagnose cleanup drift.
    """

    _safe_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def __init__(
        self,
        root: str | Path = Path("data/images"),
        db_path: str | Path = Path("data/db/image_index.db"),
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.root = self._normalize_path(root, "root")
        self.db_path = self._normalize_path(db_path, "db_path")
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be a positive integer")
        self.busy_timeout_ms = busy_timeout_ms
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def store(
        self,
        image_id: str,
        source_path: str | Path,
        collection: str,
        *,
        doc_hash: str | None = None,
        page_num: int | None = None,
    ) -> Path:
        """Copy one image into the managed collection and atomically upsert its row."""
        normalized_id = self._normalize_identifier(image_id, "image_id")
        normalized_collection = self._normalize_identifier(collection, "collection")
        source = self._normalize_path(source_path, "source_path")
        if not source.exists():
            raise FileNotFoundError(f"Image source does not exist: {source}")
        if not source.is_file():
            raise IsADirectoryError(f"Image source is not a file: {source}")
        normalized_doc_hash = self._normalize_optional_text(doc_hash, "doc_hash")
        normalized_page = self._normalize_page(page_num)

        suffix = source.suffix.lower() or ".bin"
        target = self.root / normalized_collection / f"{normalized_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        # copy2 is intentional: the managed store owns a separate file and
        # preserves basic source timestamps for diagnostics.
        shutil.copy2(source, target)

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO image_index (
                    image_id, file_path, collection, doc_hash, page_num
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    normalized_id,
                    str(target),
                    normalized_collection,
                    normalized_doc_hash,
                    normalized_page,
                ),
            )
        return target

    def store_many(
        self,
        images: Sequence[Mapping[str, Any]],
        collection: str,
        *,
        doc_hash: str | None = None,
    ) -> list[Path]:
        """Store Loader-style image mappings in input order."""
        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
            raise TypeError("images must be a non-string sequence")
        paths: list[Path] = []
        for index, image in enumerate(images):
            if not isinstance(image, Mapping):
                raise TypeError(f"images[{index}] must be a mapping")
            image_id = image.get("id")
            source_path = image.get("path")
            page_num = image.get("page")
            paths.append(
                self.store(
                    image_id,  # type: ignore[arg-type]
                    source_path,  # type: ignore[arg-type]
                    collection,
                    doc_hash=doc_hash,
                    page_num=page_num,  # type: ignore[arg-type]
                )
            )
        return paths

    def get_path(self, image_id: str) -> Path | None:
        """Return the managed path for an image ID, or ``None`` if absent/stale."""
        normalized_id = self._normalize_identifier(image_id, "image_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        path = Path(str(row[0]))
        return path if path.is_file() else None

    def list_images(
        self,
        *,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """List indexed images, optionally filtered by collection/document."""
        normalized_collection = self._normalize_optional_text(collection, "collection")
        normalized_doc_hash = self._normalize_optional_text(doc_hash, "doc_hash")
        clauses: list[str] = []
        values: list[str] = []
        if normalized_collection is not None:
            clauses.append("collection = ?")
            values.append(normalized_collection)
        if normalized_doc_hash is not None:
            clauses.append("doc_hash = ?")
            values.append(normalized_doc_hash)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT image_id, file_path, collection, doc_hash, page_num, created_at "
                f"FROM image_index{where} ORDER BY image_id",
                values,
            ).fetchall()
        return [
            {
                "image_id": str(row[0]),
                "file_path": str(row[1]),
                "collection": str(row[2]),
                "doc_hash": row[3],
                "page_num": row[4],
                "created_at": row[5],
                "exists": Path(str(row[1])).is_file(),
            }
            for row in rows
        ]

    def delete_document(self, doc_hash: str, *, delete_files: bool = True) -> int:
        """Delete all rows for a source document and optionally managed files."""
        normalized_hash = self._normalize_identifier(doc_hash, "doc_hash")
        rows = self.list_images(doc_hash=normalized_hash)
        with self._connection() as connection:
            connection.execute("DELETE FROM image_index WHERE doc_hash = ?", (normalized_hash,))
        if delete_files:
            for row in rows:
                path = Path(row["file_path"])
                if path.is_file():
                    path.unlink()
        return len(rows)

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "wal":
                raise RuntimeError("image index database could not enable WAL mode")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_image_collection
                    ON image_index(collection);
                CREATE INDEX IF NOT EXISTS idx_image_doc_hash
                    ON image_index(doc_hash);
                """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1_000)
        try:
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def _normalize_identifier(cls, value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        normalized = value.strip()
        if not cls._safe_id.fullmatch(normalized):
            raise ValueError(f"{field} contains unsafe path characters")
        return normalized

    @staticmethod
    def _normalize_path(value: object, field: str) -> Path:
        if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{field} must be a non-empty string or Path")
        return Path(value)

    @staticmethod
    def _normalize_optional_text(value: object, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string or None")
        return value.strip()

    @staticmethod
    def _normalize_page(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("page_num must be a non-negative integer or None")
        return value


__all__ = ["ImageStorage"]
