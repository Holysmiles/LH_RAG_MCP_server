"""持久化摄取流水线中的图片文件及其元数据索引。

整体链路：``Loader 图片引用 -> ImageStorage.store -> 托管文件 + SQLite 行``。

Loader 负责抽取图片，本模块把安全的栅格图片复制到 ``root/{collection}``，再记录
``image_id -> 文件路径``。文件系统与 SQLite 无法组成真正的跨资源事务，因此写入使用
临时文件、原子替换和失败补偿，保证数据库失败时恢复旧文件。SQLite 使用 WAL 与等待
超时，使并发摄取进程可以共享索引。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Generator
import uuid


class ImageStorage:
    """保存图片，并查询持久化的 ``image_id -> path`` 映射。

    输入：托管根目录、SQLite 路径、图片 ID/源路径/Collection，以及可选文档哈希和页码。
    输出：``store`` 返回绝对托管路径；查询方法返回路径或 JSON 兼容记录。
    失败边界：非法标识、扩展名和配置在写入前报错；源文件缺失抛 FileNotFoundError；任何
    从数据库读出的路径在访问或删除前都必须重新确认位于托管根目录内，防止索引被篡改
    后越权读写外部文件。
    """

    _safe_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    _allowed_extensions = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    )

    def __init__(
        self,
        root: str | Path = Path("data/images"),
        db_path: str | Path = Path("data/db/image_index.db"),
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.root = self._normalize_path(root, "root").expanduser().resolve()
        self.db_path = self._normalize_path(db_path, "db_path").expanduser().resolve()
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
        """复制一张图片，并以失败可恢复的方式更新对应数据库行。"""
        normalized_id = self._normalize_identifier(image_id, "image_id")
        normalized_collection = self._normalize_identifier(collection, "collection")
        source = self._normalize_path(source_path, "source_path").expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Image source does not exist: {source}")
        if not source.is_file():
            raise IsADirectoryError(f"Image source is not a file: {source}")
        normalized_doc_hash = self._normalize_optional_text(doc_hash, "doc_hash")
        normalized_page = self._normalize_page(page_num)

        suffix = source.suffix.lower()
        if suffix not in self._allowed_extensions:
            raise ValueError(f"unsupported image extension: {suffix or '<none>'}")
        target = (
            self.root / normalized_collection / f"{normalized_id}{suffix}"
        ).resolve()
        self._require_managed_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        token = uuid.uuid4().hex
        temporary_path = target.with_name(f".{target.name}.{token}.tmp")
        target_backup = target.with_name(f".{target.name}.{token}.bak")
        old_backup: Path | None = None
        old_path: Path | None = None
        target_installed = False
        target_backed_up = False
        old_backed_up = False
        copy_required = source != target

        # 先完整复制到同目录临时文件，os.replace 才能提供同文件系统原子替换。
        if copy_required:
            shutil.copy2(source, temporary_path)
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT file_path FROM image_index WHERE image_id = ?",
                    (normalized_id,),
                ).fetchone()
                if existing is not None:
                    old_path = self._managed_path(existing[0])

                if copy_required:
                    if target.is_file():
                        os.replace(target, target_backup)
                        target_backed_up = True
                    os.replace(temporary_path, target)
                    target_installed = True

                if old_path is not None and old_path != target and old_path.is_file():
                    old_backup = old_path.with_name(
                        f".{old_path.name}.{token}.bak"
                    )
                    os.replace(old_path, old_backup)
                    old_backed_up = True

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
        except Exception:
            # 数据库提交失败时撤销文件变化，使旧索引与旧文件仍然一致。
            if old_backed_up and old_backup is not None:
                os.replace(old_backup, old_path)  # type: ignore[arg-type]
            if target_installed:
                target.unlink(missing_ok=True)
            if target_backed_up:
                os.replace(target_backup, target)
            temporary_path.unlink(missing_ok=True)
            raise
        else:
            self._discard_backup(target_backup)
            if old_backup is not None:
                self._discard_backup(old_backup)
        finally:
            temporary_path.unlink(missing_ok=True)
        return target

    def store_many(
        self,
        images: Sequence[Mapping[str, Any]],
        collection: str,
        *,
        doc_hash: str | None = None,
    ) -> list[Path]:
        """按输入顺序保存 Loader 风格的图片映射。"""
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
        """返回 ID 对应的安全托管路径；不存在、失效或越界时返回 ``None``。"""
        normalized_id = self._normalize_identifier(image_id, "image_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        path = self._managed_path(row[0])
        return path if path is not None and path.is_file() else None

    def list_images(
        self,
        *,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出索引图片，可按 Collection 或源文档过滤。"""
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
        results: list[dict[str, Any]] = []
        for row in rows:
            managed_path = self._managed_path(row[1])
            results.append({
                "image_id": str(row[0]),
                "file_path": str(row[1]),
                "collection": str(row[2]),
                "doc_hash": row[3],
                "page_num": row[4],
                "created_at": row[5],
                "exists": managed_path is not None and managed_path.is_file(),
            })
        return results

    def delete_document(
        self, doc_hash: str, *, delete_files: bool = True, collection: str | None = None,
    ) -> int:
        """可按集合删除源文档的托管文件，全部成功后再删除数据库行。"""
        normalized_hash = self._normalize_identifier(doc_hash, "doc_hash")
        normalized_collection = self._normalize_optional_text(collection, "collection")
        where = "doc_hash = ?" + (" AND collection = ?" if normalized_collection is not None else "")
        parameters = (normalized_hash,) if normalized_collection is None else (normalized_hash, normalized_collection)
        token = uuid.uuid4().hex
        backups: list[tuple[Path, Path]] = []
        row_count = 0
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    f"SELECT file_path FROM image_index WHERE {where}",
                    parameters,
                ).fetchall()
                row_count = len(rows)
                if delete_files:
                    managed_paths: list[Path] = []
                    for row in rows:
                        path = self._managed_path(row[0])
                        if path is None:
                            raise ValueError(
                                "indexed image path is outside managed root"
                            )
                        managed_paths.append(path)
                    for path in managed_paths:
                        if path.is_file():
                            backup = path.with_name(
                                f".{path.name}.{token}.delete"
                            )
                            os.replace(path, backup)
                            backups.append((path, backup))
                connection.execute(
                    f"DELETE FROM image_index WHERE {where}",
                    parameters,
                )
        except Exception:
            # DELETE 或 COMMIT 失败时把已经移走的文件原位恢复，便于安全重试。
            for original, backup in reversed(backups):
                if backup.is_file():
                    os.replace(backup, original)
            raise
        else:
            for _, backup in backups:
                self._discard_backup(backup)
        return row_count

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
        """提供自动提交/回滚并确保关闭的 SQLite 连接上下文。"""
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

    def _managed_path(self, value: object) -> Path | None:
        """把数据库路径解析为绝对路径，仅在它仍位于托管根目录内时返回。"""
        if not isinstance(value, (str, Path)):
            return None
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, RuntimeError):
            return None
        if path == self.root or self.root not in path.parents:
            return None
        return path

    def _require_managed_path(self, path: Path) -> None:
        """断言计算得到的写入目标没有逃逸出托管根目录。"""
        if self._managed_path(path) is None:
            raise ValueError("image target path is outside managed root")

    @staticmethod
    def _discard_backup(path: Path) -> None:
        """数据库已提交后尽力清理备份；清理失败不改变已成功提交的业务结果。"""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = ["ImageStorage"]
