"""在解析文件前用 SHA256 和 SQLite 判断是否需要重复摄取。

它在完整 Ingestion Pipeline 中处于第一道关口：

``文件 -> compute_sha256 -> should_skip -> Loader -> ... -> mark_success/failed``

输入：本地文件路径，或已经计算好的 64 位 SHA256 字符串。
输出：SHA256 字符串、是否跳过的布尔值，以及持久化到 ``ingestion_history``
表中的处理状态。

只有 ``success`` 状态会返回 ``should_skip=True``。``processing`` 代表上次流程
还未完成，``failed`` 代表需要重试；这两种状态都不能跳过。业务层只依赖
``FileIntegrityChecker``，将来把 SQLite 换成 Redis/PostgreSQL 时无需改 Pipeline。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import sqlite3
from typing import ClassVar


DEFAULT_INTEGRITY_DB_PATH = Path("data/db/ingestion_history.db")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class FileIntegrityChecker(ABC):
    """定义 Pipeline 所需的文件去重和处理状态能力。"""

    @abstractmethod
    def compute_sha256(self, path: str | Path) -> str:
        """读取文件并返回稳定的 64 位十六进制 SHA256。"""
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """仅当该 hash 已成功处理时返回 ``True``。"""
        raise NotImplementedError

    @abstractmethod
    def mark_processing(
        self,
        file_hash: str,
        file_path: str | Path,
        *,
        file_size: int | None = None,
    ) -> None:
        """在 Loader 开始前记录文件正在处理。"""
        raise NotImplementedError

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        *,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """在全部入库成功后记录成功状态和可选统计信息。"""
        raise NotImplementedError

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        *,
        file_path: str | Path | None = None,
        file_size: int | None = None,
    ) -> None:
        """记录失败原因；未知 hash 首次失败时需同时提供文件路径。"""
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """使用本地 SQLite 保存文件处理历史的默认实现。

    每个方法都创建并关闭自己的数据库连接，因此同一个 checker 可以被多个
    线程共享。数据库使用 WAL 模式：读操作不会阻塞另一个进程追加写入；
    ``busy_timeout`` 则让短暂写锁等待一会，而不是立刻抛 ``database is locked``。
    """

    hash_chunk_size: ClassVar[int] = 1024 * 1024

    def __init__(
        self,
        db_path: str | Path = DEFAULT_INTEGRITY_DB_PATH,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """创建父目录、数据库表和索引，不读取任何待摄取文件。"""
        self.db_path = _normalize_path(db_path, "db_path")
        if (
            isinstance(busy_timeout_ms, bool)
            or not isinstance(busy_timeout_ms, int)
            or busy_timeout_ms < 1
        ):
            raise ValueError("busy_timeout_ms must be a positive integer")
        self.busy_timeout_ms = busy_timeout_ms
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def compute_sha256(self, path: str | Path) -> str:
        """分块读取文件并返回 SHA256，不把大文件一次性装进内存。

        Args:
            path: 必须存在且指向普通文件的路径。

        Returns:
            小写的 64 位十六进制摘要；相同字节内容始终得到相同结果。
        """
        file_path = _normalize_path(path, "path")
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {file_path}")

        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            while chunk := source.read(self.hash_chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """查询 hash 是否已有 ``success`` 记录，不修改数据库。"""
        normalized_hash = _normalize_sha256(file_hash)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM ingestion_history
                WHERE file_hash = ? AND status = 'success'
                LIMIT 1
                """,
                (normalized_hash,),
            ).fetchone()
        return row is not None

    def mark_processing(
        self,
        file_hash: str,
        file_path: str | Path,
        *,
        file_size: int | None = None,
    ) -> None:
        """插入或覆盖为 processing；此状态不会触发增量跳过。"""
        self._upsert_status(
            file_hash=file_hash,
            file_path=file_path,
            status="processing",
            file_size=file_size,
            error_msg=None,
            chunk_count=None,
        )

    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        *,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """原子写入 success；该事务提交后 ``should_skip`` 才返回 True。"""
        self._upsert_status(
            file_hash=file_hash,
            file_path=file_path,
            status="success",
            file_size=file_size,
            error_msg=None,
            chunk_count=chunk_count,
        )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        *,
        file_path: str | Path | None = None,
        file_size: int | None = None,
    ) -> None:
        """把记录改为 failed，使下次摄取能够重试。

        Pipeline 通常先调用 ``mark_processing``，此时只需传 hash 和错误信息。
        如果异常发生在 processing 记录建立前，调用者可额外传 ``file_path``，
        本方法会直接插入一条 failed 记录。
        """
        normalized_hash = _normalize_sha256(file_hash)
        normalized_error = _normalize_non_empty_text("error_msg", error_msg)
        normalized_size = _normalize_optional_count("file_size", file_size)

        if file_path is not None:
            self._upsert_status(
                file_hash=normalized_hash,
                file_path=file_path,
                status="failed",
                file_size=normalized_size,
                error_msg=normalized_error,
                chunk_count=None,
            )
            return

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE ingestion_history
                SET status = 'failed',
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = ?,
                    chunk_count = NULL,
                    file_size = COALESCE(?, file_size)
                WHERE file_hash = ?
                """,
                (normalized_error, normalized_size, normalized_hash),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    "file_path is required when marking an unknown hash as failed"
                )

    def _upsert_status(
        self,
        *,
        file_hash: str,
        file_path: str | Path,
        status: str,
        file_size: int | None,
        error_msg: str | None,
        chunk_count: int | None,
    ) -> None:
        """用一个 UPSERT 事务完成插入或状态覆盖，避免先查后写的竞争。"""
        normalized_hash = _normalize_sha256(file_hash)
        normalized_path = str(_normalize_path(file_path, "file_path"))
        normalized_size = _normalize_optional_count("file_size", file_size)
        normalized_chunks = _normalize_optional_count("chunk_count", chunk_count)

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash,
                    file_path,
                    file_size,
                    status,
                    processed_at,
                    error_msg,
                    chunk_count
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    status = excluded.status,
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = excluded.error_msg,
                    chunk_count = excluded.chunk_count
                """,
                (
                    normalized_hash,
                    normalized_path,
                    normalized_size,
                    status,
                    error_msg,
                    normalized_chunks,
                ),
            )

    def _initialize_database(self) -> None:
        """创建表、索引并确认实际启用了 WAL。"""
        with self._connection() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "wal":
                raise RuntimeError("SQLite integrity database could not enable WAL mode")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL
                        CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_status
                    ON ingestion_history(status);
                CREATE INDEX IF NOT EXISTS idx_processed_at
                    ON ingestion_history(processed_at);
                """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """提供一次短生命周期事务，退出时提交/回滚并关闭连接。"""
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1_000,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            with connection:
                yield connection
        finally:
            connection.close()


def _normalize_sha256(file_hash: object) -> str:
    if not isinstance(file_hash, str) or not _SHA256_PATTERN.fullmatch(file_hash):
        raise ValueError("file_hash must be a 64-character SHA256 hex string")
    return file_hash.lower()


def _normalize_path(path: object, field: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{field} must be a string or Path")
    if isinstance(path, str) and not path.strip():
        raise ValueError(f"{field} must not be empty")
    return Path(path)


def _normalize_non_empty_text(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalize_optional_count(field: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer or None")
    return value


__all__ = [
    "DEFAULT_INTEGRITY_DB_PATH",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
