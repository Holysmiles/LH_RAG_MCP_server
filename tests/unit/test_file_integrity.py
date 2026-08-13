"""C2 SHA256 + SQLite 文件完整性检查的离线单元测试。

测试把数据库放在 ``tmp_path``，不会创建或修改真实的
``data/db/ingestion_history.db``。它覆盖文件指纹、状态机、进程重启后的持久化
和 WAL 并发写入，等价验证 Pipeline 最前面的“是否需要重复处理”关口。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sqlite3

import pytest

from libs.loader import FileIntegrityChecker, SQLiteIntegrityChecker


@pytest.fixture
def checker(tmp_path: Path) -> SQLiteIntegrityChecker:
    """每个测试使用独立 SQLite 文件，保证状态互不污染。"""
    return SQLiteIntegrityChecker(tmp_path / "db" / "ingestion_history.db")


def sha256_of(text: str) -> str:
    """为不关心真实文件内容的状态测试生成合法 hash。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.unit
def test_interface_is_abstract_and_sqlite_implements_it(
    checker: SQLiteIntegrityChecker,
) -> None:
    """Pipeline 可依赖抽象接口，而默认实现必须满足该接口。"""
    with pytest.raises(TypeError):
        FileIntegrityChecker()
    assert isinstance(checker, FileIntegrityChecker)


@pytest.mark.unit
def test_compute_sha256_is_deterministic_and_content_sensitive(
    checker: SQLiteIntegrityChecker,
    tmp_path: Path,
) -> None:
    """相同字节必须得到相同 hash，内容改变后 hash 必须改变。"""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"same document bytes")

    first = checker.compute_sha256(source)
    second = checker.compute_sha256(str(source))
    source.write_bytes(b"changed document bytes")
    changed = checker.compute_sha256(source)

    assert first == second
    assert first == hashlib.sha256(b"same document bytes").hexdigest()
    assert len(first) == 64
    assert changed != first


@pytest.mark.unit
def test_compute_sha256_rejects_missing_file_and_directory(
    checker: SQLiteIntegrityChecker,
    tmp_path: Path,
) -> None:
    """错误路径在昂贵的 Loader 启动前就应给出清晰异常。"""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        checker.compute_sha256(tmp_path / "missing.pdf")
    with pytest.raises(IsADirectoryError, match="not a file"):
        checker.compute_sha256(tmp_path)


@pytest.mark.unit
def test_default_database_is_created_at_documented_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不传配置时数据库必须落在 data/db/ingestion_history.db。"""
    monkeypatch.chdir(tmp_path)

    default_checker = SQLiteIntegrityChecker()

    assert default_checker.db_path == Path("data/db/ingestion_history.db")
    assert (tmp_path / "data" / "db" / "ingestion_history.db").is_file()


@pytest.mark.unit
def test_only_success_status_is_skipped(
    checker: SQLiteIntegrityChecker,
    tmp_path: Path,
) -> None:
    """processing/failed 必须重试，只有全部完成的 success 可以零成本跳过。"""
    source = tmp_path / "report.pdf"
    source.write_bytes(b"report")
    file_hash = checker.compute_sha256(source)

    assert checker.should_skip(file_hash) is False
    checker.mark_processing(file_hash, source, file_size=source.stat().st_size)
    assert checker.should_skip(file_hash) is False
    checker.mark_failed(file_hash, "temporary parser failure")
    assert checker.should_skip(file_hash) is False
    checker.mark_success(
        file_hash,
        source,
        file_size=source.stat().st_size,
        chunk_count=7,
    )
    assert checker.should_skip(file_hash) is True


@pytest.mark.unit
def test_success_record_contains_expected_fields(
    checker: SQLiteIntegrityChecker,
    tmp_path: Path,
) -> None:
    """成功记录要保存路径、大小、chunk 数，并清除旧错误。"""
    source = tmp_path / "report.pdf"
    source.write_bytes(b"report")
    file_hash = checker.compute_sha256(source)
    checker.mark_failed(
        file_hash,
        "first attempt failed",
        file_path=source,
        file_size=6,
    )

    checker.mark_success(file_hash, source, file_size=6, chunk_count=3)

    with sqlite3.connect(checker.db_path) as connection:
        row = connection.execute(
            """
            SELECT file_path, file_size, status, error_msg, chunk_count
            FROM ingestion_history
            WHERE file_hash = ?
            """,
            (file_hash,),
        ).fetchone()
    assert row == (str(source), 6, "success", None, 3)


@pytest.mark.unit
def test_database_uses_wal_and_expected_indexes(
    checker: SQLiteIntegrityChecker,
) -> None:
    """WAL 和两个查询索引是并发与后续 Dashboard 列表查询的基础。"""
    with sqlite3.connect(checker.db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list('ingestion_history')")
        }

    assert journal_mode.lower() == "wal"
    assert {"idx_status", "idx_processed_at"}.issubset(indexes)


@pytest.mark.unit
def test_records_persist_across_checker_instances(
    checker: SQLiteIntegrityChecker,
) -> None:
    """进程重启后仍应识别已完成文件，这才是真正的增量摄取。"""
    file_hash = sha256_of("persistent")
    checker.mark_success(file_hash, "persistent.pdf", chunk_count=2)

    reopened = SQLiteIntegrityChecker(checker.db_path)

    assert reopened.should_skip(file_hash) is True


@pytest.mark.unit
def test_concurrent_writes_are_not_lost(
    checker: SQLiteIntegrityChecker,
) -> None:
    """多个 worker 同时完成文件时，WAL + UPSERT 应保留全部成功记录。"""
    record_count = 24

    def write_record(index: int) -> None:
        checker.mark_success(
            sha256_of(f"document-{index}"),
            f"document-{index}.pdf",
            file_size=index,
            chunk_count=index + 1,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_record, range(record_count)))

    with sqlite3.connect(checker.db_path) as connection:
        success_count = connection.execute(
            "SELECT COUNT(*) FROM ingestion_history WHERE status = 'success'"
        ).fetchone()[0]
    assert success_count == record_count


@pytest.mark.unit
def test_mark_failed_requires_path_for_unknown_hash(
    checker: SQLiteIntegrityChecker,
) -> None:
    """表中 file_path 非空；未知 hash 首次失败时调用者必须补充路径。"""
    with pytest.raises(ValueError, match="file_path is required"):
        checker.mark_failed(sha256_of("unknown"), "parse failed")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda item: item.should_skip("not-a-sha256"), "64-character"),
        (
            lambda item: item.mark_success(
                sha256_of("bad-size"),
                "file.pdf",
                file_size=-1,
            ),
            "file_size",
        ),
        (
            lambda item: item.mark_success(
                sha256_of("bad-count"),
                "file.pdf",
                chunk_count=True,
            ),
            "chunk_count",
        ),
    ],
)
def test_invalid_state_inputs_are_rejected(
    checker: SQLiteIntegrityChecker,
    operation: Callable[[SQLiteIntegrityChecker], object],
    message: str,
) -> None:
    """坏 hash 或统计值不能写进共享历史表。"""
    with pytest.raises(ValueError, match=message):
        operation(checker)
