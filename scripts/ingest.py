"""离线文档摄取命令行入口。

CLI 只负责用户输入、配置加载、文件发现和退出码；Loader、Embedding、存储一致性等
业务规则全部委托给 ``IngestionPipeline``，避免脚本成为第二套实现。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
# 直接执行 ``python scripts/ingest.py`` 时，Python 只会自动加入 scripts 目录。
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
SUPPORTED_SUFFIXES = frozenset({".pdf"})


def build_parser() -> argparse.ArgumentParser:
    """定义稳定的 CLI 契约，供真实命令和测试共用。"""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the configured RAG indexes."
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="PDF file or directory containing PDF files.",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Logical collection stored in chunk metadata (default: default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even when their content hash is already successful.",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help=argparse.SUPPRESS,
    )
    return parser


def discover_sources(path: str | Path) -> list[Path]:
    """把单文件或目录解析为确定排序、去重后的 PDF 文件列表。"""
    if not isinstance(path, (str, Path)) or (
        isinstance(path, str) and not path.strip()
    ):
        raise ValueError("path must be a non-empty string or Path")

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"ingestion path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported document type: {source.suffix or '<none>'}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"ingestion path is neither a file nor directory: {source}")

    # resolve 后去重可避免同一 PDF 通过符号链接或大小写形式被重复摄取。
    discovered = {
        item.resolve()
        for item in source.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    }
    sources = sorted(discovered, key=lambda item: str(item).casefold())
    if not sources:
        raise ValueError(f"no supported PDF files found under: {source}")
    return sources


def main(argv: Sequence[str] | None = None) -> int:
    """运行摄取任务；任一文件失败时返回 1，输入/配置错误返回 2。"""
    args = build_parser().parse_args(argv)
    # argparse 的 --help/参数错误路径不需要加载重量级 Chroma、PDF 和模型依赖。
    from core.settings import SettingsError, load_settings
    from ingestion import IngestionPipeline, IngestionPipelineError
    from observability.logger import get_logger

    bootstrap_logger = get_logger("scripts.ingest")

    try:
        settings = load_settings(args.settings)
        sources = discover_sources(args.path)
        pipeline = IngestionPipeline(settings)
    except (OSError, SettingsError, TypeError, ValueError) as exc:
        bootstrap_logger.error("Cannot start ingestion: %s", exc)
        return 2

    logger = get_logger("scripts.ingest", settings.observability["log_level"])
    failed = False
    for source in sources:
        try:
            result = pipeline.run(
                source,
                collection=args.collection,
                force=args.force,
            )
        except IngestionPipelineError as exc:
            failed = True
            logger.error("Failed to ingest %s: %s", source, exc)
            continue

        # JSON Lines 既适合人读取，也让自动化调用方不必解析日志文本。
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
