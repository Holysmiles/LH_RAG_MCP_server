"""加载配置并启动 MCP stdio 服务；stdout 专用于协议消息。"""

import argparse
from pathlib import Path

from core.settings import SettingsError, load_settings
from observability.logger import get_logger

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "config" / "settings.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG MCP server over stdio.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser


def main(settings_path: str | Path | None = None) -> int:
    """配置错误立即退出；正常运行直到客户端关闭 stdin。"""
    path = settings_path or DEFAULT_SETTINGS_PATH
    bootstrap_logger = get_logger(__name__)

    try:
        settings = load_settings(path)
        logger = get_logger(__name__, settings.observability["log_level"])
    except (OSError, SettingsError, ValueError) as exc:
        bootstrap_logger.error("Failed to load settings: %s", exc)
        return 1

    logger.info("Configuration loaded from %s", path)
    try:
        # 配置无效时不加载 SDK，更不会初始化模型或数据库。
        from mcp_server.server import run_stdio_server

        run_stdio_server(settings)
    except KeyboardInterrupt:
        logger.info("MCP server interrupted")
    except Exception as exc:
        logger.error("MCP server failed: %s", type(exc).__name__)
        return 1

    return 0


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(main(arguments.settings))
