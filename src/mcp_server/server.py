"""MCP 服务生命周期；协议编解码和握手由官方 SDK 负责。

这里只建立 stdio 通道，不初始化检索模型或数据库。工具注册会在后续任务接入
``create_server``，保持服务启动与具体业务逻辑分离。
"""

from __future__ import annotations

from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
import sys

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server

from core.settings import Settings
from observability.logger import get_logger


SERVER_NAME = "modular-rag-mcp-server"


def create_server() -> Server:
    """创建 SDK Server，版本与项目的安装元数据保持一致。"""
    try:
        server_version = version(SERVER_NAME)
    except PackageNotFoundError:
        # 源码 checkout 未安装时仍可启动；不声称一个无法确认的发布版本。
        server_version = "0.0.0+uninstalled"

    return Server(SERVER_NAME, version=server_version)


async def serve_stdio(settings: Settings) -> None:
    """处理一个 stdio 会话，客户端关闭 stdin 后结束并释放资源。"""
    logger = get_logger(
        "mcp_server.server",
        settings.observability.get("log_level", "INFO"),
    )
    with redirect_stdout(sys.stderr):
        server = create_server()
        initialization_options = server.create_initialization_options()

    logger.info("MCP stdio server starting: %s", SERVER_NAME)
    try:
        async with stdio_server() as (read_stream, write_stream):
            # Transport 已保留协议输出流。业务 print 转向 stderr，不会混入 JSON-RPC。
            with redirect_stdout(sys.stderr):
                await server.run(
                    read_stream,
                    write_stream,
                    initialization_options,
                )
    finally:
        logger.info("MCP stdio server stopped")


def run_stdio_server(settings: Settings) -> None:
    """给同步程序入口提供单一启动函数；异步调用方使用 serve_stdio。"""
    anyio.run(serve_stdio, settings)


__all__ = ["SERVER_NAME", "create_server", "serve_stdio", "run_stdio_server"]
