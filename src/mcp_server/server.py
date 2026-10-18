"""MCP 服务生命周期；协议编解码和握手由官方 SDK 负责。

这里只建立 stdio 通道和工具协议，不初始化检索模型或数据库。
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
from mcp_server.protocol_handler import ProtocolHandler, protocol_messages


SERVER_NAME = "modular-rag-mcp-server"


def create_server(
    protocol_handler: ProtocolHandler | None = None, *, settings: Settings | None = None,
) -> Server:
    """创建 SDK Server，版本与项目的安装元数据保持一致。"""
    if protocol_handler is not None:
        return protocol_handler.server
    try:
        server_version = version(SERVER_NAME)
    except PackageNotFoundError:
        # 源码 checkout 未安装时仍可启动；不声称一个无法确认的发布版本。
        server_version = "0.0.0+uninstalled"

    handler = ProtocolHandler(SERVER_NAME, server_version)
    if settings is not None:
        from mcp_server.tools.list_collections import register_list_collections_tool
        from mcp_server.tools.query_knowledge_hub import register_query_tool

        register_query_tool(handler, settings)
        register_list_collections_tool(handler, settings)
    return handler.server


async def serve_stdio(settings: Settings) -> None:
    """处理一个 stdio 会话，客户端关闭 stdin 后结束并释放资源。"""
    logger = get_logger(
        "mcp_server.server",
        settings.observability.get("log_level", "INFO"),
    )
    with redirect_stdout(sys.stderr):
        server = create_server(settings=settings)
        initialization_options = server.create_initialization_options()

    logger.info("MCP stdio server starting: %s", SERVER_NAME)
    try:
        async with stdio_server() as (read_stream, write_stream):
            # Transport 已保留协议输出流。业务 print 转向 stderr，不会混入 JSON-RPC。
            async with protocol_messages(read_stream, write_stream) as messages:
                with redirect_stdout(sys.stderr):
                    await server.run(
                        messages,
                        write_stream,
                        initialization_options,
                    )
    finally:
        logger.info("MCP stdio server stopped")


def run_stdio_server(settings: Settings) -> None:
    """给同步程序入口提供单一启动函数；异步调用方使用 serve_stdio。"""
    anyio.run(serve_stdio, settings)


__all__ = ["SERVER_NAME", "create_server", "serve_stdio", "run_stdio_server"]
