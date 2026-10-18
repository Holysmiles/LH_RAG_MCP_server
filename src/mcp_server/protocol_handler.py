"""Tool registry and protocol contracts on top of the MCP SDK 2 dispatcher.

The SDK owns JSON-RPC envelopes, request IDs, handshake state and unknown-method
errors. This layer validates tool arguments and hides implementation failures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
import inspect
from typing import Any

import anyio
from jsonschema import validators
from jsonschema.exceptions import ValidationError as SchemaValidationError
from mcp import types
from mcp.server import Server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, LATEST_HANDSHAKE_VERSION
from pydantic import ValidationError

from observability.logger import get_logger


@asynccontextmanager
async def protocol_messages(read_stream, write_stream):
    """Turn SDK transport validation failures into sanitized JSON-RPC errors.

    The SDK otherwise drops malformed envelopes. Their IDs cannot be trusted,
    so the response uses null; valid requests retain their IDs in the SDK.
    """
    sender, receiver = anyio.create_memory_object_stream(0)

    async def forward():
        async with sender:
            async for item in read_stream:
                if isinstance(item, Exception):
                    parse_error = (isinstance(item, ValidationError) and
                                   any(e["type"] == "json_invalid" for e in item.errors()))
                    code = types.PARSE_ERROR if parse_error else types.INVALID_REQUEST
                    message = "Parse error" if parse_error else "Invalid Request"
                    await write_stream.send(SessionMessage(types.JSONRPCError(
                        jsonrpc="2.0", id=None,
                        error=types.ErrorData(code=code, message=message),
                    )))
                else:
                    await sender.send(item)

    async with receiver, anyio.create_task_group() as tasks:
        tasks.start_soon(forward)
        try:
            yield receiver
        finally:
            tasks.cancel_scope.cancel()


class ProtocolHandler:
    """Register tools before serving; callbacks accept one arguments dictionary.

    Callbacks may be synchronous or asynchronous and return a CallToolResult or
    its wire dictionary. Sync callbacks run in a worker thread. Registration is
    fixed for a session (no tools/list_changed notifications are advertised).
    """

    def __init__(self, name: str, version: str) -> None:
        self._tools: dict[str, tuple[types.Tool, Any, Callable]] = {}
        self.server = Server(
            name, version=version,
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    def register_tool(
        self, name: str, description: str, input_schema: dict[str, Any],
        callback: Callable,
    ) -> None:
        """Reject duplicate names and invalid schemas at registration time."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if not callable(callback):
            raise TypeError("Tool callback must be callable")
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            raise ValueError("Tool input schema must describe an object")
        schema = deepcopy(input_schema)
        validator_type = validators.validator_for(schema)
        validator_type.check_schema(schema)
        tool = types.Tool(name=name, description=description, inputSchema=schema)
        self._tools[name] = (tool, validator_type(schema), callback)

    def handle_initialize(self, params: Mapping[str, Any]) -> types.InitializeResult:
        """Preview the SDK handshake result without creating session state.

        The live SDK performs the same negotiation and enforces initialization
        order; callers must not use this preview as a replacement for a session.
        """
        try:
            request = types.InitializeRequestParams.model_validate(params)
        except ValidationError:
            raise MCPError(types.INVALID_PARAMS, "Invalid initialize parameters") from None
        requested = request.protocol_version
        negotiated = (requested if requested in HANDSHAKE_PROTOCOL_VERSIONS
                      else LATEST_HANDSHAKE_VERSION)
        return types.InitializeResult(
            protocolVersion=negotiated,
            serverInfo=self.server.server_info,
            capabilities=self.server.create_initialization_options().capabilities,
        )

    def handle_tools_list(self) -> types.ListToolsResult:
        """Return independent schema copies in registration order."""
        return types.ListToolsResult(
            tools=[tool.model_copy(deep=True) for tool, _, _ in self._tools.values()]
        )

    async def handle_tools_call(
        self, name: str, arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        if not isinstance(name, str) or name not in self._tools:
            raise MCPError(types.INVALID_PARAMS, "Unknown tool")
        if arguments is not None and not isinstance(arguments, dict):
            raise MCPError(types.INVALID_PARAMS, "Tool arguments must be an object")
        _, validator, callback = self._tools[name]
        payload = deepcopy(arguments) if arguments is not None else {}
        try:
            validator.validate(payload)
        except SchemaValidationError:
            # Do not echo arguments: they may contain credentials or documents.
            raise MCPError(types.INVALID_PARAMS, "Invalid tool arguments") from None
        except Exception:
            get_logger(__name__).error("Tool schema validation failed: %s", name)
            raise MCPError(types.INTERNAL_ERROR, "Internal error") from None
        try:
            if inspect.iscoroutinefunction(callback):
                result = await callback(payload)
            else:
                result = await anyio.to_thread.run_sync(callback, payload)
                if inspect.isawaitable(result):
                    result = await result
            # Validate even SDK model instances, which a callback may have built
            # without validation through model_construct().
            if isinstance(result, types.CallToolResult):
                result = result.model_dump(by_alias=True, mode="json")
            return types.CallToolResult.model_validate(result)
        except Exception:
            get_logger(__name__).error("Tool execution failed: %s", name)
            raise MCPError(types.INTERNAL_ERROR, "Internal error") from None

    async def _on_list_tools(self, ctx, params) -> types.ListToolsResult:
        if params is not None and params.cursor is not None:
            raise MCPError(types.INVALID_PARAMS, "Tool pagination is not supported")
        return self.handle_tools_list()

    async def _on_call_tool(self, ctx, params) -> types.CallToolResult:
        return await self.handle_tools_call(params.name, params.arguments)
