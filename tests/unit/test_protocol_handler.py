"""Tool contracts without a database, model, or external service."""

from copy import deepcopy
from threading import get_ident

import pytest
from jsonschema.exceptions import SchemaError
from mcp import types
from mcp.shared.exceptions import MCPError
from mcp_types.version import LATEST_HANDSHAKE_VERSION

from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.server import create_server


pytestmark = [pytest.mark.unit, pytest.mark.anyio]
SCHEMA = {"type": "object", "properties": {"query": {"type": "string", "minLength": 1}},
          "required": ["query"], "additionalProperties": False}
INIT = {"protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def handler():
    return ProtocolHandler("test-server", "1.2.3")


def echo(arguments):
    return {"content": [{"type": "text", "text": arguments["query"]}]}


def test_initialize_negotiates_version_and_advertises_tools(handler):
    response = handler.handle_initialize(INIT).model_dump(by_alias=True)
    assert response["protocolVersion"] == "2024-11-05"
    assert response["serverInfo"]["name"] == "test-server"
    assert response["serverInfo"]["version"] == "1.2.3"
    assert response["capabilities"]["tools"] == {"listChanged": False}
    assert handler.handle_initialize({**INIT, "protocolVersion": "unknown"}).protocol_version == LATEST_HANDSHAKE_VERSION
    assert create_server(handler) is handler.server


@pytest.mark.parametrize("params", [{}, {**INIT, "capabilities": []}, {**INIT, "clientInfo": {}}])
def test_invalid_initialize_is_invalid_params(handler, params):
    with pytest.raises(MCPError) as caught:
        handler.handle_initialize(params)
    assert caught.value.code == -32602


def test_tools_list_is_ordered_and_schemas_are_isolated(handler):
    assert handler.handle_tools_list().tools == []
    schema = deepcopy(SCHEMA)
    handler.register_tool("echo", "Echo the query", schema, echo)
    handler.register_tool("second", "Second tool", SCHEMA, echo)
    schema["required"].clear()
    first = handler.handle_tools_list()
    assert [tool.name for tool in first.tools] == ["echo", "second"]
    assert first.tools[0].description == "Echo the query"
    assert first.tools[0].model_dump(by_alias=True)["inputSchema"] == SCHEMA
    first.tools[0].input_schema["required"].clear()
    assert handler.handle_tools_list().tools[0].input_schema == SCHEMA


@pytest.mark.parametrize("name,schema,callback,error", [
    ("", SCHEMA, echo, ValueError),
    ("echo", {"type": "array"}, echo, ValueError),
    ("echo", {"type": "object", "properties": []}, echo, SchemaError),
    ("echo", SCHEMA, None, TypeError),
])
def test_bad_registration_fails_early(handler, name, schema, callback, error):
    with pytest.raises(error):
        handler.register_tool(name, "test", schema, callback)
    assert handler.handle_tools_list().tools == []


def test_duplicate_registration_does_not_replace_tool(handler):
    handler.register_tool("echo", "original", SCHEMA, echo)
    with pytest.raises(ValueError, match="already registered"):
        handler.register_tool("echo", "replacement", SCHEMA, echo)
    assert handler.handle_tools_list().tools[0].description == "original"


async def test_sync_callback_runs_outside_event_loop(handler):
    threads = []

    def callback(arguments):
        threads.append(get_ident())
        arguments["query"] = "worker result"
        return echo(arguments)

    handler.register_tool("echo", "Echo", SCHEMA, callback)
    arguments = {"query": "中文问题"}
    response = await handler.handle_tools_call("echo", arguments)
    assert response.content[0].text == "worker result"
    assert threads[0] != get_ident()
    assert arguments == {"query": "中文问题"}


async def test_async_callback_and_explicit_tool_error_result(handler):
    async def callback(arguments):
        return types.CallToolResult(content=[types.TextContent(type="text", text="No results")], isError=True)

    handler.register_tool("echo", "Echo", SCHEMA, callback)
    result = await handler.handle_tools_call("echo", {"query": "test"})
    assert result.is_error is True
    assert result.content[0].text == "No results"


async def test_optional_arguments_default_to_empty_object(handler):
    handler.register_tool("empty", "Empty", {"type": "object"},
                          lambda args: {"content": [{"type": "text", "text": str(args)}]})
    assert (await handler.handle_tools_call("empty")).content[0].text == "{}"


@pytest.mark.parametrize("name,arguments", [
    ("unknown", {}), ("echo", []), ("echo", "secret"),
    ("echo", {}), ("echo", {"query": 12}), ("echo", {"query": ""}),
    ("echo", {"query": "test", "extra": "secret"}),
])
async def test_invalid_tool_call_does_not_execute_callback(handler, name, arguments):
    calls = []
    handler.register_tool("echo", "Echo", SCHEMA, calls.append)
    with pytest.raises(MCPError) as caught:
        await handler.handle_tools_call(name, arguments)
    assert caught.value.code == -32602
    assert "secret" not in str(caught.value)
    assert not calls


@pytest.mark.parametrize("failure", [RuntimeError("secret traceback"),
                                     MCPError(-32602, "secret provider error")])
async def test_callback_failures_are_internal_errors_without_details(handler, failure):
    async def callback(arguments):
        raise failure

    handler.register_tool("echo", "Echo", SCHEMA, callback)
    with pytest.raises(MCPError) as caught:
        await handler.handle_tools_call("echo", {"query": "test"})
    assert caught.value.error.model_dump(exclude_none=True) == {"code": -32603, "message": "Internal error"}


@pytest.mark.parametrize("result", [None, "plain text", {"content": [{"type": "text"}]}])
async def test_bad_callback_result_is_internal_error(handler, result):
    handler.register_tool("echo", "Echo", SCHEMA, lambda args: result)
    with pytest.raises(MCPError) as caught:
        await handler.handle_tools_call("echo", {"query": "test"})
    assert caught.value.code == -32603
