"""OllamaLLM 单元测试。

测试通过 monkeypatch 替换真正的网络函数，所以不要求开发机器安装或启动
Ollama。每个测试只验证一个可观察行为，可作为阅读实现代码时的使用示例。
"""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from core.settings import Settings
from libs.llm.base_http_llm import LLMProviderError
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM


class FakeHTTPResponse:
    """模拟 ``urlopen`` 返回的、支持 ``with`` 语句的 HTTP 响应。"""

    def __init__(self, body: bytes) -> None:
        self.body = body

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> FakeHTTPResponse:
        return cls(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.mark.unit
def test_factory_creates_builtin_ollama_provider() -> None:
    """provider=ollama 时，工厂应返回 OllamaLLM 并隔离配置对象。"""
    config = {"provider": "ollama", "model": "qwen2.5:7b"}
    settings = Settings(llm=config)

    llm = LLMFactory.create(settings)

    assert isinstance(llm, OllamaLLM)
    assert llm.config == config
    assert llm.config is not settings.llm
    assert llm.base_url == "http://localhost:11434"


@pytest.mark.unit
def test_chat_sends_native_ollama_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """请求应发送到 /api/chat，并显式关闭流式响应。"""
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured.update(request=request, timeout=timeout)
        return FakeHTTPResponse.from_json(
            {"message": {"role": "assistant", "content": "本地模型回答"}}
        )

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", fake_urlopen)
    llm = OllamaLLM(
        {
            "model": "qwen2.5:7b",
            "base_url": "http://ollama.local:11434/",
            "timeout": 5,
        }
    )

    answer = llm.chat([{"role": "user", "content": "你好"}])

    request = captured["request"]
    assert answer == "本地模型回答"
    assert request.full_url == "http://ollama.local:11434/api/chat"
    assert request.get_header("Authorization") is None
    assert json.loads(request.data) == {
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
    }
    assert captured["timeout"] == 5.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("messages", "detail"),
    [
        ([], "must not be empty"),
        ("hello", "non-string sequence"),
        ([{"content": "hello"}], r"messages\[0\]\.role"),
        ([{"role": "user", "content": 42}], r"messages\[0\]\.content"),
    ],
)
def test_chat_reuses_readable_message_validation(messages: Any, detail: str) -> None:
    """Ollama 与其他 LLM provider 应遵守同一消息输入契约。"""
    llm = OllamaLLM({"model": "qwen2.5:7b"})

    with pytest.raises(ValueError, match=rf"ollama validation error:.*{detail}"):
        llm.chat(messages)


@pytest.mark.unit
def test_chat_rejects_non_json_message_fields_before_network_call() -> None:
    """附加字段不能转成 JSON 时，也应得到带 provider 名称的校验错误。"""
    llm = OllamaLLM({"model": "qwen2.5:7b"})

    with pytest.raises(
        ValueError,
        match=r"ollama validation error:.*JSON-serializable",
    ):
        llm.chat(
            [
                {
                    "role": "user",
                    "content": "你好",
                    "unsupported_value": object(),
                }
            ]
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    [URLError("connection refused"), TimeoutError("request timed out")],
)
def test_connection_failure_is_readable_and_does_not_leak_config(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """连接失败只暴露 provider 与错误类型，不回显敏感配置。"""

    def failing_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        raise failure

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", failing_urlopen)
    secret = "must-not-appear"
    llm = OllamaLLM(
        {
            "model": "private-model",
            "base_url": "http://private-host:11434",
            "api_key": secret,
        }
    )

    with pytest.raises(LLMProviderError) as error:
        llm.chat([{"role": "user", "content": "你好"}])

    message = str(error.value)
    assert f"ollama {type(failure).__name__}" in message
    assert secret not in message
    assert "private-host" not in message
    assert "private-model" not in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (FakeHTTPResponse(b"\xff"), "UnicodeDecodeError"),
        (FakeHTTPResponse(b"not-json"), "JSONDecodeError"),
        (FakeHTTPResponse.from_json({"unexpected": "shape"}), "KeyError"),
        (
            FakeHTTPResponse.from_json({"message": {"content": 42}}),
            "TypeError",
        ),
    ],
)
def test_invalid_response_reports_provider_and_error_type(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeHTTPResponse,
    error_type: str,
) -> None:
    """坏响应应转换为统一异常，而不是把底层堆栈直接暴露给业务层。"""

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        return response

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", fake_urlopen)
    llm = OllamaLLM({"model": "qwen2.5:7b"})

    with pytest.raises(
        LLMProviderError,
        match=rf"ollama {error_type}: invalid chat response",
    ):
        llm.chat([{"role": "user", "content": "你好"}])


@pytest.mark.unit
def test_missing_model_reports_configuration_path() -> None:
    """缺少模型名时应在访问网络前失败，并指出 llm.model。"""
    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*llm\.model",
    ):
        OllamaLLM({})


@pytest.mark.unit
def test_invalid_base_url_fails_before_network_call() -> None:
    """缺少 http/https 协议的地址不是可调用的 HTTP endpoint。"""
    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*absolute http or https URL",
    ):
        OllamaLLM({"model": "qwen2.5:7b", "base_url": "localhost:11434"})


@pytest.mark.unit
@pytest.mark.parametrize("base_url", [" http://localhost:11434/ ", "http://[invalid"])
def test_base_url_is_normalized_or_rejected_safely(base_url: str) -> None:
    """地址前后空格应被清理，非法地址应转成 provider 配置错误。"""
    if base_url.startswith(" "):
        llm = OllamaLLM({"model": "qwen2.5:7b", "base_url": base_url})
        assert llm.base_url == "http://localhost:11434"
        return

    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*absolute http or https URL",
    ):
        OllamaLLM({"model": "qwen2.5:7b", "base_url": base_url})


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
def test_non_finite_timeout_is_rejected(timeout: float) -> None:
    """NaN 和 Infinity 不是可靠的超时时间，必须在初始化时拒绝。"""
    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*positive number",
    ):
        OllamaLLM({"model": "qwen2.5:7b", "timeout": timeout})
