"""Ollama Embedding Provider 的 mock HTTP 单元测试。

测试只替换 ``urlopen``，所以不需要本机安装或启动 Ollama；每个用例都能
清楚地看到请求地址、请求体和错误边界，适合第一次阅读 Provider 代码时
作为行为说明。
"""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from core.settings import Settings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import EmbeddingProviderError, OllamaEmbedding


class FakeHTTPResponse:
    """模拟 ``urlopen`` 返回的、支持 ``with`` 语句的 HTTP 响应。"""

    def __init__(self, body: bytes) -> None:
        self.body = body

    @classmethod
    def from_json(cls, payload: Any) -> "FakeHTTPResponse":
        return cls(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.mark.unit
def test_factory_creates_builtin_ollama_embedding() -> None:
    """provider=ollama 时，工厂应创建 OllamaEmbedding 且使用默认地址。"""
    config = {"provider": "ollama", "model": "nomic-embed-text"}

    embedding = EmbeddingFactory.create(Settings(embedding=config))

    assert isinstance(embedding, OllamaEmbedding)
    assert embedding.config == config
    assert embedding.config is not config
    assert embedding.base_url == "http://localhost:11434"


@pytest.mark.unit
def test_embed_sends_one_batch_request_and_preserves_vector_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量文本应放在同一个 input 数组中，并按服务返回顺序输出。"""
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured.update(request=request, timeout=timeout)
        return FakeHTTPResponse.from_json(
            {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        )

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding(
        {
            "model": "nomic-embed-text",
            "base_url": "http://ollama.local:11434/",
            "timeout": 8,
        }
    )

    vectors = embedding.embed(["first text", "second text"])

    request = captured["request"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert request.full_url == "http://ollama.local:11434/api/embed"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") is None
    assert json.loads(request.data) == {
        "model": "nomic-embed-text",
        "input": ["first text", "second text"],
    }
    assert captured["timeout"] == 8.0


@pytest.mark.unit
def test_single_text_uses_the_same_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单条输入也使用 input 列表，保证调用方不需要两套接口。"""
    captured: dict[str, Request] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured["request"] = request
        return FakeHTTPResponse.from_json({"embeddings": [[0.5, 0.6]]})

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)

    assert OllamaEmbedding({"model": "mxbai-embed-large"}).embed(["hello"]) == [
        [0.5, 0.6]
    ]
    assert json.loads(captured["request"].data)["input"] == ["hello"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("texts", "detail"),
    [
        ([], "must not be empty"),
        ([""], r"texts\[0\] must not be empty"),
        (["   "], r"texts\[0\] must not be empty"),
        (["valid", 42], r"texts\[1\] must be a string"),
    ],
)
def test_embed_reports_clear_input_errors(texts: list[Any], detail: str) -> None:
    """输入问题应在发请求前报告，避免浪费一次本地模型调用。"""
    embedding = OllamaEmbedding({"model": "nomic-embed-text"})

    with pytest.raises(ValueError, match=rf"ollama validation error:.*{detail}"):
        embedding.embed(texts)  # type: ignore[arg-type]


@pytest.mark.unit
def test_long_text_is_sent_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider 不会擅自截断长文本，超出模型限制时交给 Ollama 报错。"""
    captured: dict[str, Request] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured["request"] = request
        return FakeHTTPResponse.from_json({"embeddings": [[0.1]]})

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    text = "long text " * 1000

    OllamaEmbedding({"model": "model"}).embed([text])

    assert json.loads(captured["request"].data)["input"] == [text]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (FakeHTTPResponse(b"not-json"), "JSONDecodeError"),
        (FakeHTTPResponse(b"\xff"), "UnicodeDecodeError"),
        (FakeHTTPResponse.from_json({"unexpected": "shape"}), "KeyError"),
        (FakeHTTPResponse.from_json({"embeddings": []}), "ValueError"),
        (
            FakeHTTPResponse.from_json({"embeddings": [[True]]}),
            "TypeError",
        ),
        (
            FakeHTTPResponse.from_json({"embeddings": [[math.nan]]}),
            "ValueError",
        ),
    ],
)
def test_invalid_response_is_wrapped_safely(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeHTTPResponse,
    error_type: str,
) -> None:
    """坏响应只暴露 provider 和错误类型，不把原始正文交给上层。"""

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        return response

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding({"model": "private-model"})

    with pytest.raises(
        EmbeddingProviderError,
        match=rf"ollama {error_type}: invalid embedding response",
    ):
        embedding.embed(["hello"])


@pytest.mark.unit
@pytest.mark.parametrize("failure", [URLError("connection refused"), TimeoutError()])
def test_network_failure_does_not_leak_configuration(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """连接失败和超时要可读，但不能泄露私有地址或模型名。"""

    def failing_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        raise failure

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", failing_urlopen)
    embedding = OllamaEmbedding(
        {
            "model": "private-model",
            "base_url": "http://private-host:11434",
        }
    )

    with pytest.raises(EmbeddingProviderError) as error:
        embedding.embed(["hello"])

    message = str(error.value)
    assert f"ollama {type(failure).__name__}" in message
    assert "private-host" not in message
    assert "private-model" not in message


@pytest.mark.unit
def test_missing_model_reports_configuration_path() -> None:
    """缺少模型名称时要指出 embedding.model，而不是等到联网才失败。"""
    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*embedding\.model",
    ):
        OllamaEmbedding({})


@pytest.mark.unit
def test_invalid_base_url_fails_before_network_call() -> None:
    """服务地址必须是完整 HTTP(S) 地址。"""
    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*absolute http or https URL",
    ):
        OllamaEmbedding({"model": "model", "base_url": "localhost:11434"})


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [True, 0, math.nan, math.inf, -math.inf])
def test_invalid_timeout_is_rejected(timeout: object) -> None:
    """布尔值、零和非有限数字都不能作为 HTTP 超时。"""
    with pytest.raises(
        ValueError,
        match=r"ollama configuration error:.*positive number",
    ):
        OllamaEmbedding({"model": "model", "timeout": timeout})
