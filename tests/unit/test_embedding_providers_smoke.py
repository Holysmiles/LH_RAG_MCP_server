"""OpenAI/Azure Embedding provider 的 mock HTTP 单元测试。"""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from core.settings import Settings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import EmbeddingProviderError, OpenAIEmbedding


class FakeHTTPResponse:
    """模拟 urlopen 返回的简单响应，不启动真实 API 服务。"""

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
@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (
            {"provider": "openai", "model": "text-embedding-3-small", "api_key": "secret"},
            OpenAIEmbedding,
        ),
        (
            {
                "provider": "azure",
                "model": "text-embedding-ada-002",
                "api_key": "secret",
                "azure_endpoint": "https://example.openai.azure.com",
                "deployment_name": "embedding-deployment",
                "api_version": "2024-10-21",
            },
            AzureEmbedding,
        ),
    ],
)
def test_factory_routes_builtin_embedding_providers(
    config: dict[str, Any], expected_type: type[OpenAIEmbedding]
) -> None:
    """不同 provider 配置应由同一个工厂创建对应实现。"""
    assert isinstance(EmbeddingFactory.create(Settings(embedding=config)), expected_type)


@pytest.mark.unit
def test_openai_embed_sends_batch_and_restores_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量请求应传 input 列表，返回向量顺序必须跟输入顺序一致。"""
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured.update(request=request, timeout=timeout)
        return FakeHTTPResponse.from_json(
            {
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = OpenAIEmbedding(
        {
            "model": "text-embedding-3-small",
            "api_key": "openai-secret",
            "base_url": "https://example.openai/v1/",
            "timeout": 7,
        }
    )

    vectors = embedding.embed(["first text", "second text"])

    request = captured["request"]
    assert vectors == [[0.1, 0.2], [0.2, 0.3]]
    assert request.full_url == "https://example.openai/v1/embeddings"
    assert request.get_header("Authorization") == "Bearer openai-secret"
    assert json.loads(request.data) == {
        "input": ["first text", "second text"],
        "model": "text-embedding-3-small",
    }
    assert captured["timeout"] == 7.0


@pytest.mark.unit
def test_azure_embed_uses_deployment_url_and_api_key_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure 只替换 endpoint 和鉴权，批量请求体仍沿用 OpenAI 格式。"""
    captured: dict[str, Request] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured["request"] = request
        return FakeHTTPResponse.from_json(
            {"data": [{"index": 0, "embedding": [0.5, 0.6]}]}
        )

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = AzureEmbedding(
        {
            "model": "text-embedding-ada-002",
            "api_key": "azure-secret",
            "azure_endpoint": "https://example.openai.azure.com/",
            "deployment_name": "embedding deployment",
            "api_version": "2024-10-21",
        }
    )

    assert embedding.embed(["hello"]) == [[0.5, 0.6]]
    request = captured["request"]
    assert request.full_url == (
        "https://example.openai.azure.com/openai/deployments/"
        "embedding%20deployment/embeddings?api-version=2024-10-21"
    )
    assert request.get_header("Api-key") == "azure-secret"
    assert request.get_header("Authorization") is None


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
    """空批次、空文本和错误类型在联网前就应被发现。"""
    embedding = OpenAIEmbedding({"model": "text-embedding-3-small", "api_key": "secret"})

    with pytest.raises(ValueError, match=rf"openai validation error:.*{detail}"):
        embedding.embed(texts)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
def test_embed_rejects_non_finite_timeout(timeout: float) -> None:
    """NaN/Infinity 会让超时语义失效，因此配置阶段必须拒绝。"""
    with pytest.raises(ValueError, match=r"openai configuration error:.*positive number"):
        OpenAIEmbedding(
            {
                "model": "text-embedding-3-small",
                "api_key": "secret",
                "timeout": timeout,
            }
        )


@pytest.mark.unit
def test_long_text_is_sent_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """本地不偷偷截断长文本，避免用户无法察觉的信息丢失。"""
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured["request"] = request
        return FakeHTTPResponse.from_json(
            {"data": [{"index": 0, "embedding": [0.1]}]}
        )

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    text = "long text " * 1000
    OpenAIEmbedding({"model": "model", "api_key": "secret"}).embed([text])

    assert json.loads(captured["request"].data)["input"] == [text]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (FakeHTTPResponse(b"not-json"), "JSONDecodeError"),
        (FakeHTTPResponse.from_json({"unexpected": "shape"}), "KeyError"),
        (FakeHTTPResponse.from_json({"data": []}), "ValueError"),
        (
            FakeHTTPResponse.from_json({"data": [{"index": 0, "embedding": [True]}]}),
            "TypeError",
        ),
        (
            FakeHTTPResponse.from_json(
                {"data": [{"index": 0, "embedding": [float("nan")]}]}
            ),
            "ValueError",
        ),
    ],
)
def test_invalid_response_is_wrapped_safely(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeHTTPResponse,
    error_type: str,
) -> None:
    """坏响应必须包含 provider 和错误类型，但不能暴露响应原文。"""

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        return response

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = OpenAIEmbedding({"model": "model", "api_key": "secret"})

    with pytest.raises(
        EmbeddingProviderError,
        match=rf"openai {error_type}: invalid embedding response",
    ):
        embedding.embed(["hello"])


@pytest.mark.unit
@pytest.mark.parametrize("failure", [URLError("connection refused"), TimeoutError()])
def test_network_failure_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """连接/超时异常要可读，但不能带出 API Key 或私有地址。"""

    def failing_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        raise failure

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", failing_urlopen)
    secret = "embedding-secret"
    embedding = OpenAIEmbedding(
        {
            "model": "private-model",
            "api_key": secret,
            "base_url": "http://private-host/v1",
        }
    )

    with pytest.raises(EmbeddingProviderError) as error:
        embedding.embed(["hello"])

    message = str(error.value)
    assert f"openai {type(failure).__name__}" in message
    assert secret not in message
    assert "private-host" not in message
    assert "private-model" not in message


@pytest.mark.unit
def test_azure_configuration_errors_identify_missing_field() -> None:
    """Azure 配置缺失时应指出具体字段，而不是等到网络阶段才失败。"""
    with pytest.raises(
        ValueError,
        match=r"azure configuration error:.*embedding\.deployment_name",
    ):
        AzureEmbedding({"model": "model", "api_key": "secret"})
