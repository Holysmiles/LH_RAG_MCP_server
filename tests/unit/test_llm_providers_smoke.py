"""Smoke tests for OpenAI-compatible LLM providers with mocked HTTP."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from core.settings import Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import LLMProviderError, OpenAILLM


class FakeHTTPResponse:
    """Minimal context-managed response returned by mocked ``urlopen``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def successful_response(content: str = "mock answer") -> FakeHTTPResponse:
    return FakeHTTPResponse(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (
            {"provider": "openai", "model": "gpt-test", "api_key": "secret"},
            OpenAILLM,
        ),
        (
            {
                "provider": "azure",
                "model": "gpt-test",
                "api_key": "secret",
                "azure_endpoint": "https://example.openai.azure.com",
                "deployment_name": "chat-deployment",
                "api_version": "2024-10-21",
            },
            AzureLLM,
        ),
        (
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": "secret",
            },
            DeepSeekLLM,
        ),
    ],
)
def test_factory_routes_builtin_providers(
    config: dict[str, Any], expected_type: type[OpenAILLM]
) -> None:
    assert isinstance(LLMFactory.create(Settings(llm=config)), expected_type)


@pytest.mark.unit
def test_openai_chat_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured.update(request=request, timeout=timeout)
        return successful_response("hello")

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = OpenAILLM(
        {
            "model": "gpt-test",
            "api_key": "openai-secret",
            "base_url": "https://openai.example/v1/",
            "timeout": 5,
        }
    )

    answer = llm.chat([{"role": "user", "content": "Hi"}])

    request = captured["request"]
    assert answer == "hello"
    assert request.full_url == "https://openai.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer openai-secret"
    assert json.loads(request.data) == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert captured["timeout"] == 5.0


@pytest.mark.unit
def test_deepseek_uses_its_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Request] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured["request"] = request
        return successful_response()

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = DeepSeekLLM({"model": "deepseek-chat", "api_key": "secret"})

    llm.chat([{"role": "user", "content": "Hi"}])

    assert captured["request"].full_url == (
        "https://api.deepseek.com/chat/completions"
    )


@pytest.mark.unit
def test_azure_chat_uses_deployment_url_and_api_key_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Request] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured["request"] = request
        return successful_response()

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = AzureLLM(
        {
            "model": "gpt-test",
            "api_key": "azure-secret",
            "azure_endpoint": "https://example.openai.azure.com/",
            "deployment_name": "chat deployment",
            "api_version": "2024-10-21",
        }
    )

    llm.chat([{"role": "user", "content": "Hi"}])

    request = captured["request"]
    assert request.full_url == (
        "https://example.openai.azure.com/openai/deployments/chat%20deployment/"
        "chat/completions?api-version=2024-10-21"
    )
    assert request.get_header("Api-key") == "azure-secret"
    assert request.get_header("Authorization") is None


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
def test_chat_reports_readable_input_validation_errors(
    messages: Any, detail: str
) -> None:
    llm = OpenAILLM({"model": "gpt-test", "api_key": "secret"})

    with pytest.raises(ValueError, match=rf"openai validation error:.*{detail}"):
        llm.chat(messages)


@pytest.mark.unit
def test_request_failure_identifies_provider_without_leaking_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", failing_urlopen)
    api_key = "must-not-appear"
    llm = DeepSeekLLM({"model": "deepseek-chat", "api_key": api_key})

    with pytest.raises(LLMProviderError) as error:
        llm.chat([{"role": "user", "content": "Hi"}])

    assert "deepseek URLError" in str(error.value)
    assert api_key not in str(error.value)


@pytest.mark.unit
def test_invalid_response_identifies_provider_and_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"unexpected": "shape"})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = OpenAILLM({"model": "gpt-test", "api_key": "secret"})

    with pytest.raises(LLMProviderError, match="openai KeyError"):
        llm.chat([{"role": "user", "content": "Hi"}])


@pytest.mark.unit
def test_missing_configuration_identifies_provider() -> None:
    with pytest.raises(
        ValueError,
        match=r"azure configuration error:.*llm\.deployment_name",
    ):
        AzureLLM(
            {
                "model": "gpt-test",
                "api_key": "secret",
                "azure_endpoint": "https://example.openai.azure.com",
                "api_version": "2024-10-21",
            }
        )
