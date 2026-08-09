"""B9 Azure Vision LLM 的离线单元测试。

所有 Azure HTTP 请求都被 Fake ``urlopen`` 截获，不使用真实密钥或网络。测试
关注完整数据流：图片路径/base64/bytes -> 尺寸处理 -> Azure JSON -> caption，
以及超时、401 和坏图片是否能在不泄露敏感配置的情况下给出可读错误。
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

from PIL import Image
import pytest

from core.settings import Settings
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.base_http_llm import LLMProviderError
from libs.llm.llm_factory import LLMFactory


class FakeHTTPResponse:
    """模拟 Azure 返回的 context manager 响应对象。"""

    def __init__(self, content: str = "A line chart rising over time.") -> None:
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture
def azure_config() -> dict[str, Any]:
    """构造不含真实账号信息的最小 Azure Vision 配置。"""
    return {
        "provider": "azure",
        "model": "gpt-4o",
        "api_key": "azure-secret",
        "azure_endpoint": "https://example.openai.azure.com/",
        "deployment_name": "vision deployment",
        "api_version": "2024-10-21",
        "max_image_size": 2048,
        "timeout": 7,
    }


@pytest.fixture
def png_bytes() -> bytes:
    """生成一张很小的真实 PNG，避免把损坏字节误当作图片测试。"""
    output = BytesIO()
    Image.new("RGB", (16, 8), color="white").save(output, format="PNG")
    return output.getvalue()


@pytest.mark.unit
def test_factory_creates_builtin_azure_vision_provider(
    azure_config: dict[str, Any],
) -> None:
    """vision_llm.provider=azure 时必须路由到 AzureVisionLLM。"""
    vision_llm = LLMFactory.create_vision_llm(
        Settings(vision_llm=azure_config)
    )

    assert isinstance(vision_llm, AzureVisionLLM)
    assert vision_llm.model == "gpt-4o"
    assert vision_llm.max_image_size == 2048
    assert vision_llm.config is not azure_config


@pytest.mark.unit
def test_image_path_builds_expected_azure_multimodal_request(
    tmp_path: Path,
    azure_config: dict[str, Any],
    png_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地路径应被读取成 data URI，并放进 Azure image_url 消息块。"""
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(png_bytes)
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured.update(request=request, timeout=timeout)
        return FakeHTTPResponse("mock caption")

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", fake_urlopen)
    vision_llm = AzureVisionLLM(azure_config)

    response = vision_llm.chat_with_image(" Describe the chart. ", str(image_path))

    request = captured["request"]
    payload = json.loads(request.data)
    content = payload["messages"][0]["content"]
    assert response == "mock caption"
    assert request.full_url == (
        "https://example.openai.azure.com/openai/deployments/"
        "vision%20deployment/chat/completions?api-version=2024-10-21"
    )
    assert request.get_header("Api-key") == "azure-secret"
    assert request.get_header("Authorization") is None
    assert captured["timeout"] == 7.0
    assert content[0] == {"type": "text", "text": "Describe the chart."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.unit
@pytest.mark.parametrize("input_kind", ["raw_base64", "data_uri", "bytes"])
def test_base64_and_bytes_inputs_are_supported(
    input_kind: str,
    azure_config: dict[str, Any],
    png_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """数据库中的 base64 和 Loader 的 bytes 都应走同一图片处理路径。"""
    captured_payload: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured_payload.update(json.loads(request.data))
        return FakeHTTPResponse()

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", fake_urlopen)
    encoded = b64encode(png_bytes).decode("ascii")
    image_input: str | bytes
    if input_kind == "raw_base64":
        image_input = encoded
    elif input_kind == "data_uri":
        image_input = f"data:image/png;base64,{encoded}"
    else:
        image_input = png_bytes

    AzureVisionLLM(azure_config).chat_with_image("caption", image_input)

    content = captured_payload["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.unit
def test_oversized_image_is_resized_without_changing_aspect_ratio(
    azure_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """400x200 图片配置上限为 100 时，应缩成 100x50 而不是拉伸。"""
    source = BytesIO()
    Image.new("RGB", (400, 200), color="blue").save(source, format="PNG")
    azure_config["max_image_size"] = 100
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        captured.update(json.loads(request.data))
        return FakeHTTPResponse()

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", fake_urlopen)

    AzureVisionLLM(azure_config).chat_with_image("caption", source.getvalue())

    data_uri = captured["messages"][0]["content"][1]["image_url"]["url"]
    resized_bytes = b64decode(data_uri.split(",", maxsplit=1)[1])
    with Image.open(BytesIO(resized_bytes)) as resized:
        assert resized.size == (100, 50)


@pytest.mark.unit
def test_timeout_becomes_safe_provider_error(
    azure_config: dict[str, Any],
    png_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超时错误指出调用阶段，但不能回显内部 endpoint 或密钥。"""

    def failing_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("private Azure endpoint")

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", failing_urlopen)

    with pytest.raises(
        LLMProviderError,
        match="azure_vision TimeoutError: vision request failed",
    ) as error:
        AzureVisionLLM(azure_config).chat_with_image("caption", png_bytes)

    assert "private Azure endpoint" not in str(error.value)
    assert azure_config["api_key"] not in str(error.value)


@pytest.mark.unit
def test_authentication_failure_includes_safe_http_status_code(
    azure_config: dict[str, Any],
    png_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure 401 必须可诊断，同时不能泄露 URL、响应正文或 api-key。"""

    def failing_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized: secret response",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", failing_urlopen)

    with pytest.raises(
        LLMProviderError,
        match="azure_vision HTTPError 401: vision request failed",
    ) as error:
        AzureVisionLLM(azure_config).chat_with_image("caption", png_bytes)

    assert "example.openai.azure.com" not in str(error.value)
    assert "secret response" not in str(error.value)
    assert azure_config["api_key"] not in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "image", "message"),
    [
        ("", b"image", "text must be a non-empty string"),
        ("caption", b"", "image bytes must not be empty"),
        ("caption", "missing-file.png", "existing file path or valid base64"),
        ("caption", b"not-an-image", "not a supported image"),
    ],
)
def test_invalid_inputs_fail_before_http_request(
    text: str,
    image: str | bytes,
    message: str,
    azure_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """明显错误的文字或图片必须在本地失败，不能浪费 Azure 调用。"""
    http_called = False

    def fake_urlopen(request: Request, timeout: float) -> FakeHTTPResponse:
        nonlocal http_called
        http_called = True
        return FakeHTTPResponse()

    monkeypatch.setattr("libs.llm.base_http_llm.urlopen", fake_urlopen)

    with pytest.raises(ValueError, match=message):
        AzureVisionLLM(azure_config).chat_with_image(text, image)

    assert http_called is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_change", "message"),
    [
        ({"api_key": ""}, r"vision_llm\.api_key"),
        ({"azure_endpoint": ""}, r"vision_llm\.azure_endpoint"),
        ({"deployment_name": ""}, r"vision_llm\.deployment_name"),
        ({"api_version": ""}, r"vision_llm\.api_version"),
        ({"max_image_size": 0}, "max_image_size"),
        ({"max_image_size": True}, "max_image_size"),
    ],
)
def test_invalid_configuration_fails_during_construction(
    config_change: dict[str, Any],
    message: str,
    azure_config: dict[str, Any],
) -> None:
    """必填配置在创建 Provider 时一次检查清楚，不等到图片处理中途才报错。"""
    azure_config.update(config_change)

    with pytest.raises(ValueError, match=message):
        AzureVisionLLM(azure_config)
