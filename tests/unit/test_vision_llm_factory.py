"""B8 Vision LLM 契约和工厂的离线单元测试。

测试使用 FakeVisionLLM，不读取真实图片、不调用网络。它验证的不是某个厂商
API，而是 ImageCaptioner 将来依赖的稳定边界：输入一段指令和一张图片，得到一段
文字；工厂只会路由到明确声明支持图片的 Provider。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from core.settings import Settings
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse, ImageInput
from libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    """确定性假模型：记录输入，用文字回显图片类型，方便验证工厂契约。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.calls: list[dict[str, object]] = []

    def chat_with_image(
        self,
        text: str,
        image_path: ImageInput,
        trace: object | None = None,
    ) -> ChatResponse:
        prepared_image = self.preprocess_image(image_path)
        self.calls.append(
            {
                "text": text,
                "image": prepared_image,
                "trace": trace,
            }
        )
        return f"{self.config['model']}:{text}:{type(prepared_image).__name__}"


@pytest.fixture
def registered_fake_vision_provider() -> Iterator[None]:
    """每个测试独立注册/注销假 Provider，避免污染其他 LLM 工厂测试。"""
    LLMFactory.register_vision("fake_vision", FakeVisionLLM)
    yield
    LLMFactory.unregister_vision("fake_vision")


@pytest.mark.unit
def test_base_vision_llm_cannot_be_instantiated() -> None:
    """抽象契约缺少 chat_with_image 实现时不能被直接使用。"""
    with pytest.raises(TypeError):
        BaseVisionLLM()


@pytest.mark.unit
def test_factory_routes_text_and_image_inputs_to_vision_provider(
    registered_fake_vision_provider: None,
) -> None:
    """工厂应传递完整配置；Provider 应接收指令和图片字节并输出文字。"""
    settings = Settings(
        vision_llm={
            "provider": "fake_vision",
            "model": "caption-test-model",
            "max_image_size": 2048,
        }
    )

    vision_llm = LLMFactory.create_vision_llm(settings)
    response = vision_llm.chat_with_image("Describe this chart", b"image-bytes")

    assert isinstance(vision_llm, FakeVisionLLM)
    assert response == "caption-test-model:Describe this chart:bytes"
    assert vision_llm.calls == [
        {
            "text": "Describe this chart",
            "image": b"image-bytes",
            "trace": None,
        }
    ]


@pytest.mark.unit
def test_factory_normalizes_vision_provider_and_copies_config(
    registered_fake_vision_provider: None,
) -> None:
    """Provider 名称大小写不敏感，且 Provider 不能修改 Settings 原始配置。"""
    config = {"provider": "  FAKE_VISION  ", "model": "test-model"}
    vision_llm = LLMFactory.create_vision_llm(Settings(vision_llm=config))

    assert isinstance(vision_llm, FakeVisionLLM)
    assert vision_llm.config == config
    assert vision_llm.config is not config


@pytest.mark.unit
def test_default_preprocessor_preserves_path_and_base64_text(
    registered_fake_vision_provider: None,
) -> None:
    """默认扩展点不改动输入；具体 Provider 可覆写它完成压缩或格式转换。"""
    vision_llm = LLMFactory.create_vision_llm(
        Settings(vision_llm={"provider": "fake_vision", "model": "test-model"})
    )
    assert isinstance(vision_llm, FakeVisionLLM)

    image_path = "data:image/png;base64,aGVsbG8="
    assert vision_llm.preprocess_image(image_path) == image_path
    assert vision_llm.chat_with_image("caption", image_path).endswith(":str")


@pytest.mark.unit
def test_vision_registry_isolated_from_text_llm_registry(
    registered_fake_vision_provider: None,
) -> None:
    """普通 create 不能用 Vision Provider，避免文字/图片能力被混用。"""
    settings = Settings(
        llm={"provider": "fake_vision", "model": "text-model"},
        vision_llm={"provider": "fake_vision", "model": "vision-model"},
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider 'fake_vision'"):
        LLMFactory.create(settings)

    assert isinstance(LLMFactory.create_vision_llm(settings), FakeVisionLLM)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("vision_config", "error_type", "message"),
    [
        ({"model": "test-model"}, ValueError, r"vision_llm\.provider"),
        ({"provider": 42}, TypeError, r"vision_llm\.provider.*string"),
        ({"provider": "missing"}, ValueError, "Unsupported Vision LLM provider"),
    ],
)
def test_factory_reports_invalid_vision_configuration(
    vision_config: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    """配置问题在构造时暴露，ImageCaptioner 才能做明确降级。"""
    with pytest.raises(error_type, match=message):
        LLMFactory.create_vision_llm(Settings(vision_llm=vision_config))


@pytest.mark.unit
def test_factory_rejects_non_mapping_vision_config() -> None:
    """vision_llm 必须是字典，不能把字符串等错误形状带到 Provider。"""
    with pytest.raises(TypeError, match="section 'vision_llm'.*mapping"):
        LLMFactory.create_vision_llm(Settings(vision_llm=None))  # type: ignore[arg-type]


@pytest.mark.unit
def test_factory_rejects_vision_builder_with_wrong_return_type() -> None:
    """注册函数返回普通对象时必须失败，保证调用者能安全使用图片接口。"""
    LLMFactory.register_vision("invalid", lambda config: object())  # type: ignore[return-value]
    try:
        settings = Settings(vision_llm={"provider": "invalid", "model": "test-model"})
        with pytest.raises(TypeError, match="must return BaseVisionLLM"):
            LLMFactory.create_vision_llm(settings)
    finally:
        LLMFactory.unregister_vision("invalid")
