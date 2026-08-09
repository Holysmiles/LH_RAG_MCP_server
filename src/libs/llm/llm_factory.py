"""Configuration-driven factories for text and Vision language model providers.

Providers register a builder under a case-insensitive name. The factory reads
``settings.llm.provider`` for text chat and ``settings.vision_llm.provider``
for image-aware chat. The two registries are deliberately separate: a provider
that can generate text must not accidentally be treated as one that can read
images.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from core.settings import Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM


LLMBuilder = Callable[[Mapping[str, Any]], BaseLLM]
VisionLLMBuilder = Callable[[Mapping[str, Any]], BaseVisionLLM]


class LLMFactory:
    """Create registered LLM implementations from ``Settings.llm``.

    The registry stores builders rather than concrete classes so providers can
    use either a class constructor or a dedicated construction function.
    """

    _providers: ClassVar[dict[str, LLMBuilder]] = {
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
    }
    # B8 只提供接口和工厂。B9 会把 AzureVisionLLM 注册到这张表，避免 B8
    # 因提前导入尚不存在的具体实现而破坏普通文本 LLM 的启动。
    _vision_providers: ClassVar[dict[str, VisionLLMBuilder]] = {}

    @classmethod
    def register(
        cls, provider: str, builder: LLMBuilder, *, replace: bool = False
    ) -> None:
        """Register a provider builder under a case-insensitive name.

        Args:
            provider: Non-empty provider identifier, such as ``openai``.
            builder: Callable that accepts the LLM configuration mapping.
            replace: Whether an existing registration may be replaced.

        Raises:
            TypeError: If ``builder`` is not callable.
            ValueError: If the provider name is invalid or already registered.
        """
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("LLM provider builder must be callable")
        if provider_name in cls._providers and not replace:
            raise ValueError(f"LLM provider '{provider_name}' is already registered")
        cls._providers[provider_name] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if it exists."""
        cls._providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a sorted snapshot of registered provider names."""
        return sorted(cls._providers)

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """Build the provider selected by ``settings.llm.provider``.

        Args:
            settings: Application settings containing the LLM configuration.

        Returns:
            A provider instance implementing :class:`BaseLLM`.

        Raises:
            TypeError: If the configuration shape or builder result is invalid.
            ValueError: If the provider is missing, unknown, or unregistered.
        """
        config = cls._get_config(settings)
        provider_name = cls._get_provider_name(config)

        builder = cls._providers.get(provider_name)
        if builder is None:
            registered = ", ".join(cls.list_providers()) or "none"
            raise ValueError(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        # A provider may normalize or consume its input during initialization.
        # Passing a copy keeps the shared Settings object unchanged.
        llm = builder(dict(config))
        if not isinstance(llm, BaseLLM):
            raise TypeError(
                f"Builder for LLM provider '{provider_name}' must return BaseLLM, "
                f"got {type(llm).__name__}"
            )
        return llm

    @classmethod
    def register_vision(
        cls,
        provider: str,
        builder: VisionLLMBuilder,
        *,
        replace: bool = False,
    ) -> None:
        """注册一个 Vision LLM Provider。

        输入是供应商名字和构造函数；构造函数接收 ``vision_llm`` 配置副本，输出
        必须是 ``BaseVisionLLM``。这套注册表和文字 LLM 注册表完全分离，防止
        ImageCaptioner 在运行时拿到不支持图片输入的普通聊天模型。
        """
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Vision LLM provider builder must be callable")
        if provider_name in cls._vision_providers and not replace:
            raise ValueError(
                f"Vision LLM provider '{provider_name}' is already registered"
            )
        cls._vision_providers[provider_name] = builder

    @classmethod
    def unregister_vision(cls, provider: str) -> None:
        """移除一个 Vision Provider 注册，仅影响图片模型路由。"""
        cls._vision_providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """返回已注册的 Vision Provider 名称快照，供诊断和测试使用。"""
        return sorted(cls._vision_providers)

    @classmethod
    def create_vision_llm(cls, settings: Settings) -> BaseVisionLLM:
        """按 ``settings.vision_llm`` 创建能接收图片的模型。

        输入：包含 ``provider`` 的可选 ``vision_llm`` 配置段。
        输出：一个实现 ``chat_with_image(text, image_path, trace)`` 的 Provider。
        空配置、未知 Provider 或错误的构造结果都会尽早报错，让摄取层能选择
        降级，而不是在图片处理到一半才得到模糊的属性错误。
        """
        config = cls._get_vision_config(settings)
        provider_name = cls._get_vision_provider_name(config)

        builder = cls._vision_providers.get(provider_name)
        if builder is None:
            registered = ", ".join(cls.list_vision_providers()) or "none"
            raise ValueError(
                f"Unsupported Vision LLM provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        # Provider 可自由规范化配置；传入副本保证全局 Settings 不被修改。
        vision_llm = builder(dict(config))
        if not isinstance(vision_llm, BaseVisionLLM):
            raise TypeError(
                f"Builder for Vision LLM provider '{provider_name}' must return "
                f"BaseVisionLLM, got {type(vision_llm).__name__}"
            )
        return vision_llm

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        """Return the LLM configuration after validating its container type."""
        config = settings.llm
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'llm' must be a mapping")
        return config

    @staticmethod
    def _get_vision_config(settings: Settings) -> Mapping[str, Any]:
        """读取 Vision 配置，并在配置容器错误时给出精确错误。"""
        config = settings.vision_llm
        if not isinstance(config, Mapping):
            raise TypeError(
                "Configuration section 'vision_llm' must be a mapping"
            )
        return config

    @classmethod
    def _get_provider_name(cls, config: Mapping[str, Any]) -> str:
        """Read and normalize the provider configured in the LLM section."""
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError("Missing required configuration field: llm.provider")
        if not isinstance(provider, str):
            raise TypeError("Configuration field 'llm.provider' must be a string")
        return cls._normalize_provider(provider)

    @classmethod
    def _get_vision_provider_name(cls, config: Mapping[str, Any]) -> str:
        """读取 ``vision_llm.provider`` 并复用统一的名称规范化规则。"""
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError(
                "Missing required configuration field: vision_llm.provider"
            )
        if not isinstance(provider, str):
            raise TypeError(
                "Configuration field 'vision_llm.provider' must be a string"
            )
        return cls._normalize_provider(provider)

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        """Normalize a registry key and reject ambiguous identifiers."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("LLM provider name must be a non-empty string")
        return provider.strip().lower()
