"""Configuration-driven factory for text language model providers.

Providers register a builder under a case-insensitive name. The factory reads
``settings.llm.provider``, selects the matching builder, and gives it an
isolated copy of the LLM configuration section.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from core.settings import Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.openai_llm import OpenAILLM


LLMBuilder = Callable[[Mapping[str, Any]], BaseLLM]


class LLMFactory:
    """Create registered LLM implementations from ``Settings.llm``.

    The registry stores builders rather than concrete classes so providers can
    use either a class constructor or a dedicated construction function.
    """

    _providers: ClassVar[dict[str, LLMBuilder]] = {
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "openai": OpenAILLM,
    }

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

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        """Return the LLM configuration after validating its container type."""
        config = settings.llm
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'llm' must be a mapping")
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

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        """Normalize a registry key and reject ambiguous identifiers."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("LLM provider name must be a non-empty string")
        return provider.strip().lower()
