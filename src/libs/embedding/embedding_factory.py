"""Configuration-driven factory for text embedding providers.

Providers register a builder under a case-insensitive name. The factory reads
``settings.embedding.provider``, selects the matching builder, and gives it an
isolated copy of the embedding configuration section.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.openai_embedding import OpenAIEmbedding


EmbeddingBuilder = Callable[[Mapping[str, Any]], BaseEmbedding]


class EmbeddingFactory:
    """Create registered embedding implementations from ``Settings.embedding``."""

    _providers: ClassVar[dict[str, EmbeddingBuilder]] = {
        "azure": AzureEmbedding,
        "openai": OpenAIEmbedding,
    }

    @classmethod
    def register(
        cls, provider: str, builder: EmbeddingBuilder, *, replace: bool = False
    ) -> None:
        """Register a provider builder under a case-insensitive name.

        Args:
            provider: Non-empty provider identifier, such as ``openai``.
            builder: Callable that accepts the embedding configuration mapping.
            replace: Whether an existing registration may be replaced.

        Raises:
            TypeError: If ``builder`` is not callable.
            ValueError: If the provider name is invalid or already registered.
        """
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Embedding provider builder must be callable")
        if provider_name in cls._providers and not replace:
            raise ValueError(
                f"Embedding provider '{provider_name}' is already registered"
            )
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
    def create(cls, settings: Settings) -> BaseEmbedding:
        """Build the provider selected by ``settings.embedding.provider``.

        Args:
            settings: Application settings containing embedding configuration.

        Returns:
            A provider instance implementing :class:`BaseEmbedding`.

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
                f"Unsupported embedding provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        embedding = builder(dict(config))
        if not isinstance(embedding, BaseEmbedding):
            raise TypeError(
                f"Builder for embedding provider '{provider_name}' must return "
                f"BaseEmbedding, got {type(embedding).__name__}"
            )
        return embedding

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        """Return embedding configuration after validating its container type."""
        config = settings.embedding
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'embedding' must be a mapping")
        return config

    @classmethod
    def _get_provider_name(cls, config: Mapping[str, Any]) -> str:
        """Read and normalize the configured embedding provider."""
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError("Missing required configuration field: embedding.provider")
        if not isinstance(provider, str):
            raise TypeError("Configuration field 'embedding.provider' must be a string")
        return cls._normalize_provider(provider)

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        """Normalize a registry key and reject ambiguous identifiers."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Embedding provider name must be a non-empty string")
        return provider.strip().lower()
