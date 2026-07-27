"""Configuration-driven factory for vector store backends.

Backends register a builder under a case-insensitive name. The factory reads
``settings.vector_store.provider``, selects the matching builder, and gives it
an isolated copy of the vector store configuration section.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore


VectorStoreBuilder = Callable[[Mapping[str, Any]], BaseVectorStore]


class VectorStoreFactory:
    """Create registered backends from ``Settings.vector_store``."""

    _providers: ClassVar[dict[str, VectorStoreBuilder]] = {}

    @classmethod
    def register(
        cls, provider: str, builder: VectorStoreBuilder, *, replace: bool = False
    ) -> None:
        """Register a backend builder under a case-insensitive name.

        Args:
            provider: Non-empty backend identifier, such as ``chroma``.
            builder: Callable that accepts the vector store configuration mapping.
            replace: Whether an existing registration may be replaced.

        Raises:
            TypeError: If ``builder`` is not callable.
            ValueError: If the provider name is invalid or already registered.
        """
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Vector store provider builder must be callable")
        if provider_name in cls._providers and not replace:
            raise ValueError(
                f"Vector store provider '{provider_name}' is already registered"
            )
        cls._providers[provider_name] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a backend registration if it exists."""
        cls._providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a sorted snapshot of registered backend names."""
        return sorted(cls._providers)

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """Build the provider selected by ``settings.vector_store.provider``.

        Args:
            settings: Application settings containing vector store configuration.

        Returns:
            A backend instance implementing :class:`BaseVectorStore`.

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
                f"Unsupported vector store provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        vector_store = builder(dict(config))
        if not isinstance(vector_store, BaseVectorStore):
            raise TypeError(
                f"Builder for vector store provider '{provider_name}' must return "
                f"BaseVectorStore, got {type(vector_store).__name__}"
            )
        return vector_store

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        """Return vector store config after validating its container type."""
        config = settings.vector_store
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'vector_store' must be a mapping")
        return config

    @classmethod
    def _get_provider_name(cls, config: Mapping[str, Any]) -> str:
        """Read and normalize the configured vector store provider."""
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError(
                "Missing required configuration field: vector_store.provider"
            )
        if not isinstance(provider, str):
            raise TypeError(
                "Configuration field 'vector_store.provider' must be a string"
            )
        return cls._normalize_provider(provider)

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        """Normalize a registry key and reject ambiguous identifiers."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Vector store provider name must be a non-empty string")
        return provider.strip().lower()
