"""Configuration-driven factory for text splitter strategies.

Strategies register a builder under a case-insensitive name. The factory reads
``settings.splitter.provider``, selects the matching builder, and gives it an
isolated copy of the splitter configuration section.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter


SplitterBuilder = Callable[[Mapping[str, Any]], BaseSplitter]


class SplitterFactory:
    """Create registered splitter implementations from ``Settings.splitter``."""

    _providers: ClassVar[dict[str, SplitterBuilder]] = {}

    @classmethod
    def register(
        cls, provider: str, builder: SplitterBuilder, *, replace: bool = False
    ) -> None:
        """Register a splitter builder under a case-insensitive name.

        Args:
            provider: Non-empty strategy identifier, such as ``recursive``.
            builder: Callable that accepts the splitter configuration mapping.
            replace: Whether an existing registration may be replaced.

        Raises:
            TypeError: If ``builder`` is not callable.
            ValueError: If the provider name is invalid or already registered.
        """
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Splitter provider builder must be callable")
        if provider_name in cls._providers and not replace:
            raise ValueError(
                f"Splitter provider '{provider_name}' is already registered"
            )
        cls._providers[provider_name] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a splitter registration if it exists."""
        cls._providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a sorted snapshot of registered splitter names."""
        return sorted(cls._providers)

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """Build the provider selected by ``settings.splitter.provider``.

        Args:
            settings: Application settings containing splitter configuration.

        Returns:
            A splitter instance implementing :class:`BaseSplitter`.

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
                f"Unsupported splitter provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        splitter = builder(dict(config))
        if not isinstance(splitter, BaseSplitter):
            raise TypeError(
                f"Builder for splitter provider '{provider_name}' must return "
                f"BaseSplitter, got {type(splitter).__name__}"
            )
        return splitter

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        """Return splitter configuration after validating its container type."""
        config = settings.splitter
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'splitter' must be a mapping")
        return config

    @classmethod
    def _get_provider_name(cls, config: Mapping[str, Any]) -> str:
        """Read and normalize the configured splitter provider."""
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError("Missing required configuration field: splitter.provider")
        if not isinstance(provider, str):
            raise TypeError("Configuration field 'splitter.provider' must be a string")
        return cls._normalize_provider(provider)

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        """Normalize a registry key and reject ambiguous identifiers."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Splitter provider name must be a non-empty string")
        return provider.strip().lower()
