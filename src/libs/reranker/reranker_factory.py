"""Configuration-driven factory and safe fallback for reranking backends.

Backends register a builder under a case-insensitive name. The factory reads
``settings.rerank.provider`` and includes a built-in ``none`` backend that
preserves the incoming candidate order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, CandidateT
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker


class NoneReranker(BaseReranker):
    """Disable reranking while preserving candidate objects and order."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def rerank(
        self,
        query: str,
        candidates: Sequence[CandidateT],
        trace: object | None = None,
    ) -> list[CandidateT]:
        """Return a shallow list copy without changing candidate order."""
        return list(candidates)


RerankerBuilder = Callable[[Mapping[str, Any]], BaseReranker]


class RerankerFactory:
    """Create registered reranking backends from ``Settings.rerank``."""

    _providers: ClassVar[dict[str, RerankerBuilder]] = {
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
        "none": NoneReranker,
    }

    @classmethod
    def register(
        cls, provider: str, builder: RerankerBuilder, *, replace: bool = False
    ) -> None:
        """Register a backend builder under a case-insensitive name.

        Args:
            provider: Non-empty backend identifier, such as ``llm``.
            builder: Callable that accepts the reranker configuration mapping.
            replace: Whether an existing registration may be replaced.

        Raises:
            TypeError: If ``builder`` is not callable.
            ValueError: If the provider name is invalid or already registered.
        """
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Reranker provider builder must be callable")
        if provider_name in cls._providers and not replace:
            raise ValueError(
                f"Reranker provider '{provider_name}' is already registered"
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
    def create(cls, settings: Settings) -> BaseReranker:
        """Build the provider selected by ``settings.rerank.provider``.

        Args:
            settings: Application settings containing reranker configuration.

        Returns:
            A backend instance implementing :class:`BaseReranker`.

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
                f"Unsupported reranker provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        builder_config = dict(config)
        if builder is LLMReranker and "llm" not in builder_config:
            # The main model already lives in settings.llm.  Reuse that section
            # when LLM reranking is enabled, so users do not have to duplicate
            # provider, model, endpoint, and API-key settings under rerank.
            if not isinstance(settings.llm, Mapping):
                raise TypeError("Configuration section 'llm' must be a mapping")
            builder_config["llm"] = dict(settings.llm)

        reranker = builder(builder_config)
        if not isinstance(reranker, BaseReranker):
            raise TypeError(
                f"Builder for reranker provider '{provider_name}' must return "
                f"BaseReranker, got {type(reranker).__name__}"
            )
        return reranker

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        """Return reranker config after validating its container type."""
        config = settings.rerank
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'rerank' must be a mapping")
        return config

    @classmethod
    def _get_provider_name(cls, config: Mapping[str, Any]) -> str:
        """Read and normalize the configured reranker provider."""
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError("Missing required configuration field: rerank.provider")
        if not isinstance(provider, str):
            raise TypeError("Configuration field 'rerank.provider' must be a string")
        return cls._normalize_provider(provider)

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        """Normalize a registry key and reject ambiguous identifiers."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Reranker provider name must be a non-empty string")
        return provider.strip().lower()
