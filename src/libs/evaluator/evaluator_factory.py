"""Configuration-driven factory for evaluation backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


EvaluatorBuilder = Callable[[Mapping[str, Any]], BaseEvaluator]


def _create_ragas(config: Mapping[str, Any]) -> BaseEvaluator:
    from observability.evaluation.ragas_evaluator import RagasEvaluator

    return RagasEvaluator(config)


class EvaluatorFactory:
    """Create registered evaluators from ``Settings.evaluation``."""

    _providers: ClassVar[dict[str, EvaluatorBuilder]] = {
        "custom": CustomEvaluator,
        "ragas": _create_ragas,
    }

    @classmethod
    def register(
        cls, provider: str, builder: EvaluatorBuilder, *, replace: bool = False
    ) -> None:
        """Register an evaluator builder under a case-insensitive name."""
        provider_name = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Evaluator provider builder must be callable")
        if provider_name in cls._providers and not replace:
            raise ValueError(
                f"Evaluator provider '{provider_name}' is already registered"
            )
        cls._providers[provider_name] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove an evaluator registration if it exists."""
        cls._providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a sorted snapshot of registered evaluator names."""
        return sorted(cls._providers)

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """Build the provider selected by ``settings.evaluation.provider``."""
        config = cls._get_config(settings)
        provider_name = cls._get_provider_name(config)

        builder = cls._providers.get(provider_name)
        if builder is None:
            registered = ", ".join(cls.list_providers()) or "none"
            raise ValueError(
                f"Unsupported evaluator provider '{provider_name}'. "
                f"Registered providers: {registered}"
            )

        evaluator = builder(dict(config))
        if not isinstance(evaluator, BaseEvaluator):
            raise TypeError(
                f"Builder for evaluator provider '{provider_name}' must return "
                f"BaseEvaluator, got {type(evaluator).__name__}"
            )
        return evaluator

    @staticmethod
    def _get_config(settings: Settings) -> Mapping[str, Any]:
        config = settings.evaluation
        if not isinstance(config, Mapping):
            raise TypeError("Configuration section 'evaluation' must be a mapping")
        return config

    @classmethod
    def _get_provider_name(cls, config: Mapping[str, Any]) -> str:
        provider = config.get("provider")
        if provider is None or (isinstance(provider, str) and not provider.strip()):
            raise ValueError(
                "Missing required configuration field: evaluation.provider"
            )
        if not isinstance(provider, str):
            raise TypeError(
                "Configuration field 'evaluation.provider' must be a string"
            )
        return cls._normalize_provider(provider)

    @staticmethod
    def _normalize_provider(provider: object) -> str:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Evaluator provider name must be a non-empty string")
        return provider.strip().lower()
