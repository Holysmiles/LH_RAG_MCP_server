"""Unit tests for custom retrieval metrics and evaluator factory routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class ConstantEvaluator(BaseEvaluator):
    """Small fake used to verify registration and routing."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        return {"constant": 1.0}


@pytest.mark.unit
def test_base_evaluator_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseEvaluator()


@pytest.mark.unit
def test_custom_evaluator_reports_first_rank_match() -> None:
    metrics = CustomEvaluator().evaluate(
        query="How is Azure configured?",
        retrieved_ids=["chunk-irrelevant", "chunk-answer", "chunk-other"],
        golden_ids=["chunk-answer"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


@pytest.mark.unit
def test_custom_evaluator_uses_first_of_multiple_matches() -> None:
    metrics = CustomEvaluator().evaluate(
        query="query",
        retrieved_ids=["chunk-1", "chunk-2", "chunk-3"],
        golden_ids=["chunk-3", "chunk-1"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 1.0}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("retrieved_ids", "golden_ids"),
    [
        (["chunk-1"], ["chunk-2"]),
        ([], ["chunk-1"]),
        (["chunk-1"], []),
        ([], []),
    ],
)
def test_custom_evaluator_returns_zero_without_a_match(
    retrieved_ids: list[str], golden_ids: list[str]
) -> None:
    metrics = CustomEvaluator().evaluate("query", retrieved_ids, golden_ids)

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


@pytest.mark.unit
def test_factory_creates_builtin_custom_evaluator() -> None:
    config = {"provider": "custom", "enabled": False}
    settings = Settings(evaluation=config)

    evaluator = EvaluatorFactory.create(settings)

    assert isinstance(evaluator, CustomEvaluator)
    assert evaluator.config == config
    assert evaluator.config is not settings.evaluation


@pytest.mark.unit
def test_factory_normalizes_provider_name() -> None:
    settings = Settings(evaluation={"provider": "  CUSTOM  "})

    assert isinstance(EvaluatorFactory.create(settings), CustomEvaluator)


@pytest.mark.unit
def test_factory_routes_to_registered_evaluator() -> None:
    EvaluatorFactory.register("constant", ConstantEvaluator)
    try:
        evaluator = EvaluatorFactory.create(
            Settings(evaluation={"provider": "constant"})
        )

        assert isinstance(evaluator, ConstantEvaluator)
        assert evaluator.evaluate("query", [], []) == {"constant": 1.0}
    finally:
        EvaluatorFactory.unregister("constant")


@pytest.mark.unit
def test_factory_reports_missing_provider() -> None:
    with pytest.raises(ValueError, match=r"evaluation\.provider"):
        EvaluatorFactory.create(Settings(evaluation={}))


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    settings = Settings(evaluation={"provider": "missing"})

    with pytest.raises(ValueError, match="Unsupported evaluator provider 'missing'"):
        EvaluatorFactory.create(settings)


@pytest.mark.unit
def test_factory_rejects_non_mapping_config() -> None:
    settings = Settings(evaluation=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"section 'evaluation'.*mapping"):
        EvaluatorFactory.create(settings)


@pytest.mark.unit
def test_duplicate_registration_requires_explicit_replace() -> None:
    with pytest.raises(ValueError, match="already registered"):
        EvaluatorFactory.register("CUSTOM", CustomEvaluator)
