"""Unit tests for custom retrieval metrics and evaluator factory routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
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


@pytest.mark.unit
@pytest.mark.parametrize(("retrieved", "golden", "expected_mrr"), [
    (("noise", "noise", "answer", "answer"), ("answer", "answer"), 1 / 3),
    (("ANSWER", "answer"), ("answer",), .5),
    (("answer-other", "answer"), ("answer",), .5),
    (("answer",), ("answer", "missing"), 1.),
])
def test_metrics_use_exact_ids_and_original_rank_without_mutating_inputs(retrieved, golden, expected_mrr):
    retrieved, golden = list(retrieved), list(golden)
    before = deepcopy((retrieved, golden))
    metrics = CustomEvaluator().evaluate("query", retrieved, golden, trace=TraceContext())
    assert metrics == {"hit_rate": 1., "mrr": expected_mrr}
    assert all(type(value) is float and math.isfinite(value) and 0 <= value <= 1 for value in metrics.values())
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics
    assert (retrieved, golden) == before


@pytest.mark.unit
def test_custom_sample_contract_accepts_tuples_and_ignores_semantic_fields():
    evaluator: BaseEvaluator = CustomEvaluator()
    kwargs = {"answer": "unrelated", "contexts": ("unrelated context",), "reference": "different answer"}
    metrics = evaluator.evaluate_sample("query", ("wrong", "answer"), ("answer",), **kwargs)
    assert metrics == {"hit_rate": 1., "mrr": .5}
    metrics["hit_rate"] = -1
    assert evaluator.evaluate("query", ("answer",), ("answer",)) == {"hit_rate": 1., "mrr": 1.}


@pytest.mark.unit
def test_base_sample_adapter_preserves_legacy_evaluator_inputs_and_trace(monkeypatch):
    evaluator = ConstantEvaluator({})
    trace = TraceContext()
    retrieved, golden = ["retrieved"], ["golden"]
    result = {"legacy": .75}
    def evaluate(query, retrieved_ids, golden_ids, trace=None):
        assert query == "query" and retrieved_ids is retrieved and golden_ids is golden
        assert trace is expected_trace
        return result
    expected_trace = trace
    monkeypatch.setattr(evaluator, "evaluate", evaluate)
    assert evaluator.evaluate_sample("query", retrieved, golden, trace,
                                     answer="answer", contexts=["context"], reference="reference") is result


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["", "  ", 42, True])
def test_factory_rejects_invalid_provider_identifiers(provider):
    with pytest.raises((ValueError, TypeError), match=r"evaluation\.provider"):
        EvaluatorFactory.create(Settings(evaluation={"provider": provider}))


@pytest.mark.unit
def test_factory_rejects_invalid_builder_and_result(monkeypatch):
    monkeypatch.setattr(EvaluatorFactory, "_providers", dict(EvaluatorFactory._providers))
    with pytest.raises(TypeError, match="callable"):
        EvaluatorFactory.register("invalid", None)
    EvaluatorFactory.register("invalid", lambda config: object())
    with pytest.raises(TypeError, match="must return BaseEvaluator"):
        EvaluatorFactory.create(Settings(evaluation={"provider": "invalid"}))
    EvaluatorFactory.register(" INVALID ", ConstantEvaluator, replace=True)
    assert EvaluatorFactory.create(Settings(evaluation={"provider": "invalid"})).evaluate("q", [], []) == {"constant": 1.}
    providers = EvaluatorFactory.list_providers()
    assert providers == sorted(set(providers))
    providers.clear()
    assert "custom" in EvaluatorFactory.list_providers()
