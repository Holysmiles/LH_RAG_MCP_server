"""Concurrent evaluation, complete-result semantics and factory composition."""

from threading import Barrier
from types import SimpleNamespace

import pytest

from core.settings import Settings
from core.trace import TraceContext
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.composite_evaluator import CompositeEvaluator
from observability.evaluation import ragas_evaluator


pytestmark = pytest.mark.unit


class Evaluator(BaseEvaluator):
    def __init__(self, result=None, action=None):
        self.result = {"score": .5} if result is None else result
        self.action = action

    def evaluate(self, query, retrieved_ids, golden_ids, trace=None):
        if self.action is not None:
            return self.action(query, retrieved_ids, golden_ids, trace)
        return self.result


def test_backends_really_run_in_parallel_and_propagate_trace():
    barrier = Barrier(2)
    trace = TraceContext()
    def action(name):
        def evaluate(query, retrieved, golden, seen_trace):
            assert query == "query" and retrieved == ["id"] and golden == ["gold"]
            assert seen_trace is trace
            barrier.wait(timeout=3)
            seen_trace.record_stage(name, elapsed_ms=1)
            return {name: .75}
        return evaluate
    composite = CompositeEvaluator([Evaluator(action=action("first")), Evaluator(action=action("second"))])
    assert composite.evaluate("query", ["id"], ["gold"], trace) == {"first": .75, "second": .75}
    assert {stage["stage"] for stage in trace.stages} == {"first", "second"}


def test_backend_input_mutation_is_isolated():
    barrier = Barrier(2)
    def mutate(query, retrieved, golden, trace):
        retrieved.clear()
        golden.append("other")
        barrier.wait(timeout=3)
        return {"mutator": 1}
    def observe(query, retrieved, golden, trace):
        barrier.wait(timeout=3)
        assert retrieved == ["a"] and golden == ["a"]
        return {"observer": 2}
    retrieved, golden = ["a"], ["a"]
    result = CompositeEvaluator([Evaluator(action=mutate), Evaluator(action=observe)]).evaluate("q", retrieved, golden)
    assert result == {"mutator": 1., "observer": 2.}
    assert retrieved == golden == ["a"]


def test_backend_exception_propagates_instead_of_partial_success():
    def fail(*args):
        raise TimeoutError("judge unavailable")
    with pytest.raises(TimeoutError):
        CompositeEvaluator([CustomEvaluator(), Evaluator(action=fail)]).evaluate("q", ["a"], ["a"])


def test_duplicate_metric_names_are_rejected_even_for_identical_values():
    with pytest.raises(ValueError, match="Duplicate metric"):
        CompositeEvaluator([Evaluator(), Evaluator()]).evaluate("q", [], [])


@pytest.mark.parametrize("result", [[], "invalid", {"score": float("nan")}, {"score": float("inf")},
                                     {"score": True}, {"score": "0.5"}, {"": 1}])
def test_invalid_backend_outputs_are_rejected(result):
    with pytest.raises((ValueError, TypeError)):
        CompositeEvaluator([Evaluator(result)]).evaluate("q", [], [])


@pytest.mark.parametrize("evaluators", [[], None, "custom", [object()]])
def test_invalid_backend_sequences_are_rejected(evaluators):
    with pytest.raises(ValueError):
        CompositeEvaluator(evaluators)


def test_duplicate_instances_are_rejected_and_caller_list_is_not_retained():
    evaluator = CustomEvaluator()
    with pytest.raises(ValueError, match="distinct"):
        CompositeEvaluator([evaluator, evaluator])
    backends = [evaluator]
    composite = CompositeEvaluator(backends)
    backends.clear()
    assert composite.evaluate("q", ["a"], ["a"]) == {"hit_rate": 1., "mrr": 1.}


def test_factory_composes_actual_ragas_adapter_and_custom_metrics(monkeypatch):
    constructors = SimpleNamespace(
        Faithfulness=lambda: SimpleNamespace(name="faithfulness"),
        AnswerRelevancy=lambda: SimpleNamespace(name="answer_relevancy"))
    ragas = SimpleNamespace(EvaluationDataset=SimpleNamespace(from_list=lambda rows: rows),
                            evaluate=lambda **kwargs: {metric.name: [.8] for metric in kwargs["metrics"]})
    monkeypatch.setattr(ragas_evaluator, "import_module", lambda name: ragas if name == "ragas" else constructors)
    config = {"provider": "custom", "backends": ["ragas", "custom"],
              "samples": {"q": {"answer": "answer", "contexts": ["context"]}}}
    composite = EvaluatorFactory.create(Settings(evaluation=config))
    assert isinstance(composite, CompositeEvaluator)
    assert composite.evaluate("q", ["other", "correct"], ["correct"]) == {
        "faithfulness": .8, "answer_relevancy": .8, "hit_rate": 1., "mrr": .5}
    assert config["backends"] == ["ragas", "custom"] and config["provider"] == "custom"


def test_per_backend_config_overrides_defaults_without_mutating_settings(monkeypatch):
    seen = []
    def build(config):
        seen.append(config)
        return Evaluator({config["provider"]: config["value"]})
    monkeypatch.setattr(EvaluatorFactory, "_providers", {"one": build, "two": build})
    config = {"value": 1, "backends": ["one", {"provider": "two", "value": 2}]}
    composite = EvaluatorFactory.create(Settings(evaluation=config))
    assert composite.evaluate("q", [], []) == {"one": 1., "two": 2.}
    assert "backends" not in seen[0] and config["value"] == 1


@pytest.mark.parametrize("backends", [[], None, "custom", [1], ["custom", " CUSTOM "],
                                       [{"provider": "custom", "backends": ["custom"]}], [{}], ["missing"]])
def test_factory_rejects_invalid_backend_config(backends):
    with pytest.raises((ValueError, TypeError)):
        EvaluatorFactory.create(Settings(evaluation={"backends": backends}))
