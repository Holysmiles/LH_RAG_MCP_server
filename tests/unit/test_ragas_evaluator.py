"""Ragas adapter contracts and real metrics with entirely offline model doubles."""

import json
from types import SimpleNamespace

import pytest

from core.settings import Settings
from libs.evaluator import evaluator_factory
from observability.evaluation import ragas_evaluator as module


pytestmark = pytest.mark.unit


@pytest.fixture
def backend(monkeypatch):
    calls = []
    names = {"Faithfulness": "faithfulness", "AnswerRelevancy": "answer_relevancy", "ContextPrecision": "context_precision"}
    metrics = SimpleNamespace(**{key: (lambda name=value: SimpleNamespace(name=name)) for key, value in names.items()})
    def evaluate(**kwargs):
        calls.append(kwargs)
        return {metric.name: [0.75] for metric in kwargs["metrics"]}
    ragas = SimpleNamespace(EvaluationDataset=SimpleNamespace(from_list=lambda rows: rows), evaluate=evaluate)
    monkeypatch.setattr(module, "import_module", lambda name: ragas if name == "ragas" else metrics)
    return ragas, calls


def test_all_metrics_map_semantic_fields_and_native_clients(backend):
    _, calls = backend
    llm, embeddings = object(), object()
    evaluator = module.RagasEvaluator({"metrics": list(module.RagasEvaluator.supported_metrics)},
                                       llm=llm, embeddings=embeddings)
    scores = evaluator.evaluate("Question?", ["chunk-1"], ["gold-1"],
                                answer="Answer.", contexts=["Context."], reference="Reference.")
    assert scores == {"faithfulness": .75, "answer_relevancy": .75, "context_precision": .75}
    assert calls[0]["dataset"] == [{"user_input": "Question?", "response": "Answer.",
                                   "retrieved_contexts": ["Context."], "reference": "Reference.",
                                   "retrieved_context_ids": ["chunk-1"], "reference_context_ids": ["gold-1"]}]
    assert calls[0]["llm"] is llm and calls[0]["embeddings"] is embeddings
    assert calls[0]["raise_exceptions"] and not calls[0]["show_progress"]
    evaluator.evaluate("Question?", answer="Answer.", contexts=["Context."], reference="Reference.")
    assert calls[0]["metrics"][0] is not calls[1]["metrics"][0]


def test_factory_routes_ragas_and_base_interface_reads_configured_sample(backend):
    evaluator = evaluator_factory.EvaluatorFactory.create(Settings(evaluation={
        "provider": "ragas", "samples": {"query": {"answer": "answer", "contexts": ["context"]}}}))
    assert isinstance(evaluator, module.RagasEvaluator)
    assert evaluator.evaluate("query", ["id"], ["gold"]) == {"faithfulness": .75, "answer_relevancy": .75}


def test_resolver_receives_ids_and_explicit_values_override_it(backend):
    seen = []
    def resolve(*args):
        seen.append(args)
        return {"answer": "resolved", "contexts": ["resolved context"]}
    evaluator = module.RagasEvaluator(sample_resolver=resolve)
    evaluator.evaluate("query", ("id",), ("gold",), answer="explicit")
    assert seen == [("query", ["id"], ["gold"])]
    assert backend[1][0]["dataset"][0]["response"] == "explicit"


@pytest.mark.parametrize("metrics", [[], "faithfulness", ["unknown"], ["faithfulness", "faithfulness"], [None]])
def test_invalid_metrics_are_rejected(metrics, backend):
    with pytest.raises(ValueError):
        module.RagasEvaluator({"metrics": metrics})


@pytest.mark.parametrize("kwargs", [
    {}, {"answer": "answer"}, {"answer": "answer", "contexts": []},
    {"answer": "answer", "contexts": "a context"}, {"answer": "", "contexts": ["context"]},
])
def test_ids_are_not_used_as_contexts_or_answers(kwargs, backend):
    evaluator = module.RagasEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("query", ["id"], ["gold"], **kwargs)
    assert backend[1] == []


def test_context_precision_requires_reference(backend):
    evaluator = module.RagasEvaluator({"metrics": ["context_precision"]})
    with pytest.raises(ValueError, match="reference"):
        evaluator.evaluate("query", answer="answer", contexts=["context"])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.5", None])
def test_invalid_scores_are_not_reported_as_success(score, backend):
    backend[0].evaluate = lambda **kwargs: {metric.name: [score] for metric in kwargs["metrics"]}
    with pytest.raises(ValueError, match="score"):
        module.RagasEvaluator().evaluate("query", answer="answer", contexts=["context"])


def test_missing_dependency_has_actionable_import_error(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError("ragas")
    monkeypatch.setattr(module, "import_module", unavailable)
    with pytest.raises(ImportError, match="ragas>=0.3.9"):
        module.RagasEvaluator()
    assert evaluator_factory.EvaluatorFactory.create(Settings(evaluation={"provider": "custom"}))


def test_backend_errors_propagate(backend):
    def fail(**kwargs):
        raise TimeoutError("judge unavailable")
    backend[0].evaluate = fail
    with pytest.raises(TimeoutError):
        module.RagasEvaluator().evaluate("query", answer="answer", contexts=["context"])


def test_real_ragas_metrics_with_mock_llm_and_embeddings(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    from langchain_core.embeddings import Embeddings
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    class MockLLM(BaseRagasLLM):
        def generate_text(self, prompt, n=1, **kwargs):
            text = prompt.to_string()
            if "noncommittal" in text:
                output = {"question": "Where is Paris?", "noncommittal": 0}
            elif '"verdict"' in text and '"statements"' in text:
                output = {"statements": [{"statement": "Paris is in France.", "reason": "Supported.", "verdict": 1}]}
            elif '"statements"' in text:
                output = {"statements": ["Paris is in France."]}
            else:
                output = {"reason": "Relevant context.", "verdict": 1}
            return LLMResult(generations=[[Generation(text=json.dumps(output)) for _ in range(n)]])
        async def agenerate_text(self, prompt, n=1, **kwargs):
            return self.generate_text(prompt, n=n, **kwargs)
        def is_finished(self, response):
            return True

    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1., 0.] for _ in texts]
        def embed_query(self, text):
            return [1., 0.]

    evaluator = module.RagasEvaluator({"metrics": list(module.RagasEvaluator.supported_metrics)},
                                      llm=MockLLM(), embeddings=MockEmbeddings())
    scores = evaluator.evaluate("Where is Paris?", answer="Paris is in France.",
                                 contexts=["Paris is in France."], reference="Paris is in France.")
    assert scores == pytest.approx({"faithfulness": 1., "answer_relevancy": 1., "context_precision": 1.})
    # Exercise Ragas's event-loop setup inside CompositeEvaluator's worker.
    from libs.evaluator.custom_evaluator import CustomEvaluator
    from observability.evaluation.composite_evaluator import CompositeEvaluator

    evaluator.config["samples"] = {"Where is Paris?": {
        "answer": "Paris is in France.", "contexts": ["Paris is in France."], "reference": "Paris is in France."}}
    combined = CompositeEvaluator([evaluator, CustomEvaluator()]).evaluate("Where is Paris?", ["a"], ["a"])
    assert combined == pytest.approx(dict(scores, hit_rate=1., mrr=1.))
