"""Golden-set validation, relevance labels, macro metrics and semantic payloads."""

import json
from pathlib import Path

import pytest

from core.settings import Settings
from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.composite_evaluator import CompositeEvaluator
from observability.evaluation.eval_runner import EvalRunner


pytestmark = pytest.mark.unit


class Search:
    def __init__(self, results):
        self.results, self.calls = results, []
    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results[query]


def hit(key, source="docs/guide.pdf"):
    return RetrievalResult(key, .5, f"text-{key}", {"source_path": source})


def write_cases(tmp_path, cases):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"test_cases": cases}), encoding="utf-8")
    return path


def settings(**evaluation):
    return Settings(evaluation=evaluation, observability={"trace_enabled": False})


def test_macro_metrics_sources_filters_top_k_and_serializable_report(tmp_path):
    path = write_cases(tmp_path, [
        {"query": "q1", "expected_chunk_ids": ["b"], "collection": "alpha"},
        {"query": "q2", "expected_sources": ["guide.pdf"]},
        {"query": "q3", "expected_chunk_ids": ["absent"]},
    ])
    search = Search({"q1": [hit("a"), hit("b")], "q2": [hit("c")], "q3": []})
    report = EvalRunner(settings(top_k=2), search, CustomEvaluator()).run(path)
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == .5
    assert report.metrics == {"hit_rate": pytest.approx(2 / 3), "mrr": .5}
    assert report.results[1]["expected_chunk_ids"] == []
    assert report.results[1]["metrics"]["hit_rate"] == 1.
    assert search.calls[0][1]["filters"] == {"collection": "alpha"}
    assert search.calls[1][1]["filters"] is None
    assert all(call[1]["top_k"] == 2 for call in search.calls)
    assert json.loads(json.dumps(report.to_dict()))["top_k"] == 2
    assert report.test_set_path == str(path.resolve())


def test_exact_sources_do_not_match_other_directory_and_ids_remain_relevant(tmp_path):
    path = write_cases(tmp_path, [
        {"query": "q", "expected_sources": ["docs/guide.pdf"], "expected_chunk_ids": ["exact"]}])
    report = EvalRunner(settings(top_k=2), Search({"q": [hit("wrong", "elsewhere/guide.pdf"), hit("exact")]}),
                        CustomEvaluator()).run(path)
    assert report.mrr == .5
    assert EvalRunner._source_matches(r"D:\docs\guide.pdf", "guide.pdf")


def test_retrieval_is_truncated_at_top_k_and_default_collection_is_used(tmp_path):
    path = write_cases(tmp_path, [{"query": "q", "expected_chunk_ids": ["b"]}])
    search = Search({"q": [hit("a"), hit("b")]})
    report = EvalRunner(settings(top_k=1, collection="docs"), search, CustomEvaluator()).run(path)
    assert report.hit_rate == 0 and report.results[0]["retrieved_ids"] == ["a"]
    assert search.calls[0][1]["filters"] == {"collection": "docs"}


class SemanticEvaluator(BaseEvaluator):
    def evaluate(self, *args, **kwargs):
        raise AssertionError("semantic dispatch required")
    def evaluate_sample(self, query, retrieved_ids, golden_ids, trace=None, **sample):
        assert sample == {"answer": "candidate answer", "reference": "reference answer", "contexts": ["text-a"]}
        return {"faithfulness": .8}


def test_semantic_payload_uses_actual_retrieved_context_with_composite(tmp_path):
    path = write_cases(tmp_path, [{"query": "q", "expected_chunk_ids": ["a"],
                                  "answer": "candidate answer", "reference": "reference answer"}])
    evaluator = CompositeEvaluator([SemanticEvaluator(), CustomEvaluator()])
    report = EvalRunner(settings(), Search({"q": [hit("a")]}), evaluator).run(path)
    assert report.metrics == {"faithfulness": .8, "hit_rate": 1., "mrr": 1.}


@pytest.mark.parametrize("cases", [[], [None], [{}], [{"query": " "}],
    [{"query": "q", "expected_chunk_ids": "id"}], [{"query": "q", "expected_sources": [""]}],
    [{"query": "q", "expected_chunk_ids": ["a", "a"]}],
    [{"query": "q", "expected_chunk_ids": ["a"], "answer": ""}],
])
def test_invalid_cases_fail_before_retrieval(tmp_path, cases):
    search = Search({})
    with pytest.raises(ValueError):
        EvalRunner(settings(), search, CustomEvaluator()).run(write_cases(tmp_path, cases))
    assert search.calls == []


def test_whole_set_is_validated_before_first_query(tmp_path):
    path = write_cases(tmp_path, [{"query": "q", "expected_chunk_ids": ["a"]}, {"query": "invalid"}])
    search = Search({"q": [hit("a")]})
    with pytest.raises(ValueError):
        EvalRunner(settings(), search, CustomEvaluator()).run(path)
    assert search.calls == []


@pytest.mark.parametrize("top_k", [0, -1, True, "2"])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError):
        EvalRunner(settings(top_k=top_k), Search({}), CustomEvaluator())


def test_backend_failure_is_not_returned_as_successful_report(tmp_path):
    class Broken(BaseEvaluator):
        def evaluate(self, *args, **kwargs):
            raise TimeoutError("judge unavailable")
    with pytest.raises(TimeoutError):
        EvalRunner(settings(), Search({"q": []}), Broken()).run(
            write_cases(tmp_path, [{"query": "q", "expected_chunk_ids": ["a"]}]))


def test_fixture_labels_reference_real_sample_documents():
    root = Path(__file__).resolve().parents[1]
    cases = EvalRunner.load_test_set(root / "fixtures/golden_test_set.json")
    assert len(cases) >= 2
    assert all((root / "fixtures/sample_documents" / source).is_file()
               for case in cases for source in case["expected_sources"])
