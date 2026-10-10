"""D6: backend routing, bounded fallback, and caller-state preservation."""

from collections.abc import Sequence
from copy import deepcopy
from threading import Event

import pytest

from core.query_engine import Reranker
from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker, LLMRerankerError
from libs.reranker.reranker_factory import RerankerFactory


pytestmark = pytest.mark.unit


@pytest.fixture
def candidates() -> list[RetrievalResult]:
    return [
        RetrievalResult(key, score, f"text-{key}", {"source_path": f"{key}.pdf"})
        for key, score in (("a", 0.03), ("b", 0.02), ("c", 0.01))
    ]


def settings(**fields: object) -> Settings:
    return Settings(rerank={"enabled": True, "provider": "fake", **fields})


class FakeBackend(BaseReranker):
    def __init__(self, action=None):
        self.action = action
        self.calls = 0

    def rerank(self, query, candidates, trace=None):
        self.calls += 1
        if self.action:
            return self.action(query, candidates, trace)
        return list(reversed(candidates))


def test_success_reranks_before_top_k_and_preserves_payloads(candidates):
    backend = FakeBackend()
    trace = TraceContext()
    output = Reranker(settings(), backend).rerank("query", candidates, top_k=2, trace=trace)
    assert output == [candidates[2], candidates[1]]
    assert output[0] is candidates[2]
    assert trace.stages[-1]["fallback"] is False
    assert trace.stages[-1]["output_count"] == 2


@pytest.mark.parametrize("error", [RuntimeError, TimeoutError, LLMRerankerError])
def test_backend_exception_keeps_fusion_order(candidates, error):
    def fail(query, items, trace):
        raise error("private backend details")

    trace = TraceContext()
    output = Reranker(settings(), FakeBackend(fail)).rerank(
        "query", candidates, top_k=2, trace=trace
    )
    assert output == candidates[:2]
    assert output[0] is candidates[0]
    assert trace.stages[-1]["fallback"] is True
    assert trace.stages[-1]["error_type"] == error.__name__
    assert "private backend details" not in str(trace.stages)


def test_factory_is_lazy_and_initialization_failure_falls_back(candidates, monkeypatch):
    def fail(settings):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(RerankerFactory, "create", fail)
    reranker = Reranker(settings())
    trace = TraceContext()
    assert reranker.rerank("query", candidates, trace=trace) == candidates
    assert trace.stages[-1]["fallback"] is True


def test_registered_factory_backend_is_reused(candidates, monkeypatch):
    backend = FakeBackend()
    builds = []

    def build(config):
        builds.append(config)
        return backend

    monkeypatch.setitem(RerankerFactory._providers, "fake", build)
    reranker = Reranker(settings())
    for _ in range(2):
        assert reranker.rerank("query", candidates) == list(reversed(candidates))
    assert len(builds) == 1
    assert backend.calls == 2


@pytest.mark.parametrize("config", [settings(enabled=False), settings(provider="none")])
def test_disabled_does_not_initialize_backend(candidates, config, monkeypatch):
    def unexpected(settings):
        pytest.fail("disabled reranking must not build a backend")

    monkeypatch.setattr(RerankerFactory, "create", unexpected)
    trace = TraceContext()
    assert Reranker(config).rerank("query", candidates, trace=trace) == candidates
    assert trace.stages[-1]["fallback"] is False
    assert trace.stages[-1]["reason"] == "disabled"


def test_empty_candidates_do_not_call_backend():
    backend = FakeBackend()
    trace = TraceContext()
    assert Reranker(settings(), backend).rerank("query", [], trace=trace) == []
    assert backend.calls == 0
    assert trace.stages[-1]["reason"] == "empty_candidates"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "replacement", "bad_container"])
def test_invalid_backend_output_falls_back(candidates, mode):
    def invalid(query, items, trace):
        return {
            "missing": items[:1],
            "duplicate": [items[0]] * len(items),
            "replacement": deepcopy(items),
            "bad_container": None,
        }[mode]

    trace = TraceContext()
    assert Reranker(settings(), FakeBackend(invalid)).rerank(
        "query", candidates, trace=trace
    ) == candidates
    assert trace.stages[-1]["fallback"] is True


def test_in_place_backend_cannot_mutate_caller_payload(candidates):
    snapshot = deepcopy(candidates)

    def mutate(query, items, trace):
        items.reverse()
        items[0].metadata["source_path"] = "changed.pdf"
        items[0].score = 999
        return items

    output = Reranker(settings(), FakeBackend(mutate)).rerank("query", candidates)
    assert output == list(reversed(candidates))
    assert candidates == snapshot


def test_timeout_returns_before_worker_and_isolates_late_writes(candidates):
    entered, release, finished = Event(), Event(), Event()

    def blocked(query, items, trace):
        entered.set()
        try:
            release.wait(3)
            items[0].text = "late write"
            trace.record_stage("late_backend")
            return list(reversed(items))
        finally:
            finished.set()

    backend = FakeBackend(blocked)
    reranker = Reranker(settings(timeout_seconds=0.05), backend)
    trace = TraceContext()
    snapshot = deepcopy(candidates)
    try:
        assert reranker.rerank("query", candidates, trace=trace) == candidates
        assert entered.is_set() and not finished.is_set()
        assert trace.stages[-1]["fallback"] is True
        assert trace.stages[-1]["reason"] == "timeout"
        assert reranker.rerank("query", candidates, trace=trace) == candidates
        assert trace.stages[-1]["reason"] == "backend_busy"
        assert backend.calls == 1
    finally:
        release.set()
        assert finished.wait(2)
    assert candidates == snapshot
    assert all(stage["stage"] == "rerank" for stage in trace.stages)


def test_successful_backend_trace_is_merged(candidates):
    def traced(query, items, trace):
        assert query == "query"
        trace.record_stage("backend", count=len(items))
        return list(items)

    trace = TraceContext()
    trace.record_stage("fusion")
    Reranker(settings(), FakeBackend(traced)).rerank(" query ", candidates, trace=trace)
    assert [stage["stage"] for stage in trace.stages] == ["fusion", "backend", "rerank"]


@pytest.mark.parametrize("value", [0, -1, True, "1", float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(value):
    with pytest.raises(ValueError, match="timeout_seconds"):
        Reranker(settings(timeout_seconds=value))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected_before_backend(candidates, value):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="top_k"):
        Reranker(settings(), backend).rerank("query", candidates, top_k=value)
    assert backend.calls == 0


@pytest.mark.parametrize("query,items,error", [
    ("", [], ValueError), (None, [], TypeError), ("query", "bad", TypeError),
    ("query", [object()], TypeError),
])
def test_invalid_inputs_raise(query, items, error):
    with pytest.raises(error):
        Reranker(settings(), FakeBackend()).rerank(query, items)


def test_duplicate_input_ids_are_rejected(candidates):
    with pytest.raises(ValueError, match="unique"):
        Reranker(settings(), FakeBackend()).rerank("query", [candidates[0]] * 2)


def test_real_cross_encoder_adapter_with_offline_scorer(candidates):
    class Scorer:
        def predict(self, sentences: Sequence, **kwargs):
            return [0.1, 0.9, 0.3]

    backend = CrossEncoderReranker({}, scorer=Scorer())
    result = Reranker(settings(provider="cross_encoder"), backend).rerank("query", candidates)
    assert result == [candidates[1], candidates[2], candidates[0]]


def test_real_llm_adapter_with_offline_model(candidates):
    from libs.llm.base_llm import BaseLLM

    class Model(BaseLLM):
        def chat(self, messages, trace=None):
            return '{"ranked_ids": ["c", "a", "b"]}'

    backend = LLMReranker({}, llm=Model())
    result = Reranker(settings(provider="llm"), backend).rerank("query", candidates)
    assert result == [candidates[2], candidates[0], candidates[1]]
