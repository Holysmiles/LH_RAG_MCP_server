"""Hybrid retrieval cooperation and CLI stage-record regressions."""

from threading import Barrier

import pytest

from core.query_engine import Fusion, HybridSearch, QueryProcessor
from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult

pytestmark = pytest.mark.integration


def item(key, collection="alpha"):
    return RetrievalResult(key, 0.5, f"text-{key}", {
        "source_path": f"{key}.pdf", "collection": collection,
    })


class Route:
    def __init__(self, results=None, error=None, barrier=None):
        self.results = results or []
        self.error = error
        self.barrier = barrier
        self.calls = []

    def retrieve(self, *args):
        self.calls.append(args)
        if self.barrier:
            self.barrier.wait(timeout=2)
        if self.error:
            raise self.error("private backend data")
        return self.results


def searcher(dense, sparse):
    settings = Settings(retrieval={"dense_top_k": 5, "sparse_top_k": 8, "fusion_top_k": 2})
    return HybridSearch(settings, QueryProcessor(), dense, sparse, Fusion(settings))


def test_parallel_routes_preserve_payload_and_record_same_run():
    barrier = Barrier(2)
    dense = Route([item("a"), item("b")], barrier=barrier)
    sparse = Route([item("b"), item("c")], barrier=barrier)
    trace = TraceContext()
    results = searcher(dense, sparse).search("Azure", top_k=2, trace=trace)
    assert [entry.chunk_id for entry in results] == ["b", "a"]
    assert results[0].text == "text-b"
    assert results[0].metadata["source_path"] == "b.pdf"
    assert [stage["stage"] for stage in trace.stages] == ["dense_retrieval", "sparse_retrieval", "fusion"]
    assert trace.stages[0]["results"][0]["chunk_id"] == "a"
    assert trace.stages[2]["results"] == [entry.to_dict() for entry in results]
    assert dense.calls[0][1] == 5
    assert sparse.calls[0][1] == 8


def test_filtering_precedes_top_k_truncation():
    route = [item("other", "beta"), item("target")]
    dense, sparse = Route(route), Route(route)
    results = searcher(dense, sparse).search("Azure", top_k=1, filters={"collection": "alpha"})
    assert [entry.chunk_id for entry in results] == ["target"]
    assert dense.calls[0][2] == {"collection": "alpha"}


@pytest.mark.parametrize("failed_route", ["dense", "sparse"])
def test_single_route_failure_retains_available_results(failed_route):
    routes = {"dense": Route([item("a")]), "sparse": Route([item("a")])}
    routes[failed_route] = Route(error=RuntimeError)
    trace = TraceContext()
    results = searcher(routes["dense"], routes["sparse"]).search("Azure", trace=trace)
    assert [entry.chunk_id for entry in results] == ["a"]
    stage = next(stage for stage in trace.stages if stage["stage"] == failed_route + "_retrieval")
    assert stage["fallback"] is True
    assert stage["error_type"] == "RuntimeError"
    assert "private backend data" not in str(trace.stages)


def test_both_route_failures_are_not_reported_as_no_documents():
    with pytest.raises(RuntimeError, match="Both retrieval routes failed"):
        searcher(Route(error=RuntimeError), Route(error=ValueError)).search("Azure")


def test_empty_index_is_a_successful_empty_result():
    assert searcher(Route(), Route()).search("Azure") == []


def test_large_top_k_widens_recall():
    dense, sparse = Route(), Route()
    searcher(dense, sparse).search("Azure", top_k=30)
    assert dense.calls[0][1] == sparse.calls[0][1] == 30
