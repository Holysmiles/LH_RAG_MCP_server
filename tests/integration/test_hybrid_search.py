"""Hybrid retrieval cooperation and CLI stage-record regressions."""

from threading import Barrier
import json

import pytest

from core.query_engine import Fusion, HybridSearch, QueryProcessor
from core.query_engine.reranker import Reranker
from core.trace.trace_collector import query_trace
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
    assert [stage["stage"] for stage in trace.stages] == ["query_processing", "dense_retrieval", "sparse_retrieval", "fusion"]
    assert trace.stages[1]["results"][0]["chunk_id"] == "a"
    assert trace.stages[3]["results"] == [entry.to_dict() for entry in results]
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


def test_query_trace_records_all_stages_and_persists_finished_snapshot(tmp_path):
    path = tmp_path / "traces.jsonl"
    hybrid = searcher(Route([item("a")]), Route([item("a")]))
    hybrid.settings.observability = {"trace_enabled": True, "trace_file": str(path)}
    hybrid.settings.embedding = {"provider": "fake-embedding"}
    hybrid.settings.rerank = {"provider": "none"}
    with query_trace(hybrid.settings) as trace:
        candidates = hybrid.search("Azure", trace=trace)
        Reranker(hybrid.settings).rerank("Azure", candidates, trace=trace)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["trace_type"] == "query" and stored["finished_at"] is not None
    assert stored["total_elapsed_ms"] >= 0
    assert [stage["stage"] for stage in stored["stages"]] == [
        "query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank",
    ]
    for stage in stored["stages"]:
        assert stage["elapsed_ms"] >= 0
        assert stage["method"] and stage["provider"]
        assert isinstance(stage["details"], dict)
    assert stored["stages"][1]["provider"] == "fake-embedding"


def test_failed_retrieval_still_finishes_trace_and_preserves_safe_error_types(tmp_path):
    path = tmp_path / "failed.jsonl"
    hybrid = searcher(Route(error=RuntimeError), Route(error=ValueError))
    hybrid.settings.observability = {"trace_file": str(path)}
    with pytest.raises(RuntimeError, match="Both retrieval routes failed"):
        with query_trace(hybrid.settings) as trace:
            hybrid.search("Azure", trace=trace)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["finished_at"] is not None
    assert [stage["error_type"] for stage in stored["stages"][1:]] == ["RuntimeError", "ValueError"]
    assert "private backend data" not in path.read_text(encoding="utf-8")


def test_trace_write_failure_does_not_replace_results_or_query_error(tmp_path, capsys):
    path = tmp_path / "not-a-directory"
    path.write_text("file", encoding="utf-8")
    settings = Settings(observability={"trace_file": str(path / "traces.jsonl")})
    with query_trace(settings) as trace:
        trace.record_stage("query_processing", method="test", elapsed_ms=1)
    assert trace.finished_at is not None
    with pytest.raises(RuntimeError, match="original query error"):
        with query_trace(settings):
            raise RuntimeError("original query error")
    assert "Cannot persist query trace" in capsys.readouterr().err


@pytest.mark.parametrize("force", [True, False])
def test_disabled_trace_can_keep_cli_verbose_data_without_a_file(tmp_path, force):
    path = tmp_path / "disabled.jsonl"
    settings = Settings(observability={"trace_enabled": False, "trace_file": str(path)})
    with query_trace(settings, force=force) as trace:
        assert (trace is not None) is force
        if trace is not None:
            trace.record_stage("query_processing", elapsed_ms=1)
    assert not path.exists()
