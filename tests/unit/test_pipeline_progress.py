"""Progress observers follow committed stages and cannot change ingestion state."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from core import ChunkRecord, Document
from core.settings import Settings
from ingestion import IngestionPipeline, IngestionPipelineError
from ingestion.chunking import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessResult, BatchProcessor
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage import BM25Indexer, VectorUpserter
from libs.loader import BaseLoader, SQLiteIntegrityChecker
from observability.logger import get_logger


pytestmark = pytest.mark.unit
PHASES = ["integrity", "load", "split", "transform", "embed", "upsert"]


class Loader(BaseLoader):
    def __init__(self):
        self.calls = 0
        self.fail = False

    def load(self, path):
        self.calls += 1
        if self.fail:
            raise ValueError("loader failed")
        return Document("doc", "Alpha section.\n\nBeta section.\n\nGamma section.",
                        {"source_path": str(path), "images": []})


class Encoder(BatchProcessor):
    def __init__(self):
        pass

    def process(self, chunks, trace=None):
        return BatchProcessResult(tuple(chunks), tuple((1.0, 0.0) for _ in chunks),
                                  SparseEncoder().encode(chunks).records, len(chunks))


class Upserter(VectorUpserter):
    def __init__(self):
        self.calls = 0

    def upsert(self, chunks, dense_vectors, trace=None):
        self.calls += 1
        return [ChunkRecord(chunk.id, chunk.text, chunk.metadata, list(vector))
                for chunk, vector in zip(chunks, dense_vectors)]


@pytest.fixture
def pipeline_data(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"fixture source for integrity checking")
    settings = Settings(
        splitter={"provider": "recursive", "chunk_size": 20, "chunk_overlap": 0},
        embedding={"provider": "fake"}, vector_store={"provider": "fake"},
        observability={"trace_enabled": False, "trace_file": str(tmp_path / "traces.jsonl")},
    )
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    loader, upserter = Loader(), Upserter()
    pipeline = IngestionPipeline(
        settings, integrity_checker=checker, loader=loader, chunker=DocumentChunker(settings),
        transforms=[], batch_processor=Encoder(), vector_upserter=upserter,
        bm25_indexer=BM25Indexer(tmp_path / "bm25"),
    )
    return pipeline, source, checker, loader, upserter


@pytest.mark.parametrize("trace_enabled", [False, True])
def test_reports_monotonic_completed_stages_with_or_without_tracing(pipeline_data, trace_enabled):
    pipeline, source, checker, loader, upserter = pipeline_data
    pipeline.settings.observability["trace_enabled"] = trace_enabled
    observer = Mock()
    result = pipeline.run(source, on_progress=observer)
    assert result.status == "success" and result.batch_count > 1
    assert [call.args for call in observer.call_args_list] == [
        (stage, index, 6) for index, stage in enumerate(PHASES, start=1)
    ]
    assert checker.should_skip(result.file_hash)
    assert upserter.calls == 1


def test_skipped_file_reports_complete_without_reloading(pipeline_data):
    pipeline, source, checker, loader, upserter = pipeline_data
    pipeline.run(source)
    observer = Mock()
    result = pipeline.run(source, on_progress=observer)
    assert result.skipped
    observer.assert_called_once_with("integrity", 6, 6)
    assert loader.calls == upserter.calls == 1


def test_failed_stage_does_not_report_completion(pipeline_data):
    pipeline, source, checker, loader, upserter = pipeline_data
    loader.fail = True
    observer = Mock()
    with pytest.raises(IngestionPipelineError) as caught:
        pipeline.run(source, on_progress=observer)
    assert caught.value.stage == "load"
    observer.assert_called_once_with("integrity", 1, 6)
    assert upserter.calls == 0
    assert not checker.should_skip(checker.compute_sha256(source))


def test_callback_failure_after_commit_cannot_mark_ingestion_failed(pipeline_data, capsys):
    pipeline, source, checker, loader, upserter = pipeline_data
    pipeline.logger = get_logger("ingestion.pipeline")
    events = []

    def observer(stage, current, total):
        events.append((stage, current, total))
        if stage == "upsert":
            assert checker.should_skip(checker.compute_sha256(source))
            raise RuntimeError("private observer error")

    result = pipeline.run(source, on_progress=observer)
    assert result.status == "success" and checker.should_skip(result.file_hash)
    assert events[-1] == ("upsert", 6, 6)
    stderr = capsys.readouterr().err
    assert "progress callback failed: RuntimeError" in stderr
    assert "private observer error" not in stderr


def test_none_callback_preserves_results_and_force_restarts_progress(pipeline_data):
    pipeline, source, checker, loader, upserter = pipeline_data
    baseline = pipeline.run(source, on_progress=None)
    observer = Mock()
    forced = pipeline.run(source, force=True, on_progress=observer)
    assert forced == baseline
    assert observer.call_args_list[0].args == ("integrity", 1, 6)
    assert observer.call_args_list[-1].args == ("upsert", 6, 6)


def test_non_callable_observer_fails_before_processing(pipeline_data):
    pipeline, source, checker, loader, upserter = pipeline_data
    pipeline.settings.observability["trace_enabled"] = True
    with pytest.raises(TypeError, match="on_progress"):
        pipeline.run(source, on_progress=42)
    assert loader.calls == 0
    assert not Path(pipeline.settings.observability["trace_file"]).exists()
