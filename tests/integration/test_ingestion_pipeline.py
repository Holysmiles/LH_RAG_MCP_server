"""C14 摄取编排的真实存储集成测试。

测试使用真实 PDF Loader、Splitter、Chroma、BM25、ImageStorage 和 SQLite，只把
Embedding 换成确定性 Fake，确保测试离线且不依赖 API Key。
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3

import pytest

from core import Chunk, Document
from core.settings import Settings
from ingestion import IngestionPipeline, IngestionPipelineError
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from ingestion.transform import BaseTransform
from libs.embedding.base_embedding import BaseEmbedding
from libs.loader import BaseLoader, PdfLoader, SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "sample_documents"
IMAGE_PDF = FIXTURE_DIR / "with_images.pdf"
SIMPLE_PDF = FIXTURE_DIR / "simple.pdf"


class FakeEmbedding(BaseEmbedding):
    """返回同维非零向量，并记录调用以验证增量跳过。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.traces = []

    def embed(
        self,
        texts: list[str],
        trace: object | None = None,
    ) -> list[list[float]]:
        self.traces.append(trace)
        self.calls.append(list(texts))
        return [
            [float(len(text.encode("utf-8")) + 1), float(index + 1)]
            for index, text in enumerate(texts)
        ]


class FailingLoader(BaseLoader):
    def load(self, path: str | Path) -> Document:
        del path
        raise ValueError("corrupted PDF stream")


class StaticLoader(BaseLoader):
    def load(self, path: str | Path) -> Document:
        return Document(
            id="static-document",
            text="first section\n\nsecond section",
            metadata={"source_path": str(path), "images": []},
        )


class IdentityBreakingTransform(BaseTransform):
    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: object | None = None,
    ) -> list[Chunk]:
        del trace
        return [
            Chunk(
                id=f"changed-{chunk.id}",
                text=chunk.text,
                metadata=chunk.metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            )
            for chunk in chunks
        ]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm={"provider": "openai", "model": "unused"},
        vision_llm={"enabled": False},
        embedding={"provider": "openai", "model": "fake"},
        splitter={
            "provider": "recursive",
            "chunk_size": 300,
            "chunk_overlap": 30,
        },
        vector_store={
            "provider": "chroma",
            "persist_directory": str(tmp_path / "data" / "db" / "chroma"),
            "collection_name": "pipeline_test",
        },
        observability={"log_level": "INFO", "trace_file": str(tmp_path / "logs/traces.jsonl")},
        ingestion={"batch_size": 2},
    )


@pytest.mark.integration
def test_real_pdf_pipeline_persists_aligned_dense_sparse_and_image_data(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    checker = SQLiteIntegrityChecker(tmp_path / "data" / "db" / "history.db")
    vector_store = ChromaStore(settings.vector_store)
    bm25 = BM25Indexer(tmp_path / "data" / "db" / "bm25")
    images = ImageStorage(
        tmp_path / "data" / "images",
        tmp_path / "data" / "db" / "image_index.db",
    )
    fake_embedding = FakeEmbedding()
    batch_processor = BatchProcessor(
        settings,
        dense_encoder=DenseEncoder(settings, embedding=fake_embedding),
        sparse_encoder=SparseEncoder(),
    )
    pipeline = IngestionPipeline(
        settings,
        integrity_checker=checker,
        loader=PdfLoader(image_root=tmp_path / "extracted_images"),
        chunker=DocumentChunker(settings),
        batch_processor=batch_processor,
        vector_upserter=VectorUpserter(settings, store=vector_store),
        bm25_indexer=bm25,
        image_storage=images,
    )

    progress = []
    result = pipeline.run(IMAGE_PDF, collection="technical_docs",
                          on_progress=lambda *event: progress.append(event))
    assert progress == [(name, index, 6) for index, name in enumerate(
        ["integrity", "load", "split", "transform", "embed", "upsert"], start=1)]

    trace_path = Path(settings.observability["trace_file"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["trace_type"] == "ingestion" and trace["finished_at"] is not None
    assert trace["total_elapsed_ms"] >= 0
    stages = {item["stage"]: item for item in trace["stages"]}
    phases = ["load", "split", "transform", "embed", "upsert"]
    assert [item["stage"] for item in trace["stages"] if item["stage"] in phases] == phases
    for name in phases:
        assert stages[name]["elapsed_ms"] >= 0 and stages[name]["method"]
        assert stages[name]["details"]["status"] == "completed"
        assert stages[name]["details"]["collection"] == "technical_docs"
    assert stages["load"]["method"] == "markitdown"
    assert stages["split"]["method"] == "recursive"
    assert stages["upsert"]["method"] == "chroma"
    assert stages["upsert"]["details"]["record_count"] == result.chunk_count
    assert {"chunk_refiner", "dense_encoder", "batch_processor", "vector_upserter"} <= stages.keys()
    assert all(context.trace_id == trace["trace_id"] for context in fake_embedding.traces)

    assert result.status == "success"
    assert result.skipped is False
    assert result.chunk_count > 0
    assert result.image_count == 1
    assert result.batch_count >= 1
    assert len(result.record_ids) == result.chunk_count
    assert len(set(result.record_ids)) == result.chunk_count

    dense_results = vector_store.query(
        [1.0, 0.0],
        top_k=result.chunk_count,
        filters={"collection": "technical_docs"},
    )
    assert {item["chunk_id"] for item in dense_results} == set(result.record_ids)
    assert all(
        item["metadata"]["file_hash"] == result.file_hash
        for item in dense_results
    )

    index_payload = json.loads(bm25.index_path.read_text(encoding="utf-8"))
    # 两路索引共享 ID，后续 RRF 才能识别它们是同一条 Chunk。
    assert set(index_payload["records"]) == set(result.record_ids)
    assert bm25.document_count == result.chunk_count

    stored_images = images.list_images(
        collection="technical_docs",
        doc_hash=result.document_id,
    )
    assert len(stored_images) == 1
    assert stored_images[0]["exists"] is True
    managed_path = Path(stored_images[0]["file_path"])
    assert managed_path.parent.name == "technical_docs"
    assert managed_path.stem.startswith(result.file_hash)

    embedding_call_count = len(fake_embedding.calls)
    skipped = pipeline.run(IMAGE_PDF, collection="technical_docs")
    assert skipped.status == "skipped"
    assert skipped.file_hash == result.file_hash
    assert len(fake_embedding.calls) == embedding_call_count
    assert bm25.document_count == result.chunk_count
    saved = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 2
    assert saved[-1]["trace_id"] != saved[0]["trace_id"]
    assert saved[-1]["stages"][0]["details"]["status"] == "skipped"
    assert [item["stage"] for item in saved[-1]["stages"]] == ["integrity"]


@pytest.mark.integration
def test_failed_stage_is_reported_and_integrity_status_remains_retryable(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    pipeline = IngestionPipeline(
        settings,
        integrity_checker=checker,
        loader=FailingLoader(),
    )

    with pytest.raises(IngestionPipelineError, match="stage 'load'.*corrupted") as exc:
        pipeline.run(SIMPLE_PDF, collection="failures")

    file_hash = checker.compute_sha256(SIMPLE_PDF)
    assert exc.value.stage == "load"
    assert checker.should_skip(file_hash) is False
    with sqlite3.connect(checker.db_path) as connection:
        status, error = connection.execute(
            "SELECT status, error_msg FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
    assert status == "failed"
    assert "corrupted PDF stream" in error
    assert not (tmp_path / "data" / "db" / "chroma").exists()
    saved = json.loads(Path(settings.observability["trace_file"]).read_text(encoding="utf-8"))
    assert saved["trace_type"] == "ingestion" and saved["finished_at"] is not None
    assert saved["stages"][-1]["stage"] == "load"
    assert saved["stages"][-1]["error_type"] == "ValueError"
    assert saved["stages"][-1]["details"]["status"] == "failed"
    assert "corrupted PDF stream" not in json.dumps(saved)


@pytest.mark.integration
def test_transform_cannot_silently_break_chunk_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"content used only by the integrity checker")
    settings = make_settings(tmp_path)
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    pipeline = IngestionPipeline(
        settings,
        integrity_checker=checker,
        loader=StaticLoader(),
        chunker=DocumentChunker(settings),
        transforms=[IdentityBreakingTransform()],
    )

    with pytest.raises(IngestionPipelineError, match="changed chunk identity") as exc:
        pipeline.run(source)

    assert exc.value.stage == "transform.IdentityBreakingTransform"
    assert checker.should_skip(checker.compute_sha256(source)) is False
    saved = json.loads(Path(settings.observability["trace_file"]).read_text(encoding="utf-8"))
    assert saved["stages"][-1]["stage"] == "transform"
    assert saved["stages"][-1]["details"]["failed_step"] == "transform.IdentityBreakingTransform"


@pytest.mark.integration
def test_disabled_ingestion_trace_creates_no_trace_file(tmp_path):
    settings = make_settings(tmp_path)
    settings.observability["trace_enabled"] = False
    pipeline = IngestionPipeline(settings, loader=FailingLoader(),
                                 integrity_checker=SQLiteIntegrityChecker(tmp_path / "history.db"))
    with pytest.raises(IngestionPipelineError):
        pipeline.run(SIMPLE_PDF)
    assert not Path(settings.observability["trace_file"]).exists()


@pytest.mark.integration
def test_trace_write_failure_does_not_mask_ingestion_failure(tmp_path, capsys):
    settings = make_settings(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    settings.observability["trace_file"] = str(blocked / "traces.jsonl")
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    pipeline = IngestionPipeline(settings, loader=FailingLoader(), integrity_checker=checker)
    with pytest.raises(IngestionPipelineError) as caught:
        pipeline.run(SIMPLE_PDF)
    assert caught.value.stage == "load"
    assert checker.should_skip(checker.compute_sha256(SIMPLE_PDF)) is False
    assert "Cannot persist ingestion trace" in capsys.readouterr().err
