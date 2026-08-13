"""Contract tests for stable ingestion batch orchestration."""

from __future__ import annotations

import pytest

from core import Chunk
from core.settings import Settings
from core.trace import TraceContext
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def settings(batch_size: int = 100) -> Settings:
    return Settings(
        embedding={"provider": "openai", "model": "test", "api_key": "test"},
        ingestion={"batch_size": batch_size},
    )


def make_chunk(index: int) -> Chunk:
    text = f"chunk {index} endpoint"
    return Chunk(
        id=f"chunk-{index}",
        text=text,
        metadata={"source_path": "docs/source.pdf", "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def processor(batch_size: int = 100) -> tuple[BatchProcessor, FakeEmbedding]:
    settings_value = settings(batch_size)
    fake = FakeEmbedding()
    return (
        BatchProcessor(
            settings_value,
            dense_encoder=DenseEncoder(settings_value, embedding=fake),
            sparse_encoder=SparseEncoder(),
        ),
        fake,
    )


@pytest.mark.unit
def test_batch_size_two_splits_five_chunks_into_three_ordered_batches() -> None:
    batch_processor, embedding = processor(2)
    chunks = [make_chunk(index) for index in range(5)]

    result = batch_processor.process(chunks)

    assert result.batch_count == 3
    assert [chunk.id for chunk in result.chunks] == [f"chunk-{i}" for i in range(5)]
    assert [record.chunk_id for record in result.sparse_records] == [
        f"chunk-{i}" for i in range(5)
    ]
    assert len(result.dense_vectors) == 5
    assert embedding.calls == [
        ["chunk 0 endpoint", "chunk 1 endpoint"],
        ["chunk 2 endpoint", "chunk 3 endpoint"],
        ["chunk 4 endpoint"],
    ]


@pytest.mark.unit
def test_empty_batch_does_not_call_embedding() -> None:
    batch_processor, embedding = processor(2)

    result = batch_processor.run([])

    assert result.batch_count == 0
    assert result.chunks == ()
    assert result.dense_vectors == ()
    assert result.sparse_records == ()
    assert embedding.calls == []


@pytest.mark.unit
def test_trace_contains_one_stage_per_batch() -> None:
    batch_processor, _ = processor(2)
    trace = TraceContext()

    batch_processor.process([make_chunk(i) for i in range(5)], trace)

    batch_stages = [stage for stage in trace.stages if stage["stage"] == "batch_processor"]
    assert len(batch_stages) == 3
    assert [stage["input_count"] for stage in batch_stages] == [2, 2, 1]
    assert [stage["batch_index"] for stage in batch_stages] == [0, 1, 2]


@pytest.mark.unit
def test_invalid_batch_configuration_and_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        processor(0)
    with pytest.raises(TypeError, match="batch_size"):
        BatchProcessor(settings(), batch_size="2")  # type: ignore[arg-type]
    batch_processor, _ = processor(2)
    with pytest.raises(TypeError, match="chunks must be"):
        batch_processor.process("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"chunks\[0\]"):
        batch_processor.process([object()])  # type: ignore[list-item]
