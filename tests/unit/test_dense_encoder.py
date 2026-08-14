"""Contract tests for the ingestion dense encoder."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core import Chunk
from core.settings import Settings
from core.trace import TraceContext
from ingestion.embedding import DenseEncoder
from ingestion.embedding.dense_encoder import EmbeddingFactory
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.vectors or [[float(index), 1.0] for index, _ in enumerate(texts)]


def settings() -> Settings:
    return Settings(embedding={"provider": "openai", "model": "test", "api_key": "test"})


def make_chunk(index: int, text: str | None = None) -> Chunk:
    value = text or f"chunk text {index}"
    return Chunk(
        id=f"chunk-{index}",
        text=value,
        metadata={"source_path": "docs/source.pdf", "chunk_index": index},
        start_offset=index * 10,
        end_offset=index * 10 + len(value),
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_encode_batches_chunk_text_and_preserves_order_and_dimension() -> None:
    embedding = FakeEmbedding()
    chunks = [make_chunk(0, "first"), make_chunk(1, "second")]

    vectors = DenseEncoder(settings(), embedding=embedding).encode(chunks)

    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert embedding.calls == [["first", "second"]]
    assert len(vectors) == len(chunks)
    assert len(vectors[0]) == len(vectors[1])


@pytest.mark.unit
def test_empty_batch_does_not_call_provider_and_records_trace() -> None:
    embedding = FakeEmbedding()
    trace = TraceContext()

    vectors = DenseEncoder(settings(), embedding=embedding).encode([], trace)

    assert vectors == []
    assert embedding.calls == []
    assert trace.stages == [
        {
            "input_count": 0,
            "output_count": 0,
            "dimension": 0,
            "stage": "dense_encoder",
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "vectors,error",
    [
        ([[1.0, 2.0]], "returned 1 vectors for 2 chunks"),
        ([[1.0, 2.0], [3.0]], "inconsistent vector dimensions"),
        ([[], [1.0, 2.0]], "must not be empty"),
        ([[1.0, float("nan")], [3.0, 4.0]], "must be finite"),
        ([[1.0, "bad"], [3.0, 4.0]], "must be numeric"),
    ],
)
def test_provider_output_is_validated_before_return(
    vectors: list[list[object]], error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        DenseEncoder(settings(), embedding=FakeEmbedding(vectors)).encode(
            [make_chunk(0), make_chunk(1)]
        )


@pytest.mark.unit
def test_public_input_and_constructor_validation() -> None:
    with pytest.raises(TypeError, match="settings"):
        DenseEncoder(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="embedding"):
        DenseEncoder(settings(), embedding=object())  # type: ignore[arg-type]
    encoder = DenseEncoder(settings(), embedding=FakeEmbedding())
    with pytest.raises(TypeError, match="chunks must be"):
        encoder.encode("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"chunks\[0\]"):
        encoder.encode([object()])  # type: ignore[list-item]


@pytest.mark.unit
def test_provider_is_created_lazily_only_for_the_first_non_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = FakeEmbedding()
    create_calls: list[Settings] = []

    def create(settings_value: Settings) -> BaseEmbedding:
        create_calls.append(settings_value)
        return embedding

    monkeypatch.setattr(EmbeddingFactory, "create", create)
    encoder = DenseEncoder(settings())

    assert create_calls == []
    assert encoder.encode([]) == []
    assert create_calls == []
    encoder.encode([make_chunk(0)])
    encoder.encode([make_chunk(1)])

    assert len(create_calls) == 1
