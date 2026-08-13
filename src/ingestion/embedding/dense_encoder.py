"""Batch dense-vector encoding for ingestion.

``Chunk[] -> text batch -> BaseEmbedding -> vector[]``

This module deliberately stops before persistence.  Its only job is to bridge
the core ``Chunk`` contract and the provider-independent ``BaseEmbedding``
contract while checking provider output early.  The caller receives vectors in
exactly the same order as the input chunks; metadata and Chunk identity remain
owned by the later ``ChunkRecord``/upsert stages.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from core import Chunk
from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class DenseEncoder:
    """Encode chunks in one provider batch and validate the resulting matrix.

    Inputs:
        A ``Settings`` object for provider construction, an optional injected
        ``BaseEmbedding`` for tests, and a sequence of ``Chunk`` objects.
    Outputs:
        ``list[list[float]]`` with one non-empty vector per input Chunk.  Every
        row has the same dimension and keeps input order.
    Failure boundaries:
        Invalid caller types and malformed provider output raise immediately;
        provider/network failures are propagated with their original cause
        because dense vectors are required for the requested ingestion path.
        An empty batch returns ``[]`` without constructing or calling a model.
    """

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if embedding is not None and not isinstance(embedding, BaseEmbedding):
            raise TypeError("embedding must implement BaseEmbedding or be None")
        self.settings = settings
        self.embedding = embedding if embedding is not None else EmbeddingFactory.create(settings)

    def encode(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """Encode one batch, preserving order and rejecting unsafe responses."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
        if not chunks:
            if trace is not None:
                trace.record_stage(
                    "dense_encoder",
                    {"input_count": 0, "output_count": 0, "dimension": 0},
                )
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding.embed(texts, trace=trace)
        validated = self._validate_vectors(vectors, expected_count=len(chunks))
        if trace is not None:
            trace.record_stage(
                "dense_encoder",
                {
                    "input_count": len(chunks),
                    "output_count": len(validated),
                    "dimension": len(validated[0]),
                },
            )
        return validated

    @staticmethod
    def _validate_vectors(
        vectors: object,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
            raise TypeError("embedding provider must return a sequence of vectors")
        if len(vectors) != expected_count:
            raise ValueError(
                "embedding provider returned "
                f"{len(vectors)} vectors for {expected_count} chunks"
            )

        validated: list[list[float]] = []
        dimension: int | None = None
        for row_index, vector in enumerate(vectors):
            if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                raise TypeError(f"embedding vector[{row_index}] must be a sequence")
            if not vector:
                raise ValueError(f"embedding vector[{row_index}] must not be empty")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(
                    "embedding provider returned inconsistent vector dimensions: "
                    f"vector[0]={dimension}, vector[{row_index}]={len(vector)}"
                )
            normalized: list[float] = []
            for value_index, value in enumerate(vector):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"embedding vector[{row_index}][{value_index}] must be numeric"
                    )
                number = float(value)
                if not isfinite(number):
                    raise ValueError(
                        f"embedding vector[{row_index}][{value_index}] must be finite"
                    )
                normalized.append(number)
            validated.append(normalized)
        return validated


__all__ = ["DenseEncoder"]
