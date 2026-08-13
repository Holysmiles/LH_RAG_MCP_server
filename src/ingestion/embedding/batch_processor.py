"""Batch orchestration for dense and sparse ingestion encoders.

``Chunk[] -> fixed-size batches -> DenseEncoder/SparseEncoder -> aggregate``

The processor owns scheduling and ordering only.  DenseEncoder validates
vectors; SparseEncoder calculates BM25 statistics.  Splitting into batches
limits provider request size and memory use, then results are concatenated in
the same order as the input.  A provider failure is intentionally propagated:
silently dropping one batch would create mismatched vectors and corrupt later
upserts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from core import Chunk
from core.settings import Settings
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseChunkEncoding, SparseEncoder

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass(frozen=True, slots=True)
class BatchProcessResult:
    """Aggregated output of one ingestion encoding pass."""

    chunks: tuple[Chunk, ...]
    dense_vectors: tuple[tuple[float, ...], ...]
    sparse_records: tuple[SparseChunkEncoding, ...]
    batch_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic representation."""
        return {
            "chunk_ids": [chunk.id for chunk in self.chunks],
            "dense_vectors": [list(vector) for vector in self.dense_vectors],
            "sparse_records": [record.to_dict() for record in self.sparse_records],
            "batch_count": self.batch_count,
        }


class BatchProcessor:
    """Split chunks and drive both encoders with stable ordering.

    Inputs:
        ``Settings`` supplies ``ingestion.batch_size``.  Dense and sparse
        encoders may be injected for tests or custom deployments; otherwise
        the dense encoder uses ``EmbeddingFactory`` and sparse encoding uses
        the local deterministic tokenizer.
    Outputs:
        ``BatchProcessResult`` with one dense vector and sparse record per
        input chunk, plus the number of provider batches executed.
    Failure boundaries:
        Empty input returns an empty result without constructing model work.
        Invalid settings, chunks, or batch size raise at the boundary.  Any
        encoder/provider failure propagates so the caller can mark ingestion
        failed instead of persisting partial output.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        self.settings = settings
        self.dense_encoder = dense_encoder or DenseEncoder(settings)
        self.sparse_encoder = sparse_encoder or SparseEncoder()
        configured_size = settings.ingestion.get("batch_size", 100)
        selected_size = configured_size if batch_size is None else batch_size
        if isinstance(selected_size, bool) or not isinstance(selected_size, int):
            raise TypeError("batch_size must be a positive integer")
        if selected_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = selected_size

    def process(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> BatchProcessResult:
        """Encode all chunks batch-by-batch and concatenate ordered results."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
        if not chunks:
            return BatchProcessResult((), (), (), 0)

        all_dense: list[tuple[float, ...]] = []
        all_sparse: list[SparseChunkEncoding] = []
        batch_count = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            started = perf_counter()
            dense_vectors = self.dense_encoder.encode(batch, trace=trace)
            sparse_batch = self.sparse_encoder.encode(batch)
            if len(dense_vectors) != len(batch) or len(sparse_batch) != len(batch):
                raise RuntimeError("encoder output count does not match batch size")
            all_dense.extend(tuple(vector) for vector in dense_vectors)
            all_sparse.extend(sparse_batch.records)
            if trace is not None:
                trace.record_stage(
                    "batch_processor",
                    {
                        "batch_index": batch_count,
                        "input_start": start,
                        "input_count": len(batch),
                        "duration_ms": round((perf_counter() - started) * 1000, 3),
                    },
                )
            batch_count += 1

        return BatchProcessResult(
            chunks=tuple(chunks),
            dense_vectors=tuple(all_dense),
            sparse_records=tuple(all_sparse),
            batch_count=batch_count,
        )

    def run(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> BatchProcessResult:
        """Alias for pipeline code that names processing steps ``run``."""
        return self.process(chunks, trace=trace)


__all__ = ["BatchProcessResult", "BatchProcessor"]
