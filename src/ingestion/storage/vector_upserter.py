"""Convert encoded Chunks into idempotent vector-store records.

``Chunk[] + dense_vectors[] -> stable IDs -> BaseVectorStore.upsert``

The upserter is the boundary between ingestion business objects and a vector
store adapter.  It does not embed text and does not decide retrieval policy.
Its stable ID combines ``source_path``, ``chunk_index`` and a content hash, so
re-ingesting identical content replaces one record while changed content gets
a new ID.  Validation happens before the store call to prevent partial writes
caused by mismatched chunks and vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any

from core import Chunk, ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory
from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class VectorUpserter:
    """Build serializable vector records and write them in one idempotent batch.

    Inputs:
        Settings for the configured VectorStore, ``Chunk`` objects, and one
        dense vector per Chunk in identical order.  A store can be injected to
        keep unit tests offline.
    Outputs:
        The stable ``ChunkRecord`` objects sent to ``BaseVectorStore.upsert``;
        the method returns their records so callers can persist audit data.
    Failure boundaries:
        Invalid input or vector shape raises before any store call.  Store or
        network failures propagate because claiming success after a partial
        write would make the ingestion integrity state incorrect.
    """

    def __init__(
        self,
        settings: Settings,
        store: BaseVectorStore | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if store is not None and not isinstance(store, BaseVectorStore):
            raise TypeError("store must implement BaseVectorStore or be None")
        self.settings = settings
        self.store = store if store is not None else VectorStoreFactory.create(settings)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[Real]],
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        """Validate, build, and upsert one aligned batch of vector records."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")
        if isinstance(dense_vectors, (str, bytes)) or not isinstance(dense_vectors, Sequence):
            raise TypeError("dense_vectors must be a non-string sequence")
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks and dense_vectors must have the same length: {len(chunks)} != {len(dense_vectors)}"
            )
        records: list[ChunkRecord] = []
        seen_ids: set[str] = set()
        vector_size: int | None = None
        for index, (chunk, vector) in enumerate(zip(chunks, dense_vectors, strict=True)):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            normalized_vector = self._validate_vector(vector, index)
            if vector_size is None:
                vector_size = len(normalized_vector)
            elif len(normalized_vector) != vector_size:
                raise ValueError("all dense vectors must have the same dimension")
            chunk_id = self.generate_chunk_id(chunk)
            if chunk_id in seen_ids:
                raise ValueError(f"duplicate generated chunk id: {chunk_id}")
            seen_ids.add(chunk_id)
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    text=chunk.text,
                    metadata=self._record_metadata(chunk, chunk_id),
                    dense_vector=normalized_vector,
                )
            )

        self.store.upsert(  # type: ignore[union-attr]
            [record.to_dict() for record in records], trace=trace
        )
        if trace is not None:
            trace.record_stage(
                "vector_upserter",
                {"input_count": len(chunks), "upsert_count": len(records)},
            )
        return records

    @staticmethod
    def generate_chunk_id(chunk: Chunk) -> str:
        """Generate the deterministic ID shared by repeated ingestion runs."""
        if not isinstance(chunk, Chunk):
            raise TypeError("chunk must be a Chunk")
        source_path = chunk.metadata.get("source_path")
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("chunk.metadata.source_path must be a non-empty string")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("chunk.metadata.chunk_index must be a non-negative integer")
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
        identity = f"{source_path.strip()}|{chunk_index}|{content_hash}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"chunk_{digest}"

    @staticmethod
    def _record_metadata(chunk: Chunk, chunk_id: str) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = chunk_id
        metadata["source_ref"] = chunk.source_ref
        metadata["start_offset"] = chunk.start_offset
        metadata["end_offset"] = chunk.end_offset
        return metadata

    @staticmethod
    def _validate_vector(vector: object, index: int) -> list[float]:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise TypeError(f"dense_vectors[{index}] must be a sequence")
        if not vector:
            raise ValueError(f"dense_vectors[{index}] must not be empty")
        normalized: list[float] = []
        for position, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"dense_vectors[{index}][{position}] must be numeric")
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"dense_vectors[{index}][{position}] must be finite")
            normalized.append(number)
        return normalized


__all__ = ["VectorUpserter"]
