"""Provider-independent contracts for vector storage backends.

The storage boundary accepts serializable record mappings so the concrete
``ChunkRecord`` business type can remain in ``core`` when it is introduced.
Query results use a small stable shape required by the retrieval layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


VectorRecord = Mapping[str, Any]


class VectorSearchResult(TypedDict):
    """Stable result shape returned by vector store queries."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """Define the minimal contract implemented by vector store backends."""

    @abstractmethod
    def upsert(
        self,
        records: Sequence[VectorRecord],
        trace: TraceContext | None = None,
    ) -> None:
        """Insert or replace vector records using their stable identifiers.

        Each record is expected to contain ``id``, ``text``, ``metadata``, and
        a dense vector field understood by the concrete backend adapter.

        Args:
            records: Serializable records to write as one batch.
            trace: Optional request trace propagated by the calling pipeline.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[VectorSearchResult]:
        """Return the most relevant records for a dense query vector.

        Args:
            vector: Dense query vector produced by the embedding component.
            top_k: Maximum number of results to return.
            filters: Optional metadata equality constraints.
            trace: Optional request trace propagated by the calling pipeline.

        Returns:
            Ranked results containing chunk identity, score, text, and metadata.
        """
        raise NotImplementedError
