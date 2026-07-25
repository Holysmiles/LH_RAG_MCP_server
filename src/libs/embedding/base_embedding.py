"""Provider-independent contract for dense text embeddings.

The abstraction in this module keeps ingestion and retrieval code independent
from provider SDKs while preserving a small, batch-oriented interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    """Define the minimal contract implemented by embedding providers."""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """Convert a batch of texts into dense vectors in input order.

        Args:
            texts: Text values to encode as one batch.
            trace: Optional request trace propagated by the calling pipeline.

        Returns:
            One floating-point vector for each input text, preserving order.
        """
        raise NotImplementedError
