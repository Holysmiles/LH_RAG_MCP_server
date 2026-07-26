"""Provider-independent contract for splitting text into retrieval units.

The abstraction keeps document chunking independent from a particular
algorithm or third-party library while exposing only the operation required by
the ingestion pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """Define the minimal contract implemented by text splitters."""

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> list[str]:
        """Split text into ordered, provider-independent string chunks.

        Args:
            text: Source text to divide into retrieval units.
            trace: Optional request trace propagated by the calling pipeline.

        Returns:
            Text chunks in the same order as the source content.
        """
        raise NotImplementedError
