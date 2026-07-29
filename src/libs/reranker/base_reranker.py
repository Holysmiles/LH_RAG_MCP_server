"""Provider-independent contract for ranking retrieved candidates.

The contract preserves the candidate object type so the Libs layer does not
need to depend on the concrete ``RetrievalResult`` model introduced later in
the Core layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


CandidateT = TypeVar("CandidateT")


class BaseReranker(ABC):
    """Define the minimal contract implemented by reranking backends."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[CandidateT],
        trace: TraceContext | None = None,
    ) -> list[CandidateT]:
        """Return candidates ordered by relevance to the query.

        Args:
            query: User query used to judge candidate relevance.
            candidates: Retrieved candidate objects in their current order.
            trace: Optional request trace propagated by the query pipeline.

        Returns:
            Ranked candidates containing the original candidate object type.
        """
        raise NotImplementedError
