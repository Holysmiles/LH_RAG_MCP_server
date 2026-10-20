"""Provider-independent contract for RAG evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEvaluator(ABC):
    """Define the minimal contract implemented by evaluation backends.
    规定评估器必须实现 evaluate(...) 方法
    用来统一不同评估实现的输入/输出格式
    
    """

    def evaluate_sample(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: TraceContext | None = None,
        *,
        answer: str | None = None,
        contexts: Sequence[str] | None = None,
        reference: str | None = None,
    ) -> dict[str, float]:
        """Delegate to ID-only backends, preserving IDs and trace unchanged.

        Semantic evaluators override this method to consume answer, contexts,
        and reference. Existing retrieval evaluators may ignore those fields.
        """
        return self.evaluate(query, retrieved_ids, golden_ids, trace=trace)

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Compare retrieved IDs with expected IDs and return named metrics.

        Args:
            query: Query whose retrieval result is being evaluated.
            retrieved_ids: Retrieved item IDs ordered from best to worst.
            golden_ids: IDs considered relevant for the query.
            trace: Optional trace propagated by the evaluation caller.

        Returns:
            A mapping from metric names to numeric scores.
        """
        raise NotImplementedError
