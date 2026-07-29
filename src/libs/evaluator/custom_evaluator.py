"""Deterministic retrieval metrics without external dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class CustomEvaluator(BaseEvaluator):
    """Evaluate ranked retrieval results using hit rate and reciprocal rank."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Return hit rate and MRR for one ordered retrieval result.

        ``hit_rate`` is one when at least one relevant ID was retrieved.
        ``mrr`` is the reciprocal rank of the first relevant retrieved ID.
        Both metrics are zero when there is no match.
        """
        relevant_ids = set(golden_ids)
        first_match_rank = next(
            (
                rank
                for rank, retrieved_id in enumerate(retrieved_ids, start=1)
                if retrieved_id in relevant_ids
            ),
            None,
        )

        if first_match_rank is None:
            return {"hit_rate": 0.0, "mrr": 0.0}

        return {"hit_rate": 1.0, "mrr": 1.0 / first_match_rank}
