"""Hybrid dense and sparse retrieval orchestration."""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from core.settings import Settings
from core.types import RetrievalResult
from core.trace.trace_context import TraceContext
from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever


class HybridSearch:
    """Run both retrieval routes, fuse results, and apply defensive filters."""

    def __init__(
        self, settings: Settings, query_processor: QueryProcessor,
        dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever,
        fusion: Fusion,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings")
        self.settings, self.query_processor = settings, query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion

    def search(
        self, query: str, top_k: int | None = None,
        filters: Mapping[str, Any] | None = None, trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        processed = self.query_processor.process(query, filters)
        selected_top_k = self.fusion.top_k if top_k is None else top_k
        # Recall budgets are independent of the final result count. Increasing
        # Top-K may widen recall, but a small Top-K must not discard candidates
        # before collection filtering or fusion.
        dense_k = max(self.settings.retrieval.get("dense_top_k", 20), selected_top_k)
        sparse_k = max(self.settings.retrieval.get("sparse_top_k", 20), selected_top_k)
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(
                self.dense_retriever.retrieve, processed.query, dense_k, processed.filters, trace
            )
            sparse_future = pool.submit(
                self.sparse_retriever.retrieve, processed.keywords, sparse_k, trace
            )
            dense, dense_error = self._safe_result(dense_future)
            sparse, sparse_error = self._safe_result(sparse_future)
        if trace is not None:
            for name, results, error in (
                ("dense_retrieval", dense, dense_error),
                ("sparse_retrieval", sparse, sparse_error),
            ):
                trace.record_stage(
                    name, results=[item.to_dict() for item in results],
                    fallback=error is not None, error_type=error,
                )
        if dense_error is not None and sparse_error is not None:
            raise RuntimeError(
                f"Both retrieval routes failed: dense={dense_error}, sparse={sparse_error}"
            )
        # Filter both routes before fusion's Top-K truncation so high-ranked
        # chunks from other collections cannot crowd out eligible candidates.
        fused = self.fusion.fuse(
            self._apply_metadata_filters(dense, processed.filters),
            self._apply_metadata_filters(sparse, processed.filters),
            top_k=selected_top_k,
        )
        if trace is not None:
            trace.record_stage("fusion", results=[item.to_dict() for item in fused])
        return fused

    @staticmethod
    def _safe_result(
        future: Future[list[RetrievalResult]],
    ) -> tuple[list[RetrievalResult], str | None]:
        try:
            return future.result(), None
        except Exception as exc:
            return [], type(exc).__name__

    @staticmethod
    def _apply_metadata_filters(candidates: list[RetrievalResult], filters: Mapping[str, Any]) -> list[RetrievalResult]:
        if not filters:
            return candidates
        return [item for item in candidates if all(item.metadata.get(key) == value for key, value in filters.items())]


__all__ = ["HybridSearch"]
