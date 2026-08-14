"""用一次 Query Embedding 驱动向量数据库完成语义召回。

DenseRetriever 是业务查询与两个可插拔 Provider 之间的编排边界：Embedding 只负责
生成向量，VectorStore 只负责近邻查询，本类负责验证二者的接口结果并转换成 Core
层稳定的 ``RetrievalResult``。它不做关键词提取、结果融合或 Rerank。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import TYPE_CHECKING, Any

from core.settings import Settings
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class DenseRetriever:
    """组合 Embedding 与 VectorStore，并守住数量、向量和结果形状契约。"""

    def __init__(
        self,
        settings: Settings,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if embedding_client is not None and not isinstance(
            embedding_client, BaseEmbedding
        ):
            raise TypeError("embedding_client must implement BaseEmbedding or be None")
        if vector_store is not None and not isinstance(vector_store, BaseVectorStore):
            raise TypeError("vector_store must implement BaseVectorStore or be None")

        self.settings = settings
        self.embedding_client = (
            embedding_client
            if embedding_client is not None
            else EmbeddingFactory.create(settings)
        )
        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStoreFactory.create(settings)
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """向量化完整查询并把数据库结果按原排名转换为 Core 类型。"""
        normalized_query = self._validate_query(query)
        selected_top_k = self._resolve_top_k(top_k)
        normalized_filters = self._validate_filters(filters)

        vectors = self.embedding_client.embed([normalized_query], trace=trace)
        query_vector = self._extract_single_vector(vectors)
        raw_results = self.vector_store.query(
            query_vector,
            selected_top_k,
            normalized_filters,
            trace=trace,
        )
        return self._normalize_results(raw_results)

    @staticmethod
    def _validate_query(query: object) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    def _resolve_top_k(self, top_k: object) -> int:
        selected = self.settings.retrieval.get("dense_top_k", 20) if top_k is None else top_k
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
            raise ValueError("top_k must be a positive integer")
        return selected

    @staticmethod
    def _validate_filters(
        filters: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if filters is None:
            return None
        if not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping or None")
        return dict(filters)

    @staticmethod
    def _extract_single_vector(vectors: object) -> list[float]:
        """单查询必须恰好对应一个非空有限向量，错位结果不能进入数据库。"""
        if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
            raise TypeError("embedding client must return a sequence of vectors")
        if len(vectors) != 1:
            raise ValueError(
                f"embedding client returned {len(vectors)} vectors for one query"
            )
        vector = vectors[0]
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise TypeError("embedding vector must be a sequence")
        if not vector:
            raise ValueError("embedding vector must not be empty")

        normalized: list[float] = []
        for index, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"embedding vector[{index}] must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"embedding vector[{index}] must be finite")
            normalized.append(number)
        return normalized

    @staticmethod
    def _normalize_results(raw_results: object) -> list[RetrievalResult]:
        if isinstance(raw_results, (str, bytes)) or not isinstance(
            raw_results, Sequence
        ):
            raise TypeError("vector store must return a sequence of results")

        normalized: list[RetrievalResult] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                raise TypeError(f"vector store result[{index}] must be a mapping")
            missing = {"chunk_id", "score", "text", "metadata"}.difference(item)
            if missing:
                raise ValueError(
                    f"vector store result[{index}] is missing fields: {sorted(missing)}"
                )
            try:
                result = RetrievalResult(
                    chunk_id=item["chunk_id"],  # type: ignore[arg-type]
                    score=item["score"],  # type: ignore[arg-type]
                    text=item["text"],  # type: ignore[arg-type]
                    metadata=item["metadata"],  # type: ignore[arg-type]
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"vector store result[{index}]: {exc}") from exc
            normalized.append(result)
        return normalized


__all__ = ["DenseRetriever"]
