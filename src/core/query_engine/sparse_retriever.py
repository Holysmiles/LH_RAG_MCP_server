"""用 BM25 排名召回 ID，再从向量存储批量补全正文与元数据。

BM25 的职责是低成本关键词排序，因此索引只保存词频与 chunk_id；可返回给用户的正文
仍以 VectorStore 为事实来源。SparseRetriever 负责合并这两份数据，并确保最终顺序始终
来自 BM25 分数，而不是依赖数据库 ``get`` 的偶然返回顺序。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.settings import Settings
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class SparseRetriever:
    """组合 BM25Indexer 与 VectorStore，输出完整 ``RetrievalResult``。"""

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if bm25_indexer is not None and not isinstance(bm25_indexer, BM25Indexer):
            raise TypeError("bm25_indexer must be BM25Indexer or None")
        if vector_store is not None and not isinstance(vector_store, BaseVectorStore):
            raise TypeError("vector_store must implement BaseVectorStore or be None")

        self.settings = settings
        self.bm25_indexer = (
            bm25_indexer if bm25_indexer is not None else self._load_index(settings)
        )
        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStoreFactory.create(settings)
        )

    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """按 BM25 排名返回存在正文的结果；过期的孤立 ID 会被安全跳过。"""
        normalized_keywords = self._validate_keywords(keywords)
        selected_top_k = self._resolve_top_k(top_k)
        if not normalized_keywords:
            return []

        raw_hits = self.bm25_indexer.query(
            " ".join(normalized_keywords),
            top_k=selected_top_k,
        )
        hits = self._normalize_hits(raw_hits)
        if not hits:
            return []

        chunk_ids = [chunk_id for chunk_id, _ in hits]
        stored = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        records_by_id = self._index_stored_records(stored, requested_ids=chunk_ids)

        # VectorStore 可以省略不存在的 ID。按 hits 再次遍历才能保住 BM25 排名，并
        # 让跨存储部分写入产生的孤立 posting 降级为少一条结果，而非伪造正文。
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                text=records_by_id[chunk_id]["text"],
                metadata=records_by_id[chunk_id]["metadata"],
            )
            for chunk_id, score in hits
            if chunk_id in records_by_id
        ]

    def _resolve_top_k(self, top_k: object) -> int:
        selected = (
            self.settings.retrieval.get("sparse_top_k", 20)
            if top_k is None
            else top_k
        )
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
            raise ValueError("top_k must be a positive integer")
        return selected

    @staticmethod
    def _validate_keywords(keywords: object) -> list[str]:
        if isinstance(keywords, (str, bytes)) or not isinstance(keywords, Sequence):
            raise TypeError("keywords must be a non-string sequence")
        normalized: list[str] = []
        for index, keyword in enumerate(keywords):
            if not isinstance(keyword, str):
                raise TypeError(f"keywords[{index}] must be a string")
            if keyword.strip():
                normalized.append(keyword.strip().casefold())
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _normalize_hits(raw_hits: object) -> list[tuple[str, float]]:
        if isinstance(raw_hits, (str, bytes)) or not isinstance(raw_hits, Sequence):
            raise TypeError("BM25 indexer must return a sequence of hits")
        hits: list[tuple[str, float]] = []
        seen: set[str] = set()
        for index, hit in enumerate(raw_hits):
            if not isinstance(hit, Mapping):
                raise TypeError(f"BM25 hit[{index}] must be a mapping")
            chunk_id = hit.get("chunk_id")
            score = hit.get("score")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"BM25 hit[{index}].chunk_id must be non-empty")
            normalized_id = chunk_id.strip()
            if normalized_id in seen:
                raise ValueError(f"BM25 returned duplicate chunk_id '{normalized_id}'")
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError(f"BM25 hit[{index}].score must be numeric")
            normalized_score = float(score)
            if not math.isfinite(normalized_score):
                raise ValueError(f"BM25 hit[{index}].score must be finite")
            seen.add(normalized_id)
            hits.append((normalized_id, normalized_score))
        return hits

    @staticmethod
    def _index_stored_records(
        records: object,
        *,
        requested_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("vector store must return a sequence of records")
        requested = set(requested_ids)
        indexed: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"vector store record[{index}] must be a mapping")
            missing = {"id", "text", "metadata"}.difference(record)
            if missing:
                raise ValueError(
                    f"vector store record[{index}] is missing fields: {sorted(missing)}"
                )
            chunk_id = record["id"]
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"vector store record[{index}].id must be non-empty")
            normalized_id = chunk_id.strip()
            if normalized_id not in requested:
                raise ValueError("vector store returned an unrequested chunk_id")
            if normalized_id in indexed:
                raise ValueError("vector store returned a duplicate chunk_id")
            text = record["text"]
            metadata = record["metadata"]
            if not isinstance(text, str):
                raise TypeError(f"vector store record[{index}].text must be a string")
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    f"vector store record[{index}].metadata must be a mapping"
                )
            indexed[normalized_id] = {
                "text": text,
                "metadata": dict(metadata),
            }
        return indexed

    @staticmethod
    def _load_index(settings: Settings) -> BM25Indexer:
        value = settings.ingestion.get("bm25_path", BM25Indexer.default_path)
        if not isinstance(value, (str, Path)) or (
            isinstance(value, str) and not value.strip()
        ):
            raise ValueError("ingestion.bm25_path must be a non-empty path")
        return BM25Indexer(value)


__all__ = ["SparseRetriever"]
