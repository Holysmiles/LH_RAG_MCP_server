"""用 Reciprocal Rank Fusion 合并 Dense 与 Sparse 两路候选。

Dense cosine score 与 BM25 score 的数值范围没有共同语义，直接相加会让某一路因量纲
更大而主导结果。RRF 只使用候选在各自列表中的名次，把每一路贡献转换为
``1 / (k + rank)``，再按稳定 ``chunk_id`` 聚合同一条 Chunk。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.settings import Settings
from core.types import RetrievalResult


class Fusion:
    """计算确定性的双路 RRF 排名，并返回统一 ``RetrievalResult``。"""

    default_rrf_k: ClassVar[int] = 60
    default_top_k: ClassVar[int] = 10

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        k: int | None = None,
        top_k: int | None = None,
    ) -> None:
        if settings is not None and not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance or None")

        configured_k = (
            self.default_rrf_k
            if settings is None
            else settings.retrieval.get("rrf_k", self.default_rrf_k)
        )
        configured_top_k = (
            self.default_top_k
            if settings is None
            else settings.retrieval.get("fusion_top_k", self.default_top_k)
        )
        self.k = self._validate_k(configured_k if k is None else k)
        self.top_k = self._validate_top_k(
            configured_top_k if top_k is None else top_k
        )

    def fuse(
        self,
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """融合两路排名；空列表代表该路没有召回结果，可自然降级到另一条路。"""
        dense = self._validate_ranking(dense_results, route="dense")
        sparse = self._validate_ranking(sparse_results, route="sparse")
        selected_top_k = self.top_k if top_k is None else self._validate_top_k(top_k)

        scores: dict[str, float] = {}
        payloads: dict[str, RetrievalResult] = {}
        first_seen: dict[str, int] = {}
        for route, ranking in (("dense", dense), ("sparse", sparse)):
            seen_in_route: set[str] = set()
            for rank, result in enumerate(ranking, start=1):
                if result.chunk_id in seen_in_route:
                    raise ValueError(
                        f"{route} ranking contains duplicate chunk_id "
                        f"'{result.chunk_id}'"
                    )
                seen_in_route.add(result.chunk_id)

                previous = payloads.get(result.chunk_id)
                if previous is not None and (
                    previous.text != result.text
                    or previous.metadata != result.metadata
                ):
                    raise ValueError(
                        "retrieval routes returned conflicting payloads for "
                        f"chunk_id '{result.chunk_id}'"
                    )
                payloads.setdefault(result.chunk_id, result)
                first_seen.setdefault(result.chunk_id, len(first_seen))

                # 面试核心：Provider 原始 score 被有意丢弃，只累加名次倒数。
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (
                    1.0 / (self.k + rank)
                )

        # Python 稳定排序 + first_seen 使同分项保留 Dense/Sparse 的首次证据顺序。
        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]),
        )
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                text=payloads[chunk_id].text,
                metadata=payloads[chunk_id].metadata,
            )
            for chunk_id in ranked_ids[:selected_top_k]
        ]

    @staticmethod
    def _validate_ranking(
        results: object,
        *,
        route: str,
    ) -> tuple[RetrievalResult, ...]:
        if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
            raise TypeError(f"{route}_results must be a non-string sequence")
        normalized = tuple(results)
        for index, result in enumerate(normalized):
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    f"{route}_results[{index}] must be a RetrievalResult"
                )
        return normalized

    @staticmethod
    def _validate_k(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("k must be a positive integer")
        return value

    @staticmethod
    def _validate_top_k(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("top_k must be a positive integer")
        return value


# 保留更明确的算法名，调用方可按职责名或算法名导入同一个实现。
RRFFusion = Fusion


__all__ = ["Fusion", "RRFFusion"]
