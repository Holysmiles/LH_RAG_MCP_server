"""批量生成摄取阶段所需的稠密向量。

整体链路：``Chunk[] -> 文本列表 -> BaseEmbedding -> vector[]``。

本模块只连接业务 Chunk 与供应商无关的 Embedding 接口，不负责持久化。输出向量与输入
Chunk 严格同序，并在这里尽早检查数量、维度和数值是否合法；metadata 与 Chunk 身份
仍由后面的 ChunkRecord 和入库阶段管理。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from threading import Lock
from typing import TYPE_CHECKING

from core import Chunk
from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class DenseEncoder:
    """用一次 Provider 批调用编码 Chunk，并校验返回矩阵。

    输入：用于创建 Provider 的 Settings、可选注入的 BaseEmbedding、Chunk 序列。
    输出：``list[list[float]]``，每个 Chunk 对应一个同维、非空、有限浮点向量。
    失败边界：公开参数或 Provider 输出非法时立即报错；网络/模型异常原样上抛，避免后续
    把缺失向量误当成功数据。空批次直接返回 ``[]``，且不会创建模型客户端。
    """

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if embedding is not None and not isinstance(embedding, BaseEmbedding):
            raise TypeError("embedding must implement BaseEmbedding or be None")
        self.settings = settings
        self.embedding = embedding
        self._embedding_lock = Lock()

    def encode(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """编码一个批次，保持顺序并拒绝不可信的 Provider 响应。"""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
        if not chunks:
            if trace is not None:
                trace.record_stage(
                    "dense_encoder",
                    {"input_count": 0, "output_count": 0, "dimension": 0},
                )
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self._get_embedding().embed(texts, trace=trace)
        validated = self._validate_vectors(vectors, expected_count=len(chunks))
        if trace is not None:
            trace.record_stage(
                "dense_encoder",
                {
                    "input_count": len(chunks),
                    "output_count": len(validated),
                    "dimension": len(validated[0]),
                },
            )
        return validated

    def _get_embedding(self) -> BaseEmbedding:
        """首次遇到非空批次时创建 Provider，后续调用复用同一实例。"""
        if self.embedding is None:
            with self._embedding_lock:
                if self.embedding is None:
                    created = EmbeddingFactory.create(self.settings)
                    if not isinstance(created, BaseEmbedding):
                        raise TypeError("embedding factory must return BaseEmbedding")
                    self.embedding = created
        return self.embedding

    @staticmethod
    def _validate_vectors(
        vectors: object,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
            raise TypeError("embedding provider must return a sequence of vectors")
        if len(vectors) != expected_count:
            raise ValueError(
                "embedding provider returned "
                f"{len(vectors)} vectors for {expected_count} chunks"
            )

        validated: list[list[float]] = []
        dimension: int | None = None
        for row_index, vector in enumerate(vectors):
            if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                raise TypeError(f"embedding vector[{row_index}] must be a sequence")
            if not vector:
                raise ValueError(f"embedding vector[{row_index}] must not be empty")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(
                    "embedding provider returned inconsistent vector dimensions: "
                    f"vector[0]={dimension}, vector[{row_index}]={len(vector)}"
                )
            normalized: list[float] = []
            for value_index, value in enumerate(vector):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"embedding vector[{row_index}][{value_index}] must be numeric"
                    )
                number = float(value)
                if not isfinite(number):
                    raise ValueError(
                        f"embedding vector[{row_index}][{value_index}] must be finite"
                    )
                normalized.append(number)
            validated.append(normalized)
        return validated


__all__ = ["DenseEncoder"]
