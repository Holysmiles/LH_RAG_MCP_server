"""把已经编码的 Chunk 转换成可幂等写入向量库的记录。

整体链路：``Chunk[] + dense_vectors[] -> 稳定 ID -> BaseVectorStore.upsert``。

本模块是摄取业务对象与向量库适配器之间的边界，不负责生成向量，也不决定检索策略。
稳定 ID 由规范化源路径、Chunk 序号和完整内容哈希共同决定：相同内容重复摄取会覆盖同一
记录，内容改变则产生新记录。所有输入必须在调用 Store 前完成校验，避免错位数据造成
部分写入。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any

from core import Chunk, ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory
from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class VectorUpserter:
    """构建可序列化向量记录，并以一个幂等批次写入。

    输入：向量库 Settings、Chunk 序列和严格同序的一组稠密向量；测试可注入 Store。
    输出：实际发送给 Store 的稳定 ChunkRecord，供调用方审计或继续持久化。
    失败边界：输入、向量数量或维度非法时，Store 一次也不会被调用；数据库或网络异常
    原样上抛，避免把部分写入错误标记成完整成功。
    """

    def __init__(
        self,
        settings: Settings,
        store: BaseVectorStore | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if store is not None and not isinstance(store, BaseVectorStore):
            raise TypeError("store must implement BaseVectorStore or be None")
        self.settings = settings
        self.store = store if store is not None else VectorStoreFactory.create(settings)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[Real]],
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        """校验、构建并写入一个 Chunk 与向量严格对齐的批次。"""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")
        if isinstance(dense_vectors, (str, bytes)) or not isinstance(dense_vectors, Sequence):
            raise TypeError("dense_vectors must be a non-string sequence")
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks and dense_vectors must have the same length: {len(chunks)} != {len(dense_vectors)}"
            )
        if not chunks:
            if trace is not None:
                trace.record_stage(
                    "vector_upserter",
                    {"input_count": 0, "upsert_count": 0},
                )
            return []
        records: list[ChunkRecord] = []
        seen_ids: set[str] = set()
        vector_size: int | None = None
        for index, (chunk, vector) in enumerate(zip(chunks, dense_vectors, strict=True)):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            normalized_vector = self._validate_vector(vector, index)
            if vector_size is None:
                vector_size = len(normalized_vector)
            elif len(normalized_vector) != vector_size:
                raise ValueError("all dense vectors must have the same dimension")
            chunk_id = self.generate_chunk_id(chunk)
            if chunk_id in seen_ids:
                raise ValueError(f"duplicate generated chunk id: {chunk_id}")
            seen_ids.add(chunk_id)
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    text=chunk.text,
                    metadata=self._record_metadata(chunk, chunk_id),
                    dense_vector=normalized_vector,
                )
            )

        self.store.upsert(  # type: ignore[union-attr]
            [record.to_dict() for record in records], trace=trace
        )
        if trace is not None:
            trace.record_stage(
                "vector_upserter",
                {"input_count": len(chunks), "upsert_count": len(records)},
            )
        return records

    @staticmethod
    def generate_chunk_id(chunk: Chunk) -> str:
        """生成跨重复摄取和路径分隔符形式都一致的确定性 ID。"""
        if not isinstance(chunk, Chunk):
            raise TypeError("chunk must be a Chunk")
        source_path = chunk.metadata.get("source_path")
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("chunk.metadata.source_path must be a non-empty string")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("chunk.metadata.chunk_index must be a non-negative integer")
        normalized_source = source_path.strip().replace("\\", "/")
        while "//" in normalized_source:
            normalized_source = normalized_source.replace("//", "/")
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        identity = f"{normalized_source}|{chunk_index}|{content_hash}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"chunk_{digest}"

    @staticmethod
    def _record_metadata(chunk: Chunk, chunk_id: str) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = chunk_id
        metadata["source_ref"] = chunk.source_ref
        metadata["start_offset"] = chunk.start_offset
        metadata["end_offset"] = chunk.end_offset
        return metadata

    @staticmethod
    def _validate_vector(vector: object, index: int) -> list[float]:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise TypeError(f"dense_vectors[{index}] must be a sequence")
        if not vector:
            raise ValueError(f"dense_vectors[{index}] must not be empty")
        normalized: list[float] = []
        for position, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"dense_vectors[{index}][{position}] must be numeric")
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"dense_vectors[{index}][{position}] must be finite")
            normalized.append(number)
        return normalized


__all__ = ["VectorUpserter"]
