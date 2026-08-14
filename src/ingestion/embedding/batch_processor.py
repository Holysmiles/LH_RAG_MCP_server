"""协调摄取阶段的稠密与稀疏编码批次。

整体链路：``Chunk[] -> 固定大小分批 -> Dense/Sparse Encoder -> 顺序聚合``。

本模块只负责调度和顺序：分批限制单次模型请求和内存占用，完成后按原顺序拼接。任何
Provider 失败都会上抛，因为静默丢弃一个批次会让 Chunk 与向量错位，最终污染索引。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from core import Chunk
from core.settings import Settings
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseChunkEncoding, SparseEncoder

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass(frozen=True, slots=True)
class BatchProcessResult:
    """一次完整编码过程的聚合结果。"""

    chunks: tuple[Chunk, ...]
    dense_vectors: tuple[tuple[float, ...], ...]
    sparse_records: tuple[SparseChunkEncoding, ...]
    batch_count: int

    def to_dict(self) -> dict[str, Any]:
        """返回便于日志和诊断使用的 JSON 兼容字典。"""
        return {
            "chunk_ids": [chunk.id for chunk in self.chunks],
            "dense_vectors": [list(vector) for vector in self.dense_vectors],
            "sparse_records": [record.to_dict() for record in self.sparse_records],
            "batch_count": self.batch_count,
        }


class BatchProcessor:
    """分割 Chunk 批次并驱动两种 Encoder，始终保持输入顺序。

    输入：Settings 中的 ``ingestion.batch_size``、Chunk 序列，以及可选注入的两个
    Encoder。输出：每个 Chunk 对应一个稠密向量和一条稀疏记录，并报告批次数。
    失败边界：空输入不触发模型工作；配置、Chunk、批大小或编码结果不合法时立即报错；
    Provider 异常上抛，调用方不得把部分编码结果持久化为完整任务。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if dense_encoder is not None and not isinstance(dense_encoder, DenseEncoder):
            raise TypeError("dense_encoder must be DenseEncoder or None")
        if sparse_encoder is not None and not isinstance(sparse_encoder, SparseEncoder):
            raise TypeError("sparse_encoder must be SparseEncoder or None")
        self.settings = settings
        self.dense_encoder = (
            dense_encoder if dense_encoder is not None else DenseEncoder(settings)
        )
        self.sparse_encoder = (
            sparse_encoder if sparse_encoder is not None else SparseEncoder()
        )
        configured_size = settings.ingestion.get("batch_size", 100)
        selected_size = configured_size if batch_size is None else batch_size
        if isinstance(selected_size, bool) or not isinstance(selected_size, int):
            raise TypeError("batch_size must be a positive integer")
        if selected_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = selected_size

    def process(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> BatchProcessResult:
        """逐批编码所有 Chunk，并按原始顺序拼接结果。"""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
        if not chunks:
            return BatchProcessResult((), (), (), 0)

        all_dense: list[tuple[float, ...]] = []
        all_sparse: list[SparseChunkEncoding] = []
        batch_count = 0
        vector_dimension: int | None = None
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            started = perf_counter()
            dense_vectors = self.dense_encoder.encode(batch, trace=trace)
            sparse_batch = self.sparse_encoder.encode(batch)
            if len(dense_vectors) != len(batch) or len(sparse_batch) != len(batch):
                raise RuntimeError("encoder output count does not match batch size")
            validated_dense = DenseEncoder._validate_vectors(
                dense_vectors,
                expected_count=len(batch),
            )
            current_dimension = len(validated_dense[0])
            if vector_dimension is None:
                vector_dimension = current_dimension
            elif current_dimension != vector_dimension:
                raise ValueError(
                    "dense vector dimension changed between batches: "
                    f"expected {vector_dimension}, got {current_dimension}"
                )
            all_dense.extend(tuple(vector) for vector in validated_dense)
            all_sparse.extend(sparse_batch.records)
            if trace is not None:
                trace.record_stage(
                    "batch_processor",
                    {
                        "batch_index": batch_count,
                        "input_start": start,
                        "input_count": len(batch),
                        "duration_ms": round((perf_counter() - started) * 1000, 3),
                    },
                )
            batch_count += 1

        return BatchProcessResult(
            chunks=tuple(chunks),
            dense_vectors=tuple(all_dense),
            sparse_records=tuple(all_sparse),
            batch_count=batch_count,
        )

    def run(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> BatchProcessResult:
        """供把流水线步骤统一命名为 ``run`` 的调用方使用。"""
        return self.process(chunks, trace=trace)


__all__ = ["BatchProcessResult", "BatchProcessor"]
