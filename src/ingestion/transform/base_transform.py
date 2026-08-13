"""Ingestion Transform 的统一抽象接口。

Transform 位于切分和编码之间：

``list[Chunk] -> BaseTransform.transform(...) -> list[Chunk]``

输入已经具有稳定 ID、source_ref 和原文 offset；输出可以增强正文和 metadata，
但不能丢失这些溯源字段。具体实现可以使用规则或外部模型，调用方只依赖这一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from core import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """所有 Chunk 增强器必须实现的批量输入/输出契约。"""

    @abstractmethod
    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """返回处理后的新 Chunk 列表，并保持输入顺序和业务身份。"""
        raise NotImplementedError


__all__ = ["BaseTransform"]
