"""贯穿摄取、存储和检索链路的核心数据契约。

可以把这三个类型理解成同一份知识在流水线不同阶段使用的“快递箱”：

``原始文件 -> Document -> Chunk -> ChunkRecord -> Chroma/BM25 -> 检索结果``

- ``Document`` 是 Loader 的输出：一整个文件的规范化文本和来源信息；
- ``Chunk`` 是 DocumentChunker 的输出：文档中的一小段文本和定位信息；
- ``ChunkRecord`` 是 Encoder 的输出：Chunk 加上可选稠密/稀疏向量，可入库。

所有类型的共同输入都是 ``id``、``text`` 和 ``metadata``。``metadata`` 至少
包含非空 ``source_path``，其余业务字段可以继续增加。共同输出是稳定的 Python
字典或 JSON 文本，因此 Loader、Transformer、Storage 和 MCP 层不需要各自猜测
字段名称，也不需要依赖某个数据库 SDK 的对象类型。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from typing import Any, Required, TypeAlias, TypedDict


Metadata: TypeAlias = dict[str, Any]
DenseVector: TypeAlias = list[float]
SparseVector: TypeAlias = dict[str, float]


class ImageRef(TypedDict, total=False):
    """``metadata.images`` 中单张图片的字段形状。

    ``id/path/text_offset/text_length`` 是定位图片所需的必填字段；``page`` 和
    ``position`` 仅在 PDF Loader 能获取页码或坐标时提供。
    """

    id: Required[str]
    path: Required[str]
    text_offset: Required[int]
    text_length: Required[int]
    page: int
    position: dict[str, Any]


class _SerializableContract:
    """为三个 dataclass 提供相同的 dict/JSON 输出方式。"""

    def to_dict(self) -> dict[str, Any]:
        """返回可安全修改的深拷贝字典，不暴露对象内部 metadata。"""
        return asdict(self)  # type: ignore[arg-type]

    def to_json(self) -> str:
        """返回保留中文的 JSON 文本，供日志、文件或 MCP 响应使用。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass(slots=True)
class Document(_SerializableContract):
    """Loader 解析一个完整源文件后输出的统一对象。

    Inputs:
        id: 文档稳定 ID；后续 Chunk ID 和溯源关系都以它为父节点。
        text: PDF/Markdown 等源文件转换后的规范化全文，可包含图片占位符。
        metadata: 至少含 ``source_path``；可扩展标题、页数、images 等字段。

    Outputs:
        ``to_dict()`` / ``to_json()`` 输出相同的 ``id/text/metadata`` 字段。
    """

    id: str
    text: str
    metadata: Metadata

    def __post_init__(self) -> None:
        _validate_identity_and_text("Document", self.id, self.text)
        self.metadata = _copy_and_validate_metadata("Document", self.metadata)


@dataclass(slots=True)
class Chunk(_SerializableContract):
    """Document 经业务切分后得到的一段可独立处理的文本。

    Inputs:
        id/text/metadata: 当前切块的稳定 ID、正文和来源信息。
        start_offset/end_offset: 该段在父文档文本中的左闭右开字符范围。
        source_ref: 可选父 ``Document.id``，用于检索结果回溯原文档。

    Outputs:
        可直接交给 Transform、DenseEncoder、SparseEncoder 的稳定对象。
    """

    id: str
    text: str
    metadata: Metadata
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_identity_and_text("Chunk", self.id, self.text)
        self.metadata = _copy_and_validate_metadata("Chunk", self.metadata)
        self.start_offset = _validate_non_negative_int(
            "Chunk.start_offset",
            self.start_offset,
        )
        self.end_offset = _validate_non_negative_int(
            "Chunk.end_offset",
            self.end_offset,
        )
        if self.end_offset < self.start_offset:
            raise ValueError(
                "Chunk.end_offset must be greater than or equal to start_offset"
            )
        if self.source_ref is not None:
            self.source_ref = _validate_non_empty_text(
                "Chunk.source_ref",
                self.source_ref,
            )


@dataclass(slots=True)
class ChunkRecord(_SerializableContract):
    """准备写入向量库或 BM25 索引的 Chunk 存储载体。

    Inputs:
        id/text/metadata: 从 Chunk 继承的业务数据。
        dense_vector: DenseEncoder 生成的有限浮点数列表；尚未编码时为 ``None``。
        sparse_vector: SparseEncoder 生成的 ``term -> weight``；未编码时为 ``None``。

    Outputs:
        可 JSON 序列化的入库记录。向量字段保持可选，允许 Dense 和 Sparse
        编码在不同批次完成，而不用创建两套互不兼容的记录类型。
    """

    id: str
    text: str
    metadata: Metadata
    dense_vector: DenseVector | None = None
    sparse_vector: SparseVector | None = None

    def __post_init__(self) -> None:
        _validate_identity_and_text("ChunkRecord", self.id, self.text)
        self.metadata = _copy_and_validate_metadata(
            "ChunkRecord",
            self.metadata,
        )
        self.dense_vector = _validate_dense_vector(self.dense_vector)
        self.sparse_vector = _validate_sparse_vector(self.sparse_vector)


def _validate_identity_and_text(owner: str, record_id: object, text: object) -> None:
    _validate_non_empty_text(f"{owner}.id", record_id)
    if not isinstance(text, str):
        raise TypeError(f"{owner}.text must be a string")


def _validate_non_empty_text(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _copy_and_validate_metadata(owner: str, metadata: object) -> Metadata:
    """复制 metadata 并校验所有下游都依赖的最小公共字段。"""
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{owner}.metadata must be a mapping")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError(f"{owner}.metadata keys must be strings")

    copied_metadata = deepcopy(dict(metadata))
    _validate_non_empty_text(
        f"{owner}.metadata.source_path",
        copied_metadata.get("source_path"),
    )
    if "images" in copied_metadata:
        _validate_image_refs(owner, copied_metadata["images"])

    try:
        json.dumps(copied_metadata, ensure_ascii=False)
    except (TypeError, ValueError):
        raise ValueError(
            f"{owner}.metadata must contain only JSON-serializable values"
        ) from None
    return copied_metadata


def _validate_image_refs(owner: str, images: object) -> None:
    """校验 C1 规定的 metadata.images 列表，不限制额外扩展字段。"""
    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
        raise TypeError(f"{owner}.metadata.images must be a sequence")

    for index, image in enumerate(images):
        prefix = f"{owner}.metadata.images[{index}]"
        if not isinstance(image, Mapping):
            raise TypeError(f"{prefix} must be a mapping")
        _validate_non_empty_text(f"{prefix}.id", image.get("id"))
        _validate_non_empty_text(f"{prefix}.path", image.get("path"))
        _validate_non_negative_int(
            f"{prefix}.text_offset",
            image.get("text_offset"),
        )
        text_length = _validate_non_negative_int(
            f"{prefix}.text_length",
            image.get("text_length"),
        )
        if text_length == 0:
            raise ValueError(f"{prefix}.text_length must be greater than zero")
        if "page" in image:
            _validate_non_negative_int(f"{prefix}.page", image["page"])
        if "position" in image and not isinstance(image["position"], Mapping):
            raise TypeError(f"{prefix}.position must be a mapping")


def _validate_non_negative_int(field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_dense_vector(vector: object) -> DenseVector | None:
    if vector is None:
        return None
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise TypeError("ChunkRecord.dense_vector must be a sequence")
    if not vector:
        raise ValueError("ChunkRecord.dense_vector must not be empty")
    return [
        _validate_finite_number(f"ChunkRecord.dense_vector[{index}]", value)
        for index, value in enumerate(vector)
    ]


def _validate_sparse_vector(vector: object) -> SparseVector | None:
    if vector is None:
        return None
    if not isinstance(vector, Mapping):
        raise TypeError("ChunkRecord.sparse_vector must be a mapping")

    normalized: SparseVector = {}
    for term, weight in vector.items():
        normalized_term = _validate_non_empty_text(
            "ChunkRecord.sparse_vector term",
            term,
        )
        normalized[normalized_term] = _validate_finite_number(
            f"ChunkRecord.sparse_vector[{normalized_term!r}]",
            weight,
        )
    return normalized


def _validate_finite_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


__all__ = [
    "Chunk",
    "ChunkRecord",
    "DenseVector",
    "Document",
    "ImageRef",
    "Metadata",
    "SparseVector",
]
