"""C1 核心数据类型的契约测试。

这些测试把 Loader -> Chunker -> Encoder -> Storage 之间的字段形状固定下来。
只要这里保持通过，后续模块就可以依赖同一组对象，不需要重复写字典键猜测逻辑。
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from core import Chunk, ChunkRecord, Document


@pytest.fixture
def metadata_with_image() -> dict[str, Any]:
    """构造符合 C1 图片定位规范、且带可扩展字段的 metadata。"""
    placeholder = "[IMAGE: doc123_2_1]"
    return {
        "source_path": "tests/fixtures/report.pdf",
        "doc_type": "pdf",
        "title": "Quarterly report",
        "images": [
            {
                "id": "doc123_2_1",
                "path": "data/images/test/doc123_2_1.png",
                "page": 2,
                "text_offset": 18,
                "text_length": len(placeholder),
                "position": {"x": 10, "y": 20, "width": 400, "height": 200},
            }
        ],
        "custom_future_field": {"enabled": True},
    }


@pytest.mark.unit
def test_document_serializes_to_stable_dict_and_json(
    metadata_with_image: dict[str, Any],
) -> None:
    """Loader 输出应完整保留文本、来源、图片引用和未来扩展字段。"""
    document = Document(
        id="doc-123",
        text="季度报告\n\n[IMAGE: doc123_2_1]",
        metadata=metadata_with_image,
    )

    expected = {
        "id": "doc-123",
        "text": "季度报告\n\n[IMAGE: doc123_2_1]",
        "metadata": metadata_with_image,
    }
    assert document.to_dict() == expected
    assert json.loads(document.to_json()) == expected
    assert "季度报告" in document.to_json()


@pytest.mark.unit
def test_contract_copies_metadata_at_input_and_output(
    metadata_with_image: dict[str, Any],
) -> None:
    """调用方修改原字典或 to_dict 结果时，不能偷偷改变对象内部数据。"""
    document = Document("doc-123", "text", metadata_with_image)

    metadata_with_image["title"] = "changed outside"
    serialized = document.to_dict()
    serialized["metadata"]["title"] = "changed output"

    assert document.metadata["title"] == "Quarterly report"


@pytest.mark.unit
def test_chunk_contains_offsets_and_parent_document_reference() -> None:
    """Chunk 输出必须能回答“来自哪个文档、位于原文什么范围”。"""
    chunk = Chunk(
        id="doc-123_0000_abcd1234",
        text="first chunk",
        metadata={"source_path": "report.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=11,
        source_ref="doc-123",
    )

    assert chunk.to_dict() == {
        "id": "doc-123_0000_abcd1234",
        "text": "first chunk",
        "metadata": {"source_path": "report.pdf", "chunk_index": 0},
        "start_offset": 0,
        "end_offset": 11,
        "source_ref": "doc-123",
    }


@pytest.mark.unit
def test_chunk_record_supports_unencoded_and_encoded_states() -> None:
    """编码前向量可为空；编码后稠密/稀疏向量都应转换为稳定浮点数。"""
    unencoded = ChunkRecord(
        id="chunk-1",
        text="RAG architecture",
        metadata={"source_path": "architecture.pdf"},
    )
    encoded = ChunkRecord(
        id="chunk-1",
        text="RAG architecture",
        metadata={"source_path": "architecture.pdf"},
        dense_vector=[1, 0.25, -0.5],
        sparse_vector={"rag": 2, "architecture": 0.75},
    )

    assert unencoded.dense_vector is None
    assert unencoded.sparse_vector is None
    assert encoded.dense_vector == [1.0, 0.25, -0.5]
    assert encoded.sparse_vector == {"rag": 2.0, "architecture": 0.75}
    assert json.loads(encoded.to_json())["dense_vector"] == [1.0, 0.25, -0.5]


@pytest.mark.unit
@pytest.mark.parametrize("record_type", [Document, Chunk, ChunkRecord])
def test_every_contract_requires_source_path(record_type: type[object]) -> None:
    """没有来源的记录无法引用和删除，因此所有阶段都必须尽早拒绝。"""
    common: dict[str, Any] = {"id": "item-1", "text": "text", "metadata": {}}
    if record_type is Chunk:
        common.update(start_offset=0, end_offset=4)

    with pytest.raises(ValueError, match=r"metadata\.source_path"):
        record_type(**common)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("images", "message"),
    [
        ("not-a-list", "images must be a sequence"),
        ([{"path": "image.png", "text_offset": 0, "text_length": 5}], r"\.id"),
        ([{"id": "img", "path": "image.png", "text_offset": -1, "text_length": 5}], "text_offset"),
        ([{"id": "img", "path": "image.png", "text_offset": 0, "text_length": 0}], "text_length"),
        ([{"id": "img", "path": "image.png", "text_offset": 0, "text_length": 5, "position": "top"}], "position"),
    ],
)
def test_invalid_image_reference_shape_is_rejected(
    images: object,
    message: str,
) -> None:
    """坏图片定位必须在 C1 暴露，否则 C4/C7 会在更远处得到难懂错误。"""
    with pytest.raises((TypeError, ValueError), match=message):
        Document(
            id="doc-1",
            text="[IMAGE: img]",
            metadata={"source_path": "report.pdf", "images": images},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start_offset", "end_offset", "message"),
    [
        (-1, 4, "start_offset"),
        (0, -1, "end_offset"),
        (5, 4, "greater than or equal"),
        (True, 4, "start_offset"),
    ],
)
def test_invalid_chunk_offsets_are_rejected(
    start_offset: int,
    end_offset: int,
    message: str,
) -> None:
    """偏移必须是非负左闭右开范围，防止引用原文时截取错误。"""
    with pytest.raises(ValueError, match=message):
        Chunk(
            "chunk-1",
            "text",
            {"source_path": "report.pdf"},
            start_offset,
            end_offset,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dense_vector", "message"),
    [
        ([], "must not be empty"),
        ([0.1, math.nan], "must be finite"),
        ([0.1, math.inf], "must be finite"),
        ([0.1, True], "real number"),
    ],
)
def test_invalid_dense_vectors_are_rejected(
    dense_vector: list[object],
    message: str,
) -> None:
    """不能让 NaN/Infinity/bool 进入向量库破坏距离计算。"""
    with pytest.raises((TypeError, ValueError), match=message):
        ChunkRecord(
            "chunk-1",
            "text",
            {"source_path": "report.pdf"},
            dense_vector=dense_vector,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_non_json_metadata_is_rejected() -> None:
    """Storage 和 MCP 最终需要 JSON，因此不可序列化值应在入口被拒绝。"""
    with pytest.raises(ValueError, match="JSON-serializable"):
        Document(
            "doc-1",
            "text",
            {"source_path": "report.pdf", "bad": object()},
        )
