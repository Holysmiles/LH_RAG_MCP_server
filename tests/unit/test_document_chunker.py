"""C4 ``Document -> Chunk`` 适配层的契约测试。

测试把 ``BaseSplitter`` 替换成可预测的 Fake，只验证 DocumentChunker 自己增加的
业务价值：稳定 ID、原文 offset、metadata 继承、父文档链接和图片按需分发。
真实递归算法只在配置驱动测试中使用，避免把第三方细节混入适配器契约。
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
from typing import Any

import pytest

from core import Chunk, Document
from core.settings import Settings
from ingestion.chunking import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class PipeSplitter(BaseSplitter):
    """按竖线切分的 Fake；每个返回值都是父文本中的精确子串。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return text.split("|")


class OverlapSplitter(BaseSplitter):
    """返回固定重叠片段，用来验证 offset 不假设 chunks 首尾相接。"""

    def __init__(self, config: dict[str, Any]) -> None:
        del config

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        del text, trace
        return ["abcdefgh", "ghijklmn", "mnop"]


@pytest.fixture
def fake_splitters() -> Iterator[None]:
    SplitterFactory.register("c4-pipe", PipeSplitter)
    SplitterFactory.register("c4-overlap", OverlapSplitter)
    yield
    SplitterFactory.unregister("c4-pipe")
    SplitterFactory.unregister("c4-overlap")


def make_document(text: str = "alpha|beta|gamma") -> Document:
    """生成只含所有 Chunk 都应继承的公共 metadata 的父文档。"""
    return Document(
        id="doc-123",
        text=text,
        metadata={
            "source_path": "reports/source.pdf",
            "doc_type": "pdf",
            "title": "RAG design",
        },
    )


@pytest.mark.unit
def test_converts_splitter_strings_to_complete_chunks(
    fake_splitters: None,
) -> None:
    """输出必须同时具有类型、父引用、序号、公共 metadata 和真实 offset。"""
    document = make_document()
    chunker = DocumentChunker(Settings(splitter={"provider": "c4-pipe"}))

    chunks = chunker.split_document(document)

    assert [chunk.text for chunk in chunks] == ["alpha", "beta", "gamma"]
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.source_ref for chunk in chunks] == [document.id] * 3
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 5),
        (6, 10),
        (11, 16),
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["title"] == "RAG design" for chunk in chunks)
    assert all(
        document.text[chunk.start_offset : chunk.end_offset] == chunk.text
        for chunk in chunks
    )


@pytest.mark.unit
def test_chunk_ids_are_unique_deterministic_and_follow_contract(
    fake_splitters: None,
) -> None:
    """同文档重复运行 ID 不变，相同文字也由 index 保证 Chunk ID 不冲突。"""
    document = make_document("same|same")
    chunker = DocumentChunker(Settings(splitter={"provider": "c4-pipe"}))
    expected_hash = hashlib.sha256(b"same").hexdigest()[:8]

    first_ids = [chunk.id for chunk in chunker.split_document(document)]
    second_ids = [chunk.id for chunk in chunker.split_document(document)]

    assert first_ids == second_ids
    assert first_ids == [
        f"doc-123_0000_{expected_hash}",
        f"doc-123_0001_{expected_hash}",
    ]
    assert len(first_ids) == len(set(first_ids))


@pytest.mark.unit
def test_offsets_support_overlapping_splitter_output(
    fake_splitters: None,
) -> None:
    """chunk_overlap 会让相邻片段重复，offset 仍必须指向各自真实起点。"""
    document = make_document("abcdefghijklmnop")
    chunker = DocumentChunker(Settings(splitter={"provider": "c4-overlap"}))

    chunks = chunker.split_document(document)

    assert [(item.start_offset, item.end_offset) for item in chunks] == [
        (0, 8),
        (6, 14),
        (12, 16),
    ]


@pytest.mark.unit
def test_distributes_only_images_referenced_by_each_chunk(
    fake_splitters: None,
) -> None:
    """普通 Chunk 不携带图片；带占位符的 Chunk 只拿到自己的 ID 和文件信息。"""
    first_placeholder = "[IMAGE: image-a]"
    second_placeholder = "[IMAGE: image-b]"
    text = f"intro {first_placeholder}|plain text|{second_placeholder} ending"
    document = Document(
        id="doc-images",
        text=text,
        metadata={
            "source_path": "reports/images.pdf",
            "doc_type": "pdf",
            "images": [
                {
                    "id": "image-a",
                    "path": "data/images/image-a.png",
                    "text_offset": text.index(first_placeholder),
                    "text_length": len(first_placeholder),
                },
                {
                    "id": "image-b",
                    "path": "data/images/image-b.png",
                    "text_offset": text.index(second_placeholder),
                    "text_length": len(second_placeholder),
                },
            ],
        },
    )
    chunker = DocumentChunker(Settings(splitter={"provider": "c4-pipe"}))

    first, plain, second = chunker.split_document(document)

    assert first.metadata["image_refs"] == ["image-a"]
    assert [item["id"] for item in first.metadata["images"]] == ["image-a"]
    assert "images" not in plain.metadata
    assert "image_refs" not in plain.metadata
    assert second.metadata["image_refs"] == ["image-b"]
    assert [item["id"] for item in second.metadata["images"]] == ["image-b"]
    assert document.metadata["images"][0]["id"] == "image-a"


@pytest.mark.unit
def test_real_splitter_configuration_changes_chunking() -> None:
    """DocumentChunker 必须服从 Settings，而不是在业务层写死 chunk_size。"""
    document = make_document("abcdefghijklmnopqrstuvwxyz")
    small = DocumentChunker(
        Settings(
            splitter={
                "provider": "recursive",
                "chunk_size": 8,
                "chunk_overlap": 0,
            }
        )
    )
    large = DocumentChunker(
        Settings(
            splitter={
                "provider": "recursive",
                "chunk_size": 20,
                "chunk_overlap": 0,
            }
        )
    )

    small_chunks = small.split_document(document)
    large_chunks = large.split_document(document)

    assert len(small_chunks) > len(large_chunks)
    assert max(len(item.text) for item in small_chunks) <= 8
    assert max(len(item.text) for item in large_chunks) <= 20


@pytest.mark.unit
def test_real_splitter_keeps_long_image_reference_usable() -> None:
    """长图片 ID 即使超过 chunk_size，也必须完整进入一个可分发图片的 Chunk。"""
    image_id = f"{'a' * 64}_1_1"
    placeholder = f"[IMAGE: {image_id}]"
    text = f"Text before image.\n\n{placeholder}\n\nText after image."
    document = Document(
        id="doc-real-splitter",
        text=text,
        metadata={
            "source_path": "reports/with-image.pdf",
            "images": [
                {
                    "id": image_id,
                    "path": f"data/images/{image_id}.png",
                    "text_offset": text.index(placeholder),
                    "text_length": len(placeholder),
                }
            ],
        },
    )
    chunker = DocumentChunker(
        Settings(
            splitter={
                "provider": "recursive",
                "chunk_size": 20,
                "chunk_overlap": 0,
            }
        )
    )

    chunks = chunker.split_document(document)
    image_chunks = [chunk for chunk in chunks if placeholder in chunk.text]

    assert len(image_chunks) == 1
    assert image_chunks[0].text.strip() == placeholder
    assert image_chunks[0].metadata["image_refs"] == [image_id]
    assert image_chunks[0].metadata["images"][0]["path"].endswith(".png")


@pytest.mark.unit
def test_rejects_splitter_text_that_cannot_map_back_to_document(
    fake_splitters: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无法回到原文的片段没有可信 offset，适配层必须明确拒绝。"""
    chunker = DocumentChunker(Settings(splitter={"provider": "c4-pipe"}))
    monkeypatch.setattr(chunker.splitter, "split_text", lambda text: ["invented"])

    with pytest.raises(ValueError, match="not an exact substring"):
        chunker.split_document(make_document())


@pytest.mark.unit
def test_rejects_unknown_image_reference(
    fake_splitters: None,
) -> None:
    """占位符没有对应图片路径时应在 C4 失败，不能把坏引用传到 C7。"""
    document = make_document("[IMAGE: missing]")
    chunker = DocumentChunker(Settings(splitter={"provider": "c4-pipe"}))

    with pytest.raises(ValueError, match="missing from Document.metadata.images"):
        chunker.split_document(document)
