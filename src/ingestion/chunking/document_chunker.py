"""把纯文本切分结果转换为可追溯的业务 ``Chunk``。

整体架构中，Loader 和 Splitter 故意保持简单：

``PdfLoader: PDF -> Document``
``BaseSplitter: str -> list[str]``

两者之间仍缺少父文档关系、原文位置、稳定 ID 和图片路径。``DocumentChunker``
就是这层适配器：输入完整 ``Document``，调用配置选中的 Splitter，再输出后续
Transform、Encoder 和 Storage 可以直接使用的 ``list[Chunk]``。

本层不重新发明切分算法，也不读取 PDF。它只补齐业务链路不可缺少的信息。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any, ClassVar

from core import Chunk, Document
from core.settings import Settings
from libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """将 ``Document`` 适配为带 ID、定位和图片引用的 ``Chunk`` 列表。

    Inputs:
        settings: 至少包含 ``splitter.provider`` 的项目配置；具体 ``chunk_size``
            和 ``chunk_overlap`` 原样交给 ``SplitterFactory`` 创建的实现。

    Output:
        ``split_document`` 返回保持原文顺序的 ``list[Chunk]``。每个 Chunk 都
        包含父文档引用、原文字符范围、序号和仅属于自己的图片引用。
    """

    _image_placeholder: ClassVar[re.Pattern[str]] = re.compile(
        r"\[IMAGE:\s*([^\]\r\n]+?)\s*\]"
    )

    def __init__(self, settings: Settings) -> None:
        """根据配置创建一次 Splitter，后续文档复用同一无状态实现。"""
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        self.splitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        """切分一个文档，并把纯字符串片段升级为完整业务对象。

        数据转换顺序：

        1. ``document.text`` 交给配置选中的纯文本 Splitter；
        2. 在父文档中定位每段原文，产生左闭右开 offset；
        3. 生成 ``父ID_序号_文本hash`` 格式的稳定 Chunk ID；
        4. 复制公共 metadata，只给实际含占位符的 Chunk 分发图片引用；
        5. 设置 ``source_ref=document.id``，供检索结果回溯源文档。

        如果 Splitter 返回的文本无法在父文档中找到，本方法直接报错。offset 是
        溯源事实，宁可暴露不兼容的 Splitter，也不能填写一个看似可用的假位置。
        """
        if not isinstance(document, Document):
            raise TypeError("document must be a Document instance")

        split_texts = self.splitter.split_text(document.text)
        if not isinstance(split_texts, list):
            raise TypeError("splitter must return list[str]")

        chunks: list[Chunk] = []
        previous_start = -1
        for index, chunk_text in enumerate(split_texts):
            if not isinstance(chunk_text, str):
                raise TypeError(f"splitter chunk at index {index} must be a string")
            if not chunk_text.strip():
                raise ValueError(f"splitter chunk at index {index} must not be empty")

            start_offset = document.text.find(chunk_text, previous_start + 1)
            if start_offset < 0:
                raise ValueError(
                    f"splitter chunk at index {index} is not an exact substring "
                    "of Document.text in source order"
                )
            end_offset = start_offset + len(chunk_text)
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, chunk_text),
                    text=chunk_text,
                    metadata=self._inherit_metadata(document, index, chunk_text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
            previous_start = start_offset

        return chunks

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        """生成确定性 ID；序号保证同文档重复片段仍然互不冲突。"""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
    ) -> dict[str, Any]:
        """复制公共字段，并按占位符给当前 Chunk 分发图片引用。

        ``Document.metadata.images`` 是整篇文档的图片目录。Chunk 不能整体继承
        它，否则下游会误以为每段文字都引用所有图片；也不能全部删除，否则 C7
        只有 image ID 却找不到文件路径。因此这里先移除文档级列表，再按当前
        ``chunk_text`` 中出现的 image ID 选择子集。
        """
        metadata = deepcopy(document.metadata)
        document_images = metadata.pop("images", [])
        metadata.pop("image_refs", None)
        metadata["chunk_index"] = chunk_index

        referenced_ids = [
            match.group(1).strip()
            for match in self._image_placeholder.finditer(chunk_text)
        ]
        if not referenced_ids:
            return metadata

        images_by_id = {image["id"]: image for image in document_images}
        unknown_ids = [
            image_id for image_id in referenced_ids if image_id not in images_by_id
        ]
        if unknown_ids:
            unknown = ", ".join(dict.fromkeys(unknown_ids))
            raise ValueError(
                f"chunk references image IDs missing from Document.metadata.images: "
                f"{unknown}"
            )

        # 相同占位符重复出现时只处理一次图片，同时保持首次出现的正文顺序。
        unique_ids = list(dict.fromkeys(referenced_ids))
        metadata["images"] = [deepcopy(images_by_id[item]) for item in unique_ids]
        metadata["image_refs"] = unique_ids
        return metadata


__all__ = ["DocumentChunker"]
