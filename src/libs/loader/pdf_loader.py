"""把 PDF 转换成 C1 ``Document`` 的默认加载器。

从第一性原理看，Loader 只需守住三条边界：

1. 输入是一个真实、可读的 ``.pdf`` 文件；
2. 输出是规范化文本以及文本引用到的图片元数据；
3. 图片属于可降级信息，图片失败不能抹掉已经解析成功的正文。

因此正文交给项目约定的 MarkItDown，页面与图片位置交给 pdfplumber。MarkItDown
不会返回 PDF 坐标和图片字节，所以本实现按页面从上到下提取图片，将其裁剪成
标准 PNG，并把占位符追加到正文末尾。``metadata.images.position`` 保留真实 PDF
坐标；不会声称占位符在 MarkItDown 正文里拥有不存在的精确字符位置映射。

完整流向如下：

``PDF -> SHA256 文档 ID -> MarkItDown 正文 -> 图片 PNG/占位符 -> Document``

本模块不切分 Chunk、不调用 Vision LLM、不生成向量，也不写向量数据库。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, ClassVar

from markitdown import MarkItDown
import pdfplumber

from core import Document
from libs.loader.base_loader import BaseLoader


logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """使用 MarkItDown 解析正文、pdfplumber 提取页面图片的 PDF Loader。

    Args:
        image_root: 图片输出根目录。默认图片路径为
            ``data/images/{完整文档 SHA256}/{image_id}.png``；测试或部署时可以
            注入其他根目录，避免写入共享运行数据。
        image_resolution: 裁剪页面图片时每英寸像素数，必须是正整数。

    ``load`` 的输出字段：

    - ``Document.id``：完整 PDF 字节的 64 位 SHA256；
    - ``Document.text``：MarkItDown 正文，成功图片会附加 ``[IMAGE: id]``；
    - ``metadata.source_path``：调用方传入路径的字符串形式；
    - ``metadata.images``：每张成功图片的 ID、PNG 路径、占位符偏移、页码和坐标。
    """

    supported_suffix: ClassVar[str] = ".pdf"
    hash_chunk_size: ClassVar[int] = 1024 * 1024

    def __init__(
        self,
        image_root: str | Path = Path("data/images"),
        *,
        image_resolution: int = 144,
    ) -> None:
        self.image_root = _normalize_output_path(image_root)
        if (
            isinstance(image_resolution, bool)
            or not isinstance(image_resolution, int)
            or image_resolution < 1
        ):
            raise ValueError("image_resolution must be a positive integer")
        self.image_resolution = image_resolution
        # 禁用外部插件：PDF 是内置转换器，Loader 不应因用户本机插件而改变结果。
        self._converter = MarkItDown(enable_plugins=False)

    def load(self, path: str | Path) -> Document:
        """读取 PDF，并返回正文、图片引用和稳定文档 ID。

        正文转换是必须成功的主路径。图片提取放在独立 ``try`` 中：任何图片格式、
        坐标或写盘异常只记录 warning，返回值仍包含已经解析出的 PDF 文本。

        Raises:
            FileNotFoundError: 输入路径不存在。
            IsADirectoryError: 输入路径不是普通文件。
            ValueError: 路径不是 PDF，或 MarkItDown 没有得到非空正文。
        """
        source_path = _validate_pdf_path(path)
        document_hash = self._compute_sha256(source_path)
        converted = self._converter.convert_local(source_path)
        text = converted.text_content
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"PDF contains no extractable text: {source_path}")

        normalized_text = text.rstrip()
        images: list[dict[str, Any]] = []
        try:
            normalized_text, images = self._extract_images(
                source_path,
                document_hash,
                normalized_text,
            )
        except Exception as exc:  # 图片是可降级支路，正文结果不能被丢弃。
            logger.warning(
                "PDF image extraction failed for %s: %s: %s",
                source_path,
                type(exc).__name__,
                exc,
            )

        return Document(
            id=document_hash,
            text=normalized_text,
            metadata={
                "source_path": str(source_path),
                "doc_type": "pdf",
                "images": images,
            },
        )

    def _extract_images(
        self,
        source_path: Path,
        document_hash: str,
        text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """按页读取图片，保存 PNG，并返回追加了占位符的新文本。

        每张图片先裁剪和写盘，成功后才把引用加入输出；因此不会产生指向不存在
        文件的 metadata。页号从 1 开始，页内序号按视觉上的从上到下、从左到右。
        """
        document_image_dir = self.image_root / document_hash
        current_text = text
        image_refs: list[dict[str, Any]] = []

        with pdfplumber.open(source_path) as pdf:
            for page in pdf.pages:
                page_images = sorted(
                    page.images,
                    key=lambda item: (
                        float(item.get("top", 0)),
                        float(item.get("x0", 0)),
                    ),
                )
                for sequence, image_data in enumerate(page_images, start=1):
                    image_id = f"{document_hash}_{page.page_number}_{sequence}"
                    image_path = document_image_dir / f"{image_id}.png"
                    self._save_image(page, image_data, image_path)

                    placeholder = f"[IMAGE: {image_id}]"
                    separator = "\n\n" if current_text else ""
                    text_offset = len(current_text) + len(separator)
                    current_text = f"{current_text}{separator}{placeholder}"
                    image_refs.append(
                        {
                            "id": image_id,
                            "path": str(image_path),
                            "text_offset": text_offset,
                            "text_length": len(placeholder),
                            "page": page.page_number,
                            "position": _image_position(image_data),
                        }
                    )

        return current_text, image_refs

    def _save_image(
        self,
        page: Any,
        image_data: dict[str, Any],
        image_path: Path,
    ) -> None:
        """按图片在页面上的真实边界裁剪，并统一编码成 RGB PNG。"""
        bbox = (
            float(image_data["x0"]),
            float(image_data["top"]),
            float(image_data["x1"]),
            float(image_data["bottom"]),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"invalid PDF image bounding box: {bbox}")

        rendered = page.crop(bbox, strict=True).to_image(
            resolution=self.image_resolution,
            antialias=True,
        ).original
        image_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.convert("RGB").save(image_path, format="PNG")

    def _compute_sha256(self, source_path: Path) -> str:
        """分块计算完整 PDF 字节哈希，避免大文件一次性进入内存。"""
        digest = hashlib.sha256()
        with source_path.open("rb") as source:
            while chunk := source.read(self.hash_chunk_size):
                digest.update(chunk)
        return digest.hexdigest()


def _validate_pdf_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path")
    if isinstance(path, str) and not path.strip():
        raise ValueError("path must not be empty")

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {source_path}")
    if not source_path.is_file():
        raise IsADirectoryError(f"PDF path is not a file: {source_path}")
    if source_path.suffix.lower() != PdfLoader.supported_suffix:
        raise ValueError(f"PdfLoader only supports .pdf files: {source_path}")
    return source_path


def _normalize_output_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("image_root must be a string or Path")
    if isinstance(path, str) and not path.strip():
        raise ValueError("image_root must not be empty")
    return Path(path)


def _image_position(image_data: dict[str, Any]) -> dict[str, float]:
    """把 pdfplumber 坐标转换为稳定、可 JSON 序列化的位置字典。"""
    x0 = float(image_data["x0"])
    top = float(image_data["top"])
    x1 = float(image_data["x1"])
    bottom = float(image_data["bottom"])
    return {
        "x": x0,
        "y": top,
        "width": x1 - x0,
        "height": bottom - top,
    }


__all__ = ["PdfLoader"]
