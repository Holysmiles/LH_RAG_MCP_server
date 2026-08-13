"""C3 PDF Loader 的离线契约测试。

测试数据流是 ``真实 PDF -> PdfLoader -> Document + PNG``。测试只检查 Loader
应该负责的结果：正文、稳定文档 ID、来源路径、图片占位符与图片文件；不会在这里
测试切块、图片描述或向量化，因为那些是后续 C4/C7/C8 的职责。
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import logging
from pathlib import Path

from PIL import Image
import pytest

from core import Document
from libs.loader import BaseLoader, PdfLoader


FIXTURE_DIR = Path("tests/fixtures/sample_documents")
SIMPLE_PDF = FIXTURE_DIR / "simple.pdf"
IMAGE_PDF = FIXTURE_DIR / "with_images.pdf"


@pytest.mark.unit
def test_base_loader_is_abstract_and_pdf_loader_implements_contract(
    tmp_path: Path,
) -> None:
    """Pipeline 依赖抽象接口，PDF 默认实现必须能直接替换进来。"""
    with pytest.raises(TypeError):
        BaseLoader()

    loader = PdfLoader(image_root=tmp_path / "images")

    assert isinstance(loader, BaseLoader)


@pytest.mark.unit
def test_load_text_pdf_returns_document_with_stable_content_id(
    tmp_path: Path,
) -> None:
    """纯文本 PDF 应得到 MarkItDown 正文、来源信息和基于字节的 SHA256 ID。"""
    loader = PdfLoader(image_root=tmp_path / "images")
    expected_hash = hashlib.sha256(SIMPLE_PDF.read_bytes()).hexdigest()

    first = loader.load(SIMPLE_PDF)
    second = loader.load(SIMPLE_PDF)

    assert isinstance(first, Document)
    assert first.id == second.id == expected_hash
    assert first.text == second.text
    assert "Retrieval augmented generation" in first.text
    assert "This PDF contains text but no images." in first.text
    assert first.metadata == {
        "source_path": str(SIMPLE_PDF),
        "doc_type": "pdf",
        "images": [],
    }


@pytest.mark.unit
def test_load_image_pdf_writes_png_placeholder_and_c1_metadata(
    tmp_path: Path,
) -> None:
    """带图 PDF 的文件、占位符和 metadata 必须互相指向同一个 image_id。"""
    image_root = tmp_path / "images"
    document = PdfLoader(image_root=image_root).load(IMAGE_PDF)

    assert "Text before the embedded image" in document.text
    assert "Text after the embedded image" in document.text
    assert len(document.metadata["images"]) == 1

    image_ref = document.metadata["images"][0]
    placeholder = f"[IMAGE: {image_ref['id']}]"
    image_path = Path(image_ref["path"])
    expected_id = f"{document.id}_1_1"

    assert image_ref["id"] == expected_id
    assert image_path == image_root / document.id / f"{expected_id}.png"
    assert image_path.is_file()
    assert document.text[image_ref["text_offset"] :] == placeholder
    assert image_ref["text_length"] == len(placeholder)
    assert image_ref["page"] == 1
    assert image_ref["position"] == {
        "x": 40.0,
        "y": 100.0,
        "width": 160.0,
        "height": 80.0,
    }

    with Image.open(image_path) as extracted:
        assert extracted.format == "PNG"
        assert extracted.mode == "RGB"
        assert extracted.size == (320, 160)


@pytest.mark.unit
def test_image_extraction_failure_returns_text_and_logs_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """图片支路失败时正文仍是可用输出，并给运维留下可定位的 warning。"""
    loader = PdfLoader(image_root=tmp_path / "images")

    def fail_image_extraction(*_args: object) -> object:
        raise OSError("simulated image decoder failure")

    monkeypatch.setattr(loader, "_extract_images", fail_image_extraction)

    with caplog.at_level(logging.WARNING, logger="libs.loader.pdf_loader"):
        document = loader.load(IMAGE_PDF)

    assert "Text before the embedded image" in document.text
    assert "[IMAGE:" not in document.text
    assert document.metadata["images"] == []
    assert "PDF image extraction failed" in caplog.text
    assert "simulated image decoder failure" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path_factory", "error_type", "message"),
    [
        (lambda root: root / "missing.pdf", FileNotFoundError, "does not exist"),
        (lambda root: root, IsADirectoryError, "not a file"),
        (lambda root: root / "document.txt", ValueError, "only supports .pdf"),
    ],
)
def test_load_rejects_invalid_input_before_conversion(
    tmp_path: Path,
    path_factory: Callable[[Path], Path],
    error_type: type[Exception],
    message: str,
) -> None:
    """无效路径应在启动昂贵 PDF 解析前给出直接、可读的异常。"""
    source_path = path_factory(tmp_path)
    if source_path.suffix == ".txt":
        source_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(error_type, match=message):
        PdfLoader(image_root=tmp_path / "images").load(source_path)
