"""Verify image bytes, managed paths, and graceful multimodal fallback."""

import base64
from copy import deepcopy

from mcp import types
from PIL import Image
import pytest

from core.response.multimodal_assembler import MultimodalAssembler
from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


pytestmark = pytest.mark.unit


def result(refs, images):
    return RetrievalResult("chunk", 0.5, "Text remains available.",
                           {"source_path": "guide.pdf", "image_refs": refs, "images": images})


@pytest.mark.parametrize("format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_format_is_detected_from_bytes_and_base64_roundtrips(tmp_path, format, mime):
    path = tmp_path / "misleading-extension.bin"
    Image.new("RGB", (4, 4), "blue").save(path, format=format)
    item = result(["image-1"], [{"id": "image-1", "path": str(path)}])
    original = deepcopy(item)

    response = ResponseBuilder(MultimodalAssembler(tmp_path)).build([item], "image query")

    assert response["content"][0]["type"] == "text"
    block = response["content"][1]
    assert block["type"] == "image" and block["mimeType"] == mime
    assert base64.b64decode(block["data"], validate=True) == path.read_bytes()
    assert types.CallToolResult.model_validate(response).content[1].mime_type == mime
    assert response["structuredContent"]["citations"][0]["chunk_id"] == "chunk"
    assert item == original


def test_reference_order_and_duplicate_ids_across_chunks(tmp_path):
    images = []
    for image_id, color in (("first", "red"), ("second", "blue"), ("unused", "green")):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (2, 2), color).save(path)
        images.append({"id": image_id, "path": str(path)})
    blocks = MultimodalAssembler(tmp_path).assemble([
        result(["second", "first", "second"], images), result(["first"], images),
    ])
    assert len(blocks) == 2
    assert base64.b64decode(blocks[0]["data"]) == (tmp_path / "second.png").read_bytes()
    assert base64.b64decode(blocks[1]["data"]) == (tmp_path / "first.png").read_bytes()


def test_missing_corrupt_and_outside_images_do_not_break_text(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    corrupt = root / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    images = [{"id": "missing", "path": str(root / "missing.png")},
              {"id": "corrupt", "path": str(corrupt)},
              {"id": "outside", "path": str(outside)}]
    response = ResponseBuilder(MultimodalAssembler(root)).build([
        result(["missing", "corrupt", "outside"], images),
    ], "query")
    assert len(response["content"]) == 1
    assert "Text remains available" in response["content"][0]["text"]
    assert response["structuredContent"]["total"] == 1


def test_id_only_reference_uses_existing_image_storage_index(tmp_path):
    from ingestion.storage.image_storage import ImageStorage

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(source)
    root, database = tmp_path / "managed", tmp_path / "index.db"
    storage = ImageStorage(root, database)
    managed = storage.store("indexed-image", source, "alpha")
    assembler = MultimodalAssembler(root, image_index_db_path=database)
    blocks = assembler.assemble([result(["indexed-image"], [])])
    assert base64.b64decode(blocks[0]["data"]) == managed.read_bytes()


def test_text_only_and_unresolved_refs_do_not_create_storage(tmp_path):
    root, database = tmp_path / "managed", tmp_path / "index.db"
    assembler = MultimodalAssembler(root, image_index_db_path=database)
    assert assembler.assemble([result([], [])]) == []
    assert assembler.assemble([result(["missing"], [])]) == []
    assert not root.exists() and not database.exists()


def test_invalid_image_index_leaves_text_available(tmp_path):
    database = tmp_path / "broken.db"
    database.write_bytes(b"not sqlite")
    assembler = MultimodalAssembler(tmp_path, image_index_db_path=database)
    assert assembler.assemble([result(["missing"], [])]) == []
