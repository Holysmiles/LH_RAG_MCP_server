"""Contract tests for optional image captioning and its fallback boundary."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core import Chunk
from core.settings import Settings
from core.trace import TraceContext
from ingestion.transform import BaseTransform, ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, response: str = "A bar chart showing growth") -> None:
        self.response = response
        self.calls: list[tuple[str, ImageInput, TraceContext | None]] = []

    def chat_with_image(
        self, text: str, image_path: ImageInput, trace: TraceContext | None = None
    ) -> str:
        self.calls.append((text, image_path, trace))
        return self.response


class FailingVisionLLM(BaseVisionLLM):
    def chat_with_image(
        self, text: str, image_path: ImageInput, trace: TraceContext | None = None
    ) -> str:
        raise RuntimeError("vision unavailable")


def settings(*, enabled: bool) -> Settings:
    return Settings(
        vision_llm={"enabled": enabled, "provider": "azure", "model": "vision"},
    )


def make_chunk(with_images: bool = True) -> Chunk:
    metadata = {"source_path": "reports/chart.pdf"}
    if with_images:
        metadata.update(
            {
                "image_refs": ["image-a", "image-b"],
                "images": [
                    {"id": "image-a", "path": "data/image-a.png", "text_offset": 0, "text_length": 10},
                    {"id": "image-b", "path": "data/image-b.png", "text_offset": 10, "text_length": 10},
                ],
            }
        )
    return Chunk(
        id="chunk-1",
        text="Chart follows. [IMAGE: image-a] [IMAGE: image-b]",
        metadata=metadata,
        start_offset=0,
        end_offset=49,
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_captioner_implements_transform_contract_and_no_image_passthrough() -> None:
    captioner = ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM())
    assert isinstance(captioner, BaseTransform)
    output = captioner.transform([make_chunk(False)])[0]

    assert output.metadata == {"source_path": "reports/chart.pdf"}


@pytest.mark.unit
def test_disabled_mode_preserves_refs_and_marks_unprocessed() -> None:
    output = ImageCaptioner(settings(enabled=False)).transform([make_chunk()])[0]

    assert output.metadata["image_refs"] == ["image-a", "image-b"]
    assert output.metadata["has_unprocessed_images"] is True
    assert output.metadata["caption_fallback_reason"] == "vision_llm_disabled"
    assert "image_captions" not in output.metadata


@pytest.mark.unit
def test_enabled_mode_calls_vision_once_per_image_and_writes_captions() -> None:
    vision = FakeVisionLLM()
    output = ImageCaptioner(settings(enabled=True), vision_llm=vision).transform(
        [make_chunk()]
    )[0]

    assert len(vision.calls) == 2
    assert [call[1] for call in vision.calls] == ["data/image-a.png", "data/image-b.png"]
    assert output.metadata["image_captions"] == {
        "image-a": "A bar chart showing growth",
        "image-b": "A bar chart showing growth",
    }
    assert all("caption" in image for image in output.metadata["images"])
    assert output.metadata["has_unprocessed_images"] is False


@pytest.mark.unit
def test_existing_caption_is_not_called_again() -> None:
    chunk = make_chunk()
    chunk.metadata["image_captions"] = {
        "image-a": "Existing caption",
        "stale-image": "Must be removed",
    }
    vision = FakeVisionLLM()

    output = ImageCaptioner(settings(enabled=True), vision_llm=vision).transform([chunk])[0]

    assert len(vision.calls) == 1
    assert output.metadata["image_captions"] == {
        "image-a": "Existing caption",
        "image-b": "A bar chart showing growth",
    }
    assert output.metadata["images"][0]["caption"] == "Existing caption"
    assert output.metadata["has_unprocessed_images"] is False


@pytest.mark.unit
def test_provider_failure_does_not_block_and_keeps_references() -> None:
    output = ImageCaptioner(
        settings(enabled=True), vision_llm=FailingVisionLLM()
    ).transform([make_chunk()])[0]

    assert output.metadata["image_refs"] == ["image-a", "image-b"]
    assert output.metadata["has_unprocessed_images"] is True
    assert "vision unavailable" not in output.metadata["caption_fallback_reason"]
    assert output.metadata["caption_fallback_reason"] == (
        "image_caption_failed:image-a:RuntimeError,image-b:RuntimeError"
    )


@pytest.mark.unit
def test_factory_failure_is_deferred_and_trace_records_counts() -> None:
    trace = TraceContext()
    output = ImageCaptioner(
        Settings(
            vision_llm={"enabled": True, "provider": "missing", "model": "none"}
        )
    ).transform([make_chunk()], trace)

    assert output[0].metadata["has_unprocessed_images"] is True
    assert output[0].metadata["caption_fallback_reason"] == (
        "vision_llm_initialization_failed:ValueError"
    )
    assert trace.stages == [
        {
            "input_count": 1,
            "output_count": 1,
            "captioned_images": 0,
            "unprocessed_chunks": 1,
            "stage": "image_captioner",
        }
    ]


@pytest.mark.unit
def test_malformed_image_metadata_is_rejected_at_chunk_boundary() -> None:
    chunk = make_chunk()
    chunk.metadata["image_refs"] = ["missing"]
    output = ImageCaptioner(settings(enabled=False)).transform([chunk])[0]

    assert output.metadata["has_unprocessed_images"] is True
    assert output.metadata["caption_fallback_reason"] == (
        "image_captioning_failed:ValueError"
    )


@pytest.mark.unit
def test_trace_is_forwarded_to_the_vision_provider() -> None:
    trace = TraceContext()
    vision = FakeVisionLLM()

    ImageCaptioner(settings(enabled=True), vision_llm=vision).transform(
        [make_chunk()], trace
    )

    assert [call[2] for call in vision.calls] == [trace, trace]


@pytest.mark.unit
def test_non_string_image_reference_enters_safe_fallback() -> None:
    chunk = make_chunk()
    chunk.metadata["image_refs"] = ["image-a", 1]

    output = ImageCaptioner(settings(enabled=False)).transform([chunk])[0]

    assert output.metadata["has_unprocessed_images"] is True
    assert output.metadata["caption_fallback_reason"] == (
        "image_captioning_failed:TypeError"
    )
