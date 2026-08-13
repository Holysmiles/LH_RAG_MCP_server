"""Optional image caption generation for chunks produced by the PDF loader.

``Chunk[] -> inspect image_refs -> Vision LLM per image -> Chunk[]``

The loader owns image extraction and the Vision provider owns image encoding or
compression.  This transform only coordinates those contracts: it passes the
stored image path to ``BaseVisionLLM.chat_with_image`` and records the returned
text.  A disabled provider, missing image file, provider error, or empty
caption leaves ``image_refs`` intact and marks ``has_unprocessed_images`` so a
later retry can find the work without blocking ingestion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core import Chunk
from core.settings import Settings
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory

from ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class ImageCaptioner(BaseTransform):
    """Generate captions when Vision LLM is enabled, with per-image fallback.

    Inputs:
        Chunks whose metadata may contain ``image_refs`` and ``images``.  Each
        image entry must expose an ``id`` and ``path`` as emitted by PdfLoader.
        ``settings.vision_llm.enabled`` controls model use; a Vision provider
        can be injected to keep tests deterministic.
    Outputs:
        New Chunks with ``metadata.image_captions`` as ``id -> caption`` for
        successful images.  Successful image entries also receive ``caption``.
        ``has_unprocessed_images`` is true whenever any referenced image could
        not be captioned and false only when all references were processed.
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if vision_llm is not None and not isinstance(vision_llm, BaseVisionLLM):
            raise TypeError("vision_llm must implement BaseVisionLLM or be None")

        self.settings = settings
        self.enabled = self._read_enabled(settings)
        self.prompt_template = self._load_prompt(prompt_path)
        self.vision_llm = vision_llm
        self._initialization_reason: str | None = None
        if self.enabled and self.vision_llm is None:
            try:
                self.vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception as exc:
                self._initialization_reason = (
                    f"vision_llm_initialization_failed:{type(exc).__name__}"
                )

    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")

        output: list[Chunk] = []
        captioned_count = 0
        unprocessed_count = 0
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            try:
                enriched, captioned, unprocessed = self._caption_one(chunk)
            except Exception as exc:
                # Keep the original references even if malformed optional
                # metadata is encountered.  The ingestion batch can continue.
                metadata = deepcopy(chunk.metadata)
                metadata["has_unprocessed_images"] = True
                metadata["caption_fallback_reason"] = (
                    f"image_captioning_failed:{type(exc).__name__}"
                )
                enriched = self._copy_chunk(chunk, metadata)
                captioned, unprocessed = 0, 1
            output.append(enriched)
            captioned_count += captioned
            unprocessed_count += unprocessed

        if trace is not None:
            trace.record_stage(
                "image_captioner",
                {
                    "input_count": len(chunks),
                    "output_count": len(output),
                    "captioned_images": captioned_count,
                    "unprocessed_chunks": unprocessed_count,
                },
            )
        return output

    def _caption_one(self, chunk: Chunk) -> tuple[Chunk, int, int]:
        metadata = deepcopy(chunk.metadata)
        if not metadata.get("images") and not metadata.get("image_refs"):
            metadata.pop("has_unprocessed_images", None)
            metadata.pop("caption_fallback_reason", None)
            return self._copy_chunk(chunk, metadata), 0, 0
        refs = self._image_entries(metadata)
        if not refs:
            metadata.pop("has_unprocessed_images", None)
            metadata.pop("caption_fallback_reason", None)
            return self._copy_chunk(chunk, metadata), 0, 0

        captions = self._existing_captions(metadata.get("image_captions"))
        failures: list[str] = []
        captioned = 0
        for image in refs:
            image_id = image["id"]
            if image_id in captions:
                continue
            if not self.enabled:
                failures.append(image_id)
                continue
            if self.vision_llm is None:
                failures.append(image_id)
                continue
            try:
                caption = self._caption_image(image)
            except Exception as exc:
                failures.append(f"{image_id}:{type(exc).__name__}")
                continue
            captions[image_id] = caption
            image["caption"] = caption
            captioned += 1

        if captions:
            metadata["image_captions"] = captions
        if failures:
            metadata["has_unprocessed_images"] = True
            metadata["caption_fallback_reason"] = self._failure_reason(failures)
        else:
            metadata["has_unprocessed_images"] = False
            metadata.pop("caption_fallback_reason", None)
        return self._copy_chunk(chunk, metadata), captioned, int(bool(failures))

    def _caption_image(self, image: Mapping[str, Any]) -> str:
        path = image.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("image path must be a non-empty string")
        prompt = self.prompt_template.replace("{image_id}", str(image["id"]))
        response = self.vision_llm.chat_with_image(prompt, path)  # type: ignore[union-attr]
        if not isinstance(response, str) or not response.strip():
            raise ValueError("vision provider returned an empty caption")
        return response.strip()

    @staticmethod
    def _image_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        raw_images = metadata.get("images", [])
        raw_refs = metadata.get("image_refs", [])
        if raw_images is None:
            raw_images = []
        if isinstance(raw_images, (str, bytes)) or not isinstance(raw_images, Sequence):
            raise TypeError("metadata.images must be a sequence")
        by_id: dict[str, dict[str, Any]] = {}
        for item in raw_images:
            if not isinstance(item, Mapping):
                raise TypeError("metadata.images entries must be mappings")
            image_id = item.get("id")
            if isinstance(image_id, str) and image_id.strip():
                by_id.setdefault(image_id, dict(item))
        if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
            raise TypeError("metadata.image_refs must be a sequence")
        ids = [item for item in raw_refs if isinstance(item, str) and item.strip()]
        if not ids:
            ids = list(by_id)
        missing = [image_id for image_id in ids if image_id not in by_id]
        if missing:
            raise ValueError(f"image_refs missing image entries: {missing}")
        unique_ids = list(dict.fromkeys(ids))
        entries = [by_id[image_id] for image_id in unique_ids]
        metadata["images"] = entries
        metadata["image_refs"] = unique_ids
        return entries

    @staticmethod
    def _existing_captions(value: object) -> dict[str, str]:
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): item.strip()
            for key, item in value.items()
            if isinstance(key, str) and isinstance(item, str) and item.strip()
        }

    def _load_prompt(self, prompt_path: str | Path | None) -> str:
        if prompt_path is not None and not isinstance(prompt_path, (str, Path)):
            raise TypeError("prompt_path must be a string, Path, or None")
        path = Path(prompt_path) if prompt_path is not None else Path(
            "config/prompts/image_captioning.txt"
        )
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError:
            return "Describe the image referenced by {image_id} in one concise paragraph."
        return prompt if prompt and "{image_id}" in prompt else (
            "Describe the image referenced by {image_id} in one concise paragraph."
        )

    @staticmethod
    def _read_enabled(settings: Settings) -> bool:
        config = settings.vision_llm
        if not isinstance(config, Mapping):
            raise TypeError("configuration section 'vision_llm' must be a mapping")
        value = config.get("enabled", False)
        if not isinstance(value, bool):
            raise TypeError("vision_llm.enabled must be a boolean")
        return value

    def _failure_reason(self, failures: list[str]) -> str:
        if self._initialization_reason and all(":" not in item for item in failures):
            return self._initialization_reason
        if not self.enabled:
            return "vision_llm_disabled"
        return "image_caption_failed:" + ",".join(failures)

    @staticmethod
    def _copy_chunk(chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


__all__ = ["ImageCaptioner"]
