"""为 PDF Loader 产出的 Chunk 可选地生成图片说明。

整体链路：``Chunk[] -> 检查 image_refs -> 逐图调用 Vision LLM -> 新 Chunk[]``。

Loader 负责抽取图片，Vision Provider 负责读取、编码和压缩图片；本模块只负责协调：
把图片路径交给 ``chat_with_image``，再把文字结果写回 metadata。模型关闭、文件缺失、
服务异常或空响应都不会丢失图片引用，而会标记 ``has_unprocessed_images``，使后续任务
能够识别并重试未完成的图片。
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
    """在 Vision LLM 启用时生成说明，并把失败隔离到单张图片。

    输入：metadata 可含 ``image_refs`` 和 ``images`` 的 Chunk；每张图片必须有
    PdfLoader 生成的 ``id`` 和 ``path``，可注入 Vision Provider 供测试使用。
    输出：同顺序的新 Chunk；成功说明同时写入 ``image_captions[id]`` 和对应图片项的
    ``caption``。只要有一张未完成，``has_unprocessed_images`` 就为真。
    失败边界：坏图片元数据使当前 Chunk 整体降级；单张图片的 Provider 异常只标记该图，
    不阻塞同一 Chunk 的其他图片，也不阻塞整个批次。
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
                enriched, captioned, unprocessed = self._caption_one(chunk, trace)
            except Exception as exc:
                # 可选图片元数据损坏时保留原引用，让摄取批次继续并留下重试线索。
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

    def _caption_one(
        self,
        chunk: Chunk,
        trace: TraceContext | None,
    ) -> tuple[Chunk, int, int]:
        metadata = deepcopy(chunk.metadata)
        if not metadata.get("images") and not metadata.get("image_refs"):
            metadata.pop("image_captions", None)
            metadata.pop("has_unprocessed_images", None)
            metadata.pop("caption_fallback_reason", None)
            return self._copy_chunk(chunk, metadata), 0, 0
        refs = self._image_entries(metadata)
        if not refs:
            metadata.pop("image_captions", None)
            metadata.pop("has_unprocessed_images", None)
            metadata.pop("caption_fallback_reason", None)
            return self._copy_chunk(chunk, metadata), 0, 0

        ref_ids = {image["id"] for image in refs}
        captions = {
            image_id: caption
            for image_id, caption in self._existing_captions(
                metadata.get("image_captions")
            ).items()
            if image_id in ref_ids
        }
        failures: list[str] = []
        captioned = 0
        for image in refs:
            image_id = image["id"]
            if image_id in captions:
                image["caption"] = captions[image_id]
                continue
            if not self.enabled:
                failures.append(image_id)
                continue
            if self.vision_llm is None:
                failures.append(image_id)
                continue
            try:
                caption = self._caption_image(image, trace)
            except Exception as exc:
                failures.append(f"{image_id}:{type(exc).__name__}")
                continue
            captions[image_id] = caption
            image["caption"] = caption
            captioned += 1

        if captions:
            metadata["image_captions"] = captions
        else:
            metadata.pop("image_captions", None)
        if failures:
            metadata["has_unprocessed_images"] = True
            metadata["caption_fallback_reason"] = self._failure_reason(failures)
        else:
            metadata["has_unprocessed_images"] = False
            metadata.pop("caption_fallback_reason", None)
        return self._copy_chunk(chunk, metadata), captioned, int(bool(failures))

    def _caption_image(
        self,
        image: Mapping[str, Any],
        trace: TraceContext | None,
    ) -> str:
        """把一张图片交给 Provider，并返回长度受控的非空说明。"""
        path = image.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("image path must be a non-empty string")
        prompt = self.prompt_template.replace("{image_id}", str(image["id"]))
        response = self.vision_llm.chat_with_image(  # type: ignore[union-attr]
            prompt,
            path,
            trace=trace,
        )
        if not isinstance(response, str) or not response.strip():
            raise ValueError("vision provider returned an empty caption")
        normalized = " ".join(response.split())
        return normalized if len(normalized) <= 1_000 else normalized[:997].rstrip() + "..."

    @staticmethod
    def _image_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        raw_images = metadata.get("images", [])
        raw_refs = metadata.get("image_refs", [])
        if raw_images is None:
            raw_images = []
        if isinstance(raw_images, (str, bytes)) or not isinstance(raw_images, Sequence):
            raise TypeError("metadata.images must be a sequence")
        by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(raw_images):
            if not isinstance(item, Mapping):
                raise TypeError("metadata.images entries must be mappings")
            image_id = item.get("id")
            if not isinstance(image_id, str):
                raise TypeError(f"metadata.images[{index}].id must be a string")
            if not image_id.strip():
                raise ValueError(f"metadata.images[{index}].id must not be empty")
            normalized_id = image_id.strip()
            normalized_item = dict(item)
            normalized_item["id"] = normalized_id
            by_id.setdefault(normalized_id, normalized_item)
        if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
            raise TypeError("metadata.image_refs must be a sequence")
        ids: list[str] = []
        for index, item in enumerate(raw_refs):
            if not isinstance(item, str):
                raise TypeError(f"metadata.image_refs[{index}] must be a string")
            if not item.strip():
                raise ValueError(f"metadata.image_refs[{index}] must not be empty")
            ids.append(item.strip())
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
        captions: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str) or not item.strip():
                continue
            normalized = " ".join(item.split())
            captions[key] = (
                normalized
                if len(normalized) <= 1_000
                else normalized[:997].rstrip() + "..."
            )
        return captions

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
