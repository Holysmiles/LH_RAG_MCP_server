"""为摄取流水线补齐 Chunk 元数据。

整体链路：``Chunk[] -> 本地规则生成 -> 可选 LLM 优化 -> 新 Chunk[]``。

第一性原理上，检索需要稳定、有限且可持久化的标题、摘要和标签，因此本地规则是
永远可用的基线，LLM 只能优化结果，不能成为唯一的数据来源。模型不可用、JSON
损坏、字段缺失或单个 Chunk 处理失败时，本模块都会降级，并返回新的 Chunk，绝不
在原对象上留下只修改一半的状态。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, ClassVar

from core import Chunk
from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

from ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class MetadataEnricher(BaseTransform):
    """给每个 Chunk 添加 ``title``、``summary`` 和 ``tags``。

    输入：项目配置、按顺序排列的 Chunk；测试或托管环境可注入 ``BaseLLM``。
    输出：同顺序的新 Chunk；原 metadata 被保留，并通过 ``enriched_by`` 和
    ``fallback_reason`` 说明实际采用了 LLM、规则还是最小安全降级。
    失败边界：调用方传错公开参数类型时立即报错；模型构造、模型调用、JSON 解析
    以及单条数据处理失败只影响当前 Chunk，不会中断整个批次。
    """

    default_prompt_path: ClassVar[Path] = Path(
        "config/prompts/metadata_enrichment.txt"
    )
    default_prompt: ClassVar[str] = (
        "Return only a JSON object with string fields title and summary and "
        "an array field tags. Do not add facts that are absent from the text. "
        "Use the following RAG chunk:\n\n{text}"
    )
    _heading = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
    _bullet = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
    _sentence = re.compile(r"(?<=[.!?。！？])\s+")
    _token = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")
    _image = re.compile(r"\[IMAGE:\s*[^\]\r\n]+?\s*\]")
    _max_title_length: ClassVar[int] = 120
    _max_summary_length: ClassVar[int] = 300
    _max_tag_length: ClassVar[int] = 64

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if llm is not None and not isinstance(llm, BaseLLM):
            raise TypeError("llm must implement BaseLLM or be None")

        self.settings = settings
        self.use_llm = self._read_use_llm(settings)
        self.prompt_template = self._load_prompt(prompt_path)
        self.llm = llm
        self._llm_initialization_reason: str | None = None

        if self.use_llm and self.llm is None:
            try:
                self.llm = LLMFactory.create(settings)
            except Exception as exc:
                self._llm_initialization_reason = (
                    f"llm_initialization_failed:{type(exc).__name__}"
                )

    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """按输入顺序增强一批 Chunk，并隔离每条记录的失败。"""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")

        output: list[Chunk] = []
        rule_count = 0
        llm_count = 0
        fallback_count = 0
        original_count = 0

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            try:
                enriched, method, fallback_reason = self._enrich_one(chunk)
            except Exception as exc:
                # 即使规则代码自身异常，也要返回最小可检索元数据，不能再次调用失败规则。
                safe_metadata = self._minimal_metadata(chunk.metadata)
                safe_metadata["enriched_by"] = "original"
                safe_metadata["fallback_reason"] = (
                    f"metadata_enrichment_failed:{type(exc).__name__}"
                )
                enriched = self._copy_chunk(chunk, safe_metadata)
                method = "original"
                fallback_reason = safe_metadata["fallback_reason"]

            output.append(enriched)
            if method == "llm":
                llm_count += 1
            elif method == "rule":
                rule_count += 1
            else:
                original_count += 1
            if fallback_reason is not None:
                fallback_count += 1

        if trace is not None:
            trace.record_stage(
                "metadata_enricher",
                {
                    "input_count": len(chunks),
                    "output_count": len(output),
                    "rule_enriched": rule_count,
                    "llm_enriched": llm_count,
                    "original_preserved": original_count,
                    "fallback_count": fallback_count,
                },
            )
        return output

    def _enrich_one(self, chunk: Chunk) -> tuple[Chunk, str, str | None]:
        metadata = self._rule_metadata(chunk.text, chunk.metadata)
        fallback_reason: str | None = None
        method = "rule"

        if self.use_llm:
            llm_metadata, llm_failure_reason = self._llm_metadata(chunk.text)
            if llm_metadata is not None:
                metadata.update(llm_metadata)
                method = "llm"
            else:
                fallback_reason = llm_failure_reason or "llm_enrichment_failed"

        metadata["enriched_by"] = method
        if fallback_reason is None:
            metadata.pop("fallback_reason", None)
        else:
            metadata["fallback_reason"] = fallback_reason
        return self._copy_chunk(chunk, metadata), method, fallback_reason

    def _rule_metadata(
        self,
        text: str,
        source_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """只用本地确定性规则生成元数据，不访问网络、模型或文件系统。"""
        if not isinstance(text, str):
            raise TypeError("chunk text must be a string")
        if not isinstance(source_metadata, Mapping):
            raise TypeError("chunk metadata must be a mapping")

        metadata = deepcopy(dict(source_metadata))
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        title = self._existing_text(
            metadata.get("title"), self._max_title_length
        )
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not title:
            for line in lines:
                match = self._heading.match(line)
                if match:
                    title = self._trim(match.group(1), self._max_title_length)
                    break
        if not title and lines:
            title = self._trim(
                self._bullet.sub("", lines[0]), self._max_title_length
            )
        if not title:
            title = "Untitled chunk"

        summary = self._existing_text(
            metadata.get("summary"), self._max_summary_length
        )
        if not summary:
            summary_source = self._image.sub("", normalized)
            summary_source = re.sub(r"```.*?```|~~~.*?~~~", "", summary_source, flags=re.DOTALL)
            summary_lines = []
            for line in summary_source.splitlines():
                stripped = line.strip()
                if not stripped or self._heading.match(stripped):
                    continue
                summary_lines.append(self._bullet.sub("", stripped))
            summary_source = " ".join(summary_lines)
            summary = self._trim(
                self._sentence.split(summary_source)[0].strip(),
                self._max_summary_length,
            )
        if not summary:
            summary = title

        tags = self._normalize_tags(metadata.get("tags"))
        if not tags:
            tags = self._derive_tags(f"{title}\n{normalized}")
        if not tags:
            tags = ["untagged"]

        metadata["title"] = title
        metadata["summary"] = summary
        metadata["tags"] = tags
        return metadata

    def _llm_metadata(
        self,
        text: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """调用模型并返回 ``(有效元数据, 失败原因)``，不使用共享可变状态。"""
        if self.llm is None:
            return None, (
                self._llm_initialization_reason or "llm_unavailable"
            )
        prompt = self.prompt_template.replace("{text}", text)
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            return None, f"llm_call_failed:{type(exc).__name__}"
        if not isinstance(response, str) or not response.strip():
            return None, "llm_returned_empty_text"
        try:
            payload = self._parse_json_response(response)
            title = self._require_text(
                payload.get("title"), "title", self._max_title_length
            )
            summary = self._require_text(
                payload.get("summary"), "summary", self._max_summary_length
            )
            tags = self._normalize_tags(payload.get("tags"))
            if not tags:
                raise ValueError("tags must contain at least one non-empty value")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, f"llm_invalid_metadata:{type(exc).__name__}"
        return {"title": title, "summary": summary, "tags": tags}, None

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, Any]:
        candidate = response.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()
        payload = json.loads(candidate)
        if not isinstance(payload, Mapping):
            raise TypeError("LLM response must be a JSON object")
        if isinstance(payload.get("metadata"), Mapping):
            payload = payload["metadata"]
        return dict(payload)

    @classmethod
    def _require_text(cls, value: object, field: str, limit: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return cls._trim(value, limit)

    @classmethod
    def _normalize_tags(cls, value: object) -> list[str]:
        if isinstance(value, str):
            values = re.split(r"[,;]", value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            return []
        result: list[str] = []
        for item in values:
            if isinstance(item, str) and item.strip():
                normalized = cls._trim(item, cls._max_tag_length)
                if normalized not in result:
                    result.append(normalized)
        return result[:12]

    @classmethod
    def _derive_tags(cls, text: str) -> list[str]:
        result: list[str] = []
        for token in cls._token.findall(text):
            normalized = token.lower() if token.isascii() else token
            if normalized not in result:
                result.append(normalized)
        return result[:8]

    @classmethod
    def _existing_text(cls, value: object, limit: int) -> str | None:
        return cls._trim(value, limit) if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _trim(value: str, limit: int) -> str:
        value = " ".join(value.split())
        return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."

    @classmethod
    def _minimal_metadata(cls, source_metadata: Mapping[str, Any]) -> dict[str, Any]:
        """在所有增强逻辑都失败时生成有界、非空且可序列化的最后防线。"""
        metadata = deepcopy(dict(source_metadata))
        title = cls._existing_text(
            metadata.get("title"), cls._max_title_length
        ) or "Untitled chunk"
        summary = cls._existing_text(
            metadata.get("summary"), cls._max_summary_length
        ) or title
        tags = cls._normalize_tags(metadata.get("tags")) or ["untagged"]
        metadata.update({"title": title, "summary": summary, "tags": tags})
        return metadata

    def _load_prompt(self, prompt_path: str | Path | None) -> str:
        if prompt_path is not None and not isinstance(prompt_path, (str, Path)):
            raise TypeError("prompt_path must be a string, Path, or None")
        path = Path(prompt_path) if prompt_path is not None else self.default_prompt_path
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError:
            return self.default_prompt
        return prompt if prompt and "{text}" in prompt else self.default_prompt

    @staticmethod
    def _read_use_llm(settings: Settings) -> bool:
        config = settings.ingestion.get("metadata_enricher", {})
        if not isinstance(config, Mapping):
            raise TypeError("ingestion.metadata_enricher must be a mapping")
        value = config.get("use_llm", False)
        if not isinstance(value, bool):
            raise TypeError("ingestion.metadata_enricher.use_llm must be a boolean")
        return value

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


__all__ = ["MetadataEnricher"]
