"""Chunk metadata enrichment for the ingestion pipeline.

First-principles contract:

``Chunk[] -> deterministic rule enrichment -> optional LLM enrichment -> Chunk[]``

The rules are the reliable baseline.  The LLM can improve wording and discover
additional tags, but it is never allowed to be the only path: unavailable
providers, malformed JSON, missing fields, and per-Chunk failures all fall
back to the rule result.  The returned Chunk is a new object, so a caller can
compare the original and enriched records without observing partial mutation.
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
    """Add ``title``, ``summary`` and ``tags`` to every Chunk.

    Inputs:
        ``Settings`` controls ``ingestion.metadata_enricher.use_llm``;
        ``Chunk`` objects must retain the core identity fields and a valid
        ``source_path`` in metadata; an optional ``BaseLLM`` is injectable for
        tests or an already managed provider.
    Outputs:
        New ``Chunk`` objects with non-empty ``title``/``summary``/``tags``.
        Existing metadata is preserved.  ``enriched_by`` is ``rule`` or
        ``llm``; when enhancement falls back, ``fallback_reason`` explains why.
    Failure boundaries:
        Provider construction, provider calls, invalid JSON, and invalid LLM
        fields never escape the transform.  Only an invalid public input type
        raises immediately, because that is a caller contract error.
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
        self._last_llm_failure_reason: str | None = None

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
        """Enrich a batch while preserving order and isolating item failures."""
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
                # A malformed individual record must not discard its siblings.
                metadata = deepcopy(chunk.metadata)
                rule_metadata = self._rule_metadata(chunk.text, metadata)
                rule_metadata["enriched_by"] = "original"
                rule_metadata["fallback_reason"] = (
                    f"metadata_enrichment_failed:{type(exc).__name__}"
                )
                enriched = self._copy_chunk(chunk, rule_metadata)
                method = "original"
                fallback_reason = rule_metadata["fallback_reason"]

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
            llm_metadata = self._llm_metadata(chunk.text)
            if llm_metadata is not None:
                metadata.update(llm_metadata)
                method = "llm"
            else:
                fallback_reason = (
                    self._last_llm_failure_reason or "llm_enrichment_failed"
                )

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
        """Build predictable metadata without network, model, or filesystem IO."""
        if not isinstance(text, str):
            raise TypeError("chunk text must be a string")
        if not isinstance(source_metadata, Mapping):
            raise TypeError("chunk metadata must be a mapping")

        metadata = deepcopy(dict(source_metadata))
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        title = self._existing_text(metadata.get("title"))
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not title:
            for line in lines:
                match = self._heading.match(line)
                if match:
                    title = self._trim(match.group(1), 120)
                    break
        if not title and lines:
            title = self._trim(self._bullet.sub("", lines[0]), 120)
        if not title:
            title = "Untitled chunk"

        summary = self._existing_text(metadata.get("summary"))
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
            summary = self._trim(self._sentence.split(summary_source)[0].strip(), 300)
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

    def _llm_metadata(self, text: str) -> dict[str, Any] | None:
        self._last_llm_failure_reason = None
        if self.llm is None:
            self._last_llm_failure_reason = (
                self._llm_initialization_reason or "llm_unavailable"
            )
            return None
        prompt = self.prompt_template.replace("{text}", text)
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            self._last_llm_failure_reason = f"llm_call_failed:{type(exc).__name__}"
            return None
        if not isinstance(response, str) or not response.strip():
            self._last_llm_failure_reason = "llm_returned_empty_text"
            return None
        try:
            payload = self._parse_json_response(response)
            title = self._require_text(payload.get("title"), "title")
            summary = self._require_text(payload.get("summary"), "summary")
            tags = self._normalize_tags(payload.get("tags"))
            if not tags:
                raise ValueError("tags must contain at least one non-empty value")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._last_llm_failure_reason = f"llm_invalid_metadata:{type(exc).__name__}"
            return None
        return {"title": title, "summary": summary, "tags": tags}

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

    @staticmethod
    def _require_text(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

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
            if isinstance(item, str) and item.strip() and item.strip() not in result:
                result.append(item.strip())
        return result[:12]

    @classmethod
    def _derive_tags(cls, text: str) -> list[str]:
        result: list[str] = []
        for token in cls._token.findall(text):
            normalized = token.lower() if token.isascii() else token
            if normalized not in result:
                result.append(normalized)
        return result[:8]

    @staticmethod
    def _existing_text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _trim(value: str, limit: int) -> str:
        value = " ".join(value.split())
        return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"

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
