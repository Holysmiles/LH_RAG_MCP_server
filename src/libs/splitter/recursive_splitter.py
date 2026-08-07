"""LangChain-based recursive splitter for Markdown documents.

The dependency chain is deliberately simple::

    Settings
      -> SplitterFactory
      -> RecursiveSplitter
      -> LangChain RecursiveCharacterTextSplitter
      -> list[str]

LangChain performs the actual recursive splitting.  This wrapper only does
three project-specific jobs: read configuration, validate the public input,
and protect Markdown structures that must stay whole.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING, ClassVar

from langchain_text_splitters import Language

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]


from libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["RecursiveSplitter"]


class RecursiveSplitter(BaseSplitter):
    """Adapt LangChain's Markdown splitter to the project's ``BaseSplitter``.

    Normal Markdown is split by LangChain using Markdown-aware separators,
    including headings, horizontal rules, paragraphs, lines, spaces, and
    finally characters.  Fenced code blocks are extracted first because a
    syntactically complete example is more useful for retrieval than two
    fragments with missing fences or indentation.
    """

    provider_name: ClassVar[str] = "recursive"
    default_chunk_size: ClassVar[int] = 1000
    default_chunk_overlap: ClassVar[int] = 200
    _heading_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^[ \t]{0,3}#{1,6}[ \t]+.*(?:\r?\n)?$"
    )

    def __init__(self, config: Mapping[str, object]) -> None:
        """Validate settings and create the underlying LangChain splitter."""
        if not isinstance(config, Mapping):
            self._raise_configuration("configuration must be a mapping")

        # The factory already passes a copy, but keeping our own copy makes the
        # provider safe even when somebody constructs it directly in a test.
        self.config = dict(config)
        self.chunk_size = self._read_positive_int(
            config, "chunk_size", self.default_chunk_size
        )

        # Production uses 1000/200.  For a deliberately tiny chunk_size, use a
        # smaller implicit overlap so ``RecursiveSplitter({'chunk_size': 20})``
        # remains a valid and beginner-friendly configuration.
        default_overlap = min(self.default_chunk_overlap, self.chunk_size // 5)
        self.chunk_overlap = self._read_non_negative_int(
            config, "chunk_overlap", default_overlap
        )
        if self.chunk_overlap >= self.chunk_size:
            self._raise_configuration(
                "chunk_overlap must be smaller than chunk_size"
            )

        # from_language selects LangChain's Markdown separator hierarchy and
        # enables regex separators.  keep_separator='start' keeps a Markdown
        # heading with the content that follows it whenever the size permits.
        self._langchain_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            keep_separator="start",
        )

    def split_text(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> list[str]:
        """Split one Markdown document into ordered retrieval chunks.

        Think of this method as a three-step pipeline:

        1. reject invalid input before doing any work;
        2. temporarily separate protected Markdown structures;
        3. send normal text to LangChain and restore the protected structures.

        ``trace`` belongs to the shared interface.  A later pipeline stage is
        responsible for recording traces, so this library component does not
        create a second logging system.
        """
        del trace
        if not isinstance(text, str):
            self._raise_validation("text must be a string")
        if not text.strip():
            return []

        chunks: list[str] = []
        for segment, is_protected in self._extract_segments(text):
            if not segment.strip():
                continue
            if is_protected:
                chunks.append(segment)
            else:
                chunks.extend(self._langchain_splitter.split_text(segment))

        return [chunk for chunk in chunks if chunk.strip()]

    def _extract_segments(self, text: str) -> list[tuple[str, bool]]:
        """Return normal/protected segments without changing their order.

        Protected segments are:

        - fenced code blocks using three or more backticks or tildes;
        - an unusually long ATX heading that exceeds ``chunk_size`` by itself.

        Ordinary headings stay in normal text so LangChain can keep them with
        the following paragraph.  Only overlong headings need special handling
        to satisfy the rule that a heading must not be split into random words.
        """
        lines = text.splitlines(keepends=True)
        segments: list[tuple[str, bool]] = []
        normal_lines: list[str] = []
        code_lines: list[str] = []
        fence: tuple[str, int] | None = None

        def flush_normal() -> None:
            if normal_lines:
                segments.append(("".join(normal_lines), False))
                normal_lines.clear()

        for line in lines:
            if fence is not None:
                code_lines.append(line)
                if self._is_closing_fence(line, fence):
                    segments.append(("".join(code_lines), True))
                    code_lines.clear()
                    fence = None
                continue

            opening_fence = self._read_opening_fence(line)
            if opening_fence is not None:
                flush_normal()
                fence = opening_fence
                code_lines.append(line)
                continue

            is_overlong_heading = (
                self._is_heading_line(line)
                and len(line.rstrip("\r\n")) > self.chunk_size
            )
            if is_overlong_heading:
                flush_normal()
                segments.append((line, True))
                continue

            normal_lines.append(line)

        if code_lines:
            # An unfinished fence is still useful source text.  Preserve the
            # remainder as one chunk instead of silently dropping it.
            segments.append(("".join(code_lines), True))
        flush_normal()
        return segments

    @staticmethod
    def _read_opening_fence(line: str) -> tuple[str, int] | None:
        """Read a Markdown fence only when it begins within three spaces."""
        leading_spaces = len(line) - len(line.lstrip(" "))
        if leading_spaces > 3:
            return None

        candidate = line[leading_spaces:]
        if not candidate or candidate[0] not in "`~":
            return None

        marker = candidate[0]
        marker_length = 0
        while marker_length < len(candidate) and candidate[marker_length] == marker:
            marker_length += 1
        if marker_length < 3:
            return None
        return marker, marker_length

    @staticmethod
    def _is_closing_fence(line: str, fence: tuple[str, int]) -> bool:
        """Accept a matching closing fence plus optional trailing whitespace."""
        marker, minimum_length = fence
        leading_spaces = len(line) - len(line.lstrip(" "))
        if leading_spaces > 3:
            return False

        candidate = line[leading_spaces:].rstrip("\r\n")
        marker_length = 0
        while marker_length < len(candidate) and candidate[marker_length] == marker:
            marker_length += 1
        if marker_length < minimum_length:
            return False
        return not candidate[marker_length:].strip()

    @classmethod
    def _is_heading_line(cls, text: str) -> bool:
        """Return whether ``text`` is one complete ATX Markdown heading."""
        return bool(cls._heading_pattern.fullmatch(text))

    @classmethod
    def _read_positive_int(
        cls,
        config: Mapping[str, object],
        field: str,
        default: int,
    ) -> int:
        value = config.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            cls._raise_configuration(f"{field} must be a positive integer")
        return value

    @classmethod
    def _read_non_negative_int(
        cls,
        config: Mapping[str, object],
        field: str,
        default: int,
    ) -> int:
        value = config.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            cls._raise_configuration(
                f"{field} must be a non-negative integer"
            )
        return value

    @classmethod
    def _raise_configuration(cls, detail: str) -> None:
        raise ValueError(f"{cls.provider_name} configuration error: {detail}")

    @classmethod
    def _raise_validation(cls, detail: str) -> None:
        raise ValueError(f"{cls.provider_name} validation error: {detail}")
