"""规则优先、LLM 可选且可降级的 Chunk 文本精炼器。

从第一性原理看，知识摄取不能把网络模型当成唯一正确路径：规则清洗是确定性的
主路径，负责删除可明确判断的噪声；LLM 只在配置启用时对清洗结果做语义重写。
任何模型初始化、网络、响应或单个 Chunk 异常都必须降级，不能丢失整批数据。

完整数据流：

``Chunk[] -> 规则去噪 -> [可选 LLM 重写] -> 新 Chunk[]``

输入 Chunk 的 ``id/start_offset/end_offset/source_ref`` 表示源文档中的身份和位置，
即使正文被清理也保持不变。输出 metadata 用 ``refined_by`` 说明最终正文来源；
LLM 失败时额外写入 ``fallback_reason``，供日志和 Dashboard 解释降级原因。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
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


class ChunkRefiner(BaseTransform):
    """先确定性去噪，再按配置选择是否调用文字 LLM。

    Args:
        settings: ``ingestion.chunk_refiner.use_llm`` 控制模型增强；LLM Provider
            仍从项目共享的 ``settings.llm`` 创建，避免维护第二套模型配置。
        llm: 测试或特殊部署可注入现成 ``BaseLLM``。未注入且启用 LLM 时通过
            ``LLMFactory`` 创建。创建失败会记为降级原因，不阻止对象初始化。
        prompt_path: 可选提示词路径；缺失、不可读或无 ``{text}`` 占位符时使用
            内置安全模板，因此规则模式不依赖工作目录中的配置文件。

    Output metadata:
        ``refined_by`` 为 ``rule``、``llm`` 或极端异常时的 ``original``；
        ``fallback_reason`` 只在期望使用 LLM/规则却失败时存在。
    """

    default_prompt_path: ClassVar[Path] = Path(
        "config/prompts/chunk_refinement.txt"
    )
    default_prompt: ClassVar[str] = (
        "Rewrite the following RAG chunk into a concise, self-contained passage. "
        "Preserve every fact, identifier, number, Markdown heading, code block, "
        "and [IMAGE: id] reference. Do not add facts or commentary. Return only "
        "the refined passage.\n\n{text}"
    )
    _html_comment: ClassVar[re.Pattern[str]] = re.compile(r"<!--.*?-->", re.DOTALL)
    _page_marker: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|"
        r"第\s*\d+\s*页(?:\s*[（(]?共\s*\d+\s*页[）)]?)?)$",
        re.IGNORECASE,
    )
    _separator: ClassVar[re.Pattern[str]] = re.compile(
        r"^[ \t]*(?:-{3,}|\*{3,}|_{3,}|={3,})[ \t]*$"
    )
    _horizontal_whitespace: ClassVar[re.Pattern[str]] = re.compile(r"[ \t]+")
    _blank_lines: ClassVar[re.Pattern[str]] = re.compile(r"\n{3,}")
    _image_placeholder: ClassVar[re.Pattern[str]] = re.compile(
        r"\[IMAGE:\s*[^\]\r\n]+?\s*\]"
    )
    _code_block: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~)",
        re.DOTALL,
    )

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
                # Provider 配置错误属于可预期外部失败。只保存异常类型，避免把
                # endpoint、key 或第三方响应文本写入 Chunk metadata。
                self._llm_initialization_reason = (
                    f"llm_initialization_failed:{type(exc).__name__}"
                )

    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """精炼整批 Chunk；单项失败时保留该项原文并继续后面的项。

        Args:
            chunks: 有序 Chunk 序列。方法把每项视为只读，不修改输入对象。
            trace: 可选追踪容器；完成后记录规则/LLM/降级数量。

        Returns:
            与输入等长、等顺序的新 Chunk 列表。ID、offset 和 source_ref 不变。
        """
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")

        refined_chunks: list[Chunk] = []
        llm_count = 0
        rule_count = 0
        fallback_count = 0
        original_count = 0

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            try:
                refined, method, fallback_reason = self._refine_one(chunk, trace)
            except Exception as exc:
                # 规则实现或坏 metadata 的单项异常不应抹掉其他 Chunk。这里仍用
                # 新对象输出，保证调用者永远不会观察到半修改状态。
                metadata = deepcopy(chunk.metadata)
                metadata["refined_by"] = "original"
                metadata["fallback_reason"] = (
                    f"chunk_refinement_failed:{type(exc).__name__}"
                )
                refined = self._copy_chunk(chunk, chunk.text, metadata)
                method = "original"
                fallback_reason = metadata["fallback_reason"]

            refined_chunks.append(refined)
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
                "chunk_refiner",
                {
                    "input_count": len(chunks),
                    "output_count": len(refined_chunks),
                    "llm_refined": llm_count,
                    "rule_refined": rule_count,
                    "original_preserved": original_count,
                    "fallback_count": fallback_count,
                },
            )
        return refined_chunks

    def _refine_one(
        self,
        chunk: Chunk,
        trace: TraceContext | None,
    ) -> tuple[Chunk, str, str | None]:
        """处理单项并返回新对象、最终方法和可选降级原因。"""
        rule_text = self._rule_based_refine(chunk.text)
        fallback_reason: str | None = None
        if not rule_text.strip():
            rule_text = chunk.text
            fallback_reason = "rule_produced_empty_text"

        final_text = rule_text
        method = "rule"
        if self.use_llm:
            llm_text = self._llm_refine(rule_text, trace)
            if llm_text is not None:
                final_text = llm_text
                method = "llm"
                fallback_reason = None
            else:
                fallback_reason = (
                    self._last_llm_failure_reason or "llm_refinement_failed"
                )

        metadata = deepcopy(chunk.metadata)
        metadata["refined_by"] = method
        if fallback_reason is None:
            metadata.pop("fallback_reason", None)
        else:
            metadata["fallback_reason"] = fallback_reason
        return self._copy_chunk(chunk, final_text, metadata), method, fallback_reason

    def _rule_based_refine(self, text: str) -> str:
        """保守删除确定性噪声，同时完整保留 fenced code block 内部格式。

        清理范围只包括 HTML 注释、控制/OCR 替换字符、独立页码、独立分隔线、
        普通文本行的重复横向空白和过多空行。不会猜测拼写、改写事实、删除
        Markdown 标题/列表/强调，也不会调整代码块中的缩进和空行。
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        protected_blocks: list[tuple[str, str]] = []

        def protect_code(match: re.Match[str]) -> str:
            token = f"C5PROTECTEDCODEBLOCK{len(protected_blocks)}TOKEN"
            # 极罕见的 token 冲突也必须可逆，不能覆盖正文中的真实字符串。
            while token in normalized or any(item[0] == token for item in protected_blocks):
                token += "X"
            protected_blocks.append((token, match.group(0)))
            return token

        protected_text = self._code_block.sub(protect_code, normalized)
        refined = self._clean_normal_text(protected_text)
        for token, code_block in protected_blocks:
            refined = refined.replace(token, code_block, 1)
        return refined

    def _clean_normal_text(self, text: str) -> str:
        """清理非代码区域；逐行处理以免破坏 Markdown 的结构边界。"""
        without_comments = self._html_comment.sub("", text)
        without_ocr_controls = (
            without_comments.replace("\ufffd", "")
            .replace("\u00ad", "")
            .replace("\x0c", "\n")
        )
        cleaned_lines: list[str] = []
        for line in without_ocr_controls.split("\n"):
            stripped = line.strip()
            if self._page_marker.fullmatch(stripped) or self._separator.fullmatch(line):
                continue
            if not stripped:
                cleaned_lines.append("")
                continue

            leading = line[: len(line) - len(line.lstrip(" \t"))]
            # 保留 Markdown 嵌套所需的最多四格缩进；普通正文不保留偶然缩进。
            normalized_leading = leading if len(leading.expandtabs(4)) <= 4 else ""
            content = self._horizontal_whitespace.sub(" ", line.lstrip()).rstrip()
            cleaned_lines.append(f"{normalized_leading}{content}")
        return self._blank_lines.sub("\n\n", "\n".join(cleaned_lines)).strip()

    def _llm_refine(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> str | None:
        """调用可选 LLM；任何失败或空响应均返回 ``None`` 触发规则回退。"""
        del trace
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
        refined = response.strip()
        required_images = self._image_placeholder.findall(text)
        if any(placeholder not in refined for placeholder in required_images):
            self._last_llm_failure_reason = "llm_dropped_image_reference"
            return None
        return refined

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        """读取带 ``{text}`` 的模板，任何文件问题都回到内置安全模板。"""
        if prompt_path is not None and not isinstance(prompt_path, (str, Path)):
            raise TypeError("prompt_path must be a string, Path, or None")
        path = Path(prompt_path) if prompt_path is not None else self.default_prompt_path
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError:
            return self.default_prompt
        if not prompt or "{text}" not in prompt:
            return self.default_prompt
        return prompt

    @staticmethod
    def _read_use_llm(settings: Settings) -> bool:
        config = settings.ingestion.get("chunk_refiner", {})
        if not isinstance(config, Mapping):
            raise TypeError("ingestion.chunk_refiner must be a mapping")
        use_llm = config.get("use_llm", False)
        if not isinstance(use_llm, bool):
            raise TypeError("ingestion.chunk_refiner.use_llm must be a boolean")
        return use_llm

    @staticmethod
    def _copy_chunk(
        chunk: Chunk,
        text: str,
        metadata: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


__all__ = ["ChunkRefiner"]
