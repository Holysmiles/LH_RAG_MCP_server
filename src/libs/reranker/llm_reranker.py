"""LLM-based reranker with strict JSON output validation.

The data flow is deliberately small::

    candidates
      -> keep only chunk_id and text in the prompt
      -> BaseLLM.chat(...)
      -> {"ranked_ids": ["chunk-2", "chunk-1"]}
      -> return the original candidate objects in that order

The LLM never creates replacement candidate objects.  It only returns IDs, so
metadata and other fields already attached to each candidate cannot be lost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker, CandidateT

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["LLMReranker", "LLMRerankerError"]


class LLMRerankerError(RuntimeError):
    """Signal that Core should keep the pre-rerank order as a fallback.
    自定义异常 `LLMRerankerError`：抛出这个异常，上层就知道大模型重排序挂了，
    直接回退使用向量召回原来的顺序，不崩整个 RAG 流程。
    """


class LLMReranker(BaseReranker):
    """Ask a configured text LLM to order candidate IDs by relevance.

    The constructor supports two modes:

    - production: read ``config['llm']`` and build it through ``LLMFactory``;
    - tests: inject a fake ``llm`` and optional ``prompt_text`` directly.

    Dependency injection keeps tests offline while production still follows the
    same provider configuration and factory rules as the rest of the project.
    
    1. **生产环境**：不传`llm`，从 config 读取 llm 配置，调用`LLMFactory`创建真正的大模型实例；同时从磁盘读取重排序提示词文件（默认`config/prompts/rerank.txt`）。
    2. **单元测试**：直接传入伪造的`llm`对象、直接传入`prompt_text`字符串，**不用读磁盘、不用真调用网上大模型**，方便写离线单元测试。 
    > 这叫依赖注入：测试的时候塞假的 llm，生产用真实 llm，同一套代码跑两套场景。
    """

    provider_name: ClassVar[str] = "llm"
    default_prompt_path: ClassVar[str] = "config/prompts/rerank.txt"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        llm: BaseLLM | None = None,
        prompt_text: str | None = None,
    ) -> None:
        """Prepare the LLM dependency and load the reusable prompt text."""
        if not isinstance(config, Mapping):
            self._raise_configuration("configuration must be a mapping")

        # Copying protects the shared Settings object from accidental mutation.
        self.config = dict(config)
        self.llm = llm if llm is not None else self._create_llm(config)
        if not isinstance(self.llm, BaseLLM):
            self._raise_configuration("llm must implement BaseLLM")

        self.prompt_text = (
            self._validate_prompt_text(prompt_text)
            if prompt_text is not None
            else self._load_prompt(config)
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[CandidateT],
        trace: TraceContext | None = None,
    ) -> list[CandidateT]:
        """Return the same candidate objects in the order selected by the LLM.

        Think of the method as five small steps:

        1. validate the query and candidate list;
        2. build a JSON-safe prompt containing only IDs and text;
        3. ask the existing LLM provider for ranked IDs;
        4. reject malformed, missing, duplicate, or invented IDs;
        5. use the validated IDs to look up the original Python objects.
        
        1. 参数校验；候选为空直接返回空列表，**不调用大模型，省钱省网络**。
        2. `_prepare_candidates()`处理候选：
        - 一份精简版（只有 chunk_id、text）用来发给大模型；
        - 一个字典`candidate_by_id`：key=chunk_id，value = 完整原始候选对象，保存在本地内存。
        3. `_build_messages()`拼好 system+user 消息，发给大模型。
        4. 拿到大模型返回字符串，`_parse_ranked_ids()`严格校验 JSON 输出。
        5. 根据校验完的 id 列表，从`candidate_by_id`取出原始对象，按顺序返回。

        """
        del trace
        if not isinstance(query, str) or not query.strip():
            self._raise_validation("query must be a non-empty string")
        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Sequence
        ):
            self._raise_validation("candidates must be a sequence")
        if not candidates:
            # There is nothing to rank, so avoid an unnecessary paid/network call.
            return []

        prompt_candidates, candidate_by_id = self._prepare_candidates(candidates)
        messages = self._build_messages(query.strip(), prompt_candidates)

        try:
            response_text = self.llm.chat(messages)
        except Exception as exc:
            # The Core fallback only needs a stable failure signal.  Keeping the
            # original provider message out also avoids leaking URLs or secrets.
            raise LLMRerankerError(
                f"llm reranker failed: {type(exc).__name__}"
            ) from None

        ranked_ids = self._parse_ranked_ids(
            response_text,
            expected_ids=set(candidate_by_id),
        )
        return [candidate_by_id[chunk_id] for chunk_id in ranked_ids]

    def _create_llm(self, config: Mapping[str, Any]) -> BaseLLM:
        """从配置读出 llm 子配置，调用项目工厂，生成大模型实例。
        Build the nested reranker LLM through the project's normal factory."""
        llm_config = config.get("llm")
        if not isinstance(llm_config, Mapping):
            self._raise_configuration(
                "missing required mapping 'rerank.llm'"
            )
        return LLMFactory.create(Settings(llm=dict(llm_config)))

    def _load_prompt(self, config: Mapping[str, Any]) -> str:
        """读取磁盘上的提示词文本文件，初始化执行一次。文件读失败直接抛配置错误。
        Read the prompt once during construction, not once per query."""
        prompt_path_value = config.get("prompt_path", self.default_prompt_path)
        if not isinstance(prompt_path_value, str) or not prompt_path_value.strip():
            self._raise_configuration("prompt_path must be a non-empty string")

        prompt_path = Path(prompt_path_value).expanduser()
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._raise_configuration(
                f"cannot read prompt_path ({type(exc).__name__})"
            )
        return self._validate_prompt_text(prompt_text)

    def _validate_prompt_text(self, prompt_text: object) -> str:
        """Reject an empty prompt before making any model request."""
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            self._raise_configuration("prompt text must be a non-empty string")
        return prompt_text.strip()

    def _prepare_candidates(
        self,
        candidates: Sequence[CandidateT],
    ) -> tuple[list[dict[str, str]], dict[str, CandidateT]]:
        """
        输入候选列表，返回两个东西：
        1. `prompt_candidates`：只保留`chunk_id`和`text`。
        > ⚠️重点：**metadata、向量分数全部不传给大模型！只传判断相关性必须的信息，减少 token 消耗，保护内部数据。**
        2. `candidate_by_id`：内存字典，记住每个 id 对应的完整原始候选对象。
        同时做校验：不能重复 chunk_id，id 不能为空字符串。

        Extract prompt fields and remember each original candidate by ID."""
        prompt_candidates: list[dict[str, str]] = []
        candidate_by_id: dict[str, CandidateT] = {}

        for index, candidate in enumerate(candidates):
            chunk_id = self._read_candidate_field(candidate, "chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                self._raise_validation(
                    f"candidates[{index}].chunk_id must be a non-empty string"
                )
            if chunk_id in candidate_by_id:
                self._raise_validation(
                    f"candidates contains duplicate chunk_id '{chunk_id}'"
                )

            text = self._read_candidate_field(candidate, "text")
            if not isinstance(text, str):
                self._raise_validation(
                    f"candidates[{index}].text must be a string"
                )

            # Only data needed for judging relevance is sent to the LLM.  API
            # scores, internal metadata, and arbitrary objects stay local.
            prompt_candidates.append({"chunk_id": chunk_id, "text": text})
            candidate_by_id[chunk_id] = candidate

        return prompt_candidates, candidate_by_id

    @staticmethod
    def _read_candidate_field(candidate: object, field: str) -> object:
        """Support both dictionary candidates and future dataclass candidates."""
        if isinstance(candidate, Mapping):
            return candidate.get(field)
        return getattr(candidate, field, None)

    def _build_messages(
        self,
        query: str,
        candidates: list[dict[str, str]],
    ) -> list[ChatMessage]:
        """Build provider-independent system and user chat messages.
        
        system = 我们加载的rerank.txt提示词 + 强制JSON格式指令
        user = json.dumps({query:用户问题，candidates:精简候选列表})

        """
        schema_instruction = (
            'Return only one JSON object with exactly this shape: '
            '{"ranked_ids":["candidate-id", "candidate-id"]}. '
            "Include every supplied candidate ID exactly once. "
            "Do not add Markdown fences or explanation."
        )
        request_data = {
            "query": query,
            "candidates": candidates,
        }
        return [
            {
                "role": "system",
                "content": f"{self.prompt_text}\n\n{schema_instruction}",
            },
            {
                "role": "user",
                "content": json.dumps(
                    request_data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

    def _parse_ranked_ids(
        self,
        response_text: object,
        *,
        expected_ids: set[str],
    ) -> list[str]:
        """严格校验大模型输出
        Enforce the complete response schema before changing any order.
        1. 必须是字符串，且能解析为 JSON；
        2. JSON 必须是字典，且顶层只包含一个key：`ranked_ids`，不能多字段不能少；
        3. `ranked_ids`必须是字符串列表，且不能有重复；
        4. `ranked_ids`必须包含所有候选的 chunk_id，不能缺少、不能多余。
        5. 列表内部不能有重复ID
        
        """
        if not isinstance(response_text, str):
            self._raise_response("LLM response must be a string")
        try:
            payload = json.loads(response_text)
        except JSONDecodeError:
            self._raise_response("LLM response must be valid JSON")

        if not isinstance(payload, dict) or set(payload) != {"ranked_ids"}:
            self._raise_response(
                "JSON must contain exactly one field named 'ranked_ids'"
            )

        ranked_ids = payload["ranked_ids"]
        if not isinstance(ranked_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in ranked_ids
        ):
            self._raise_response("ranked_ids must be a list of strings")
        if len(ranked_ids) != len(set(ranked_ids)):
            self._raise_response("ranked_ids must not contain duplicates")

        actual_ids = set(ranked_ids)
        if actual_ids != expected_ids:
            missing_ids = sorted(expected_ids - actual_ids)
            unknown_ids = sorted(actual_ids - expected_ids)
            self._raise_response(
                "ranked_ids must contain every candidate exactly once; "
                f"missing={missing_ids}, unknown={unknown_ids}"
            )
        return ranked_ids

    @classmethod
    def _raise_configuration(cls, detail: str) -> NoReturn:
        raise ValueError(f"{cls.provider_name} configuration error: {detail}")

    @classmethod
    def _raise_validation(cls, detail: str) -> NoReturn:
        raise ValueError(f"{cls.provider_name} validation error: {detail}")

    @staticmethod
    def _raise_response(detail: str) -> NoReturn:
        raise LLMRerankerError(f"llm reranker response error: {detail}")
