"""HTTP JSON 聊天模型的公共执行骨架。

这个模块只处理不同 HTTP LLM 都会重复做的事情：

1. 校验 ``messages`` 的基础结构；
2. 把 Python 字典编码成 JSON 请求；
3. 使用标准库发送 HTTP POST 请求；
4. 把网络错误包装成统一、且不会泄露配置的异常；
5. 把响应字节解码成 JSON。

具体服务的 URL、请求头和响应字段并不相同，因此仍由子类负责。例如：

``OpenAILLM`` 读取 ``choices[0].message.content``，而 ``OllamaLLM``
读取 ``message.content``。公共骨架只复用真正相同的流程，避免为了复用而
把不同协议强行写进同一个大函数。

依赖关系：``BaseHTTPChatLLM -> BaseLLM``。业务代码不会直接创建本类，而是
通过 ``LLMFactory`` 创建 OpenAI、Azure、DeepSeek 或 Ollama 的具体实现。
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
import json
from json import JSONDecodeError
import math
from typing import Any, ClassVar, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM, ChatMessage


class LLMProviderError(RuntimeError):
    """表示模型服务调用失败，但隐藏 URL、密钥等敏感配置。"""


class BaseHTTPChatLLM(BaseLLM):
    """为基于 HTTP + JSON 的聊天模型提供模板方法。

    子类只需要回答四个协议问题：请求发到哪里、请求头是什么、请求体有哪些
    特殊字段、响应正文中的最终文本在哪里。其余重复且容易出错的网络流程由
    这个基类集中维护。
    """

    provider_name: ClassVar[str] = "http_llm"
    default_base_url: ClassVar[str] = ""

    def __init__(self, config: Mapping[str, Any]) -> None:
        # 复制配置，避免 provider 后续修改配置时污染全局 Settings。
        self.config = dict(config)
        self.model = self._require_text(config, "model")
        self.base_url = self._resolve_base_url(config)
        self.timeout = self._resolve_timeout(config)

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """执行一次非流式聊天请求并返回纯文本答案。

        可以把这个方法理解成一条固定流水线：

        ``校验输入 -> 构建 JSON -> HTTP POST -> 解码 UTF-8 -> 提取文本``。

        子类通过几个小方法提供协议差异，因此这里不需要用大量 ``if provider``
        分支判断当前服务是谁。
        """
        normalized_messages = self._validate_messages(messages)
        payload = self._build_payload(normalized_messages)

        try:
            request_body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as exc:
            # message 可能携带 provider 支持的额外字段。若这些字段包含 object()
            # 等不能转换成 JSON 的值，应在联网前给调用者一个清晰错误。
            raise ValueError(
                f"{self.provider_name} validation error: "
                "request payload must be JSON-serializable"
            ) from None

        # Request 只保存请求信息，此时还没有真正访问网络。
        request = Request(
            self._build_url(),
            data=request_body,
            headers=self._build_headers(),
            method="POST",
        )

        try:
            # urlopen 进入这里才真正访问模型服务。timeout 防止服务不可用时
            # 程序无限等待。
            with urlopen(request, timeout=self.timeout) as response:
                response_bytes = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # 不拼接原始异常文本，因为其中可能带 URL、Token 或代理信息。
            raise LLMProviderError(
                f"{self.provider_name} {type(exc).__name__}: chat request failed"
            ) from None

        try:
            response_body = response_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LLMProviderError(
                f"{self.provider_name} UnicodeDecodeError: invalid chat response"
            ) from None

        return self._extract_content(response_body)

    def _build_payload(
        self, messages: Sequence[ChatMessage]
    ) -> dict[str, Any]:
        """构建大多数聊天 API 共有的 model/messages 请求主体。"""
        return {"model": self.model, "messages": list(messages)}

    @abstractmethod
    def _build_url(self) -> str:
        """返回当前 provider 的完整聊天接口 URL。"""
        raise NotImplementedError

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """返回当前 provider 所需的 HTTP 请求头。"""
        raise NotImplementedError

    @abstractmethod
    def _extract_content(self, response_body: str) -> str:
        """从 provider 的 JSON 响应中取出最终回答文本。"""
        raise NotImplementedError

    def _validate_messages(
        self, messages: Sequence[ChatMessage]
    ) -> list[dict[str, Any]]:
        """把公共消息格式校验一次，避免把错误请求发送给模型服务。"""
        # str 本身也是 Sequence，但 "hello" 显然不是消息列表，所以要单独排除。
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            self._raise_validation("messages must be a non-string sequence")
        if not messages:
            self._raise_validation("messages must not be empty")

        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                self._raise_validation(f"messages[{index}] must be a mapping")

            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                self._raise_validation(
                    f"messages[{index}].role must be a non-empty string"
                )

            content = message.get("content")
            if not isinstance(content, str):
                self._raise_validation(
                    f"messages[{index}].content must be a string"
                )

            # 使用浅拷贝保留 name、tool_call_id 等可选字段，同时不修改调用者数据。
            normalized.append(dict(message))

        return normalized

    def _decode_json_response(self, response_body: str) -> Any:
        """解析 JSON；失败时补充 provider 名称并隐藏原始响应内容。"""
        try:
            return json.loads(response_body)
        except JSONDecodeError as exc:
            raise LLMProviderError(
                f"{self.provider_name} JSONDecodeError: invalid chat response"
            ) from None

    def _raise_invalid_response(self, exc: Exception) -> NoReturn:
        """把字段缺失或类型错误转换为统一、可读的响应异常。"""
        raise LLMProviderError(
            f"{self.provider_name} {type(exc).__name__}: invalid chat response"
        ) from None

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        """读取自定义地址；未配置时使用子类声明的默认地址。"""
        value = config.get("base_url", self.default_base_url)
        return self._normalize_base_url(value)

    def _normalize_base_url(self, value: object) -> str:
        """校验 HTTP(S) 根地址并移除末尾斜杠。"""
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration("base_url must be a non-empty string")

        # 配置文件中常见的缩进空格不应进入真正的 URL。
        normalized = value.strip().rstrip("/")
        try:
            parsed = urlsplit(normalized)
        except ValueError:
            # urlsplit 对非法 IPv6 等地址会直接抛 ValueError；转成统一配置错误，
            # 避免调用方看到与 provider 无关的底层实现细节。
            self._raise_configuration(
                "base_url must be an absolute http or https URL"
            )
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            self._raise_configuration(
                "base_url must be an absolute http or https URL"
            )
        if parsed.query or parsed.fragment:
            self._raise_configuration(
                "base_url must not contain a query or fragment"
            )

        # 去掉末尾斜杠，防止拼接接口路径时出现 ``//api/chat``。
        return normalized

    def _resolve_timeout(self, config: Mapping[str, Any]) -> float:
        """读取正数超时时间，默认 30 秒。"""
        value = config.get("timeout", 30.0)
        # bool 是 int 的子类，因此必须先排除 True/False。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._raise_configuration("timeout must be a positive number")
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            self._raise_configuration("timeout must be a positive number")
        return timeout

    def _require_text(self, config: Mapping[str, Any], field: str) -> str:
        """读取必填文本配置，并给出带字段路径的错误提示。"""
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration(f"missing required field 'llm.{field}'")
        return value.strip()

    def _raise_configuration(self, detail: str) -> NoReturn:
        raise ValueError(f"{self.provider_name} configuration error: {detail}")

    def _raise_validation(self, detail: str) -> NoReturn:
        raise ValueError(f"{self.provider_name} validation error: {detail}")
