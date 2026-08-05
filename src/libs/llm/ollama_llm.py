"""Ollama 本地聊天模型实现。

Ollama 在本机启动一个 HTTP 服务，默认地址为 ``http://localhost:11434``。
调用链如下：

``LLMFactory -> OllamaLLM -> BaseHTTPChatLLM -> urllib.request``

- ``LLMFactory`` 根据 ``settings.llm.provider`` 选择实现；
- ``OllamaLLM`` 描述 Ollama 独有的 URL、请求参数和响应结构；
- ``BaseHTTPChatLLM`` 负责公共校验、网络调用和异常保护；
- ``urllib.request`` 是 Python 标准库，真正向 Ollama 发送 HTTP 请求。

这样业务层始终只调用 ``chat(messages)``，无需知道 Ollama 的 API 细节。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from libs.llm.base_http_llm import BaseHTTPChatLLM
from libs.llm.base_llm import ChatMessage


class OllamaLLM(BaseHTTPChatLLM):
    """通过 Ollama 原生 ``/api/chat`` 接口调用本地模型。"""

    provider_name: ClassVar[str] = "ollama"
    default_base_url: ClassVar[str] = "http://localhost:11434"

    def _build_url(self) -> str:
        """把服务根地址与 Ollama 聊天接口路径拼接起来。"""
        return f"{self.base_url}/api/chat"

    def _build_headers(self) -> dict[str, str]:
        """Ollama 默认无需 API Key，只需声明请求正文是 JSON。"""
        return {"Content-Type": "application/json"}

    def _build_payload(
        self, messages: Sequence[ChatMessage]
    ) -> dict[str, Any]:
        """构建 Ollama 请求，并关闭流式响应以保持 ``chat() -> str`` 契约。"""
        payload = super()._build_payload(messages)

        # Ollama 默认可能返回多段 NDJSON。设置 stream=False 后只返回一个完整
        # JSON 对象，基类才能一次读取并交给 _extract_content 解析。
        payload["stream"] = False
        return payload

    def _extract_content(self, response_body: str) -> str:
        """从 ``{"message": {"content": "..."}}`` 中读取回答。"""
        payload = self._decode_json_response(response_body)
        try:
            content = payload["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("assistant content must be a string")
            return content
        except (KeyError, TypeError) as exc:
            self._raise_invalid_response(exc)
