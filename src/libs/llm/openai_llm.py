"""OpenAI-compatible chat completion provider.

本文件只描述 OpenAI-compatible 协议的差异；公共 HTTP 执行流程位于
``base_http_llm.py``。Azure 和 DeepSeek 再通过继承本类覆盖少量配置差异。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from libs.llm.base_http_llm import BaseHTTPChatLLM, LLMProviderError


__all__ = ["LLMProviderError", "OpenAILLM"]


class OpenAILLM(BaseHTTPChatLLM):
    """Call an OpenAI-compatible ``chat/completions`` endpoint."""

    provider_name: ClassVar[str] = "openai"
    default_base_url: ClassVar[str] = "https://api.openai.com/v1"

    def __init__(self, config: Mapping[str, Any]) -> None:
        # model、base_url、timeout 由 HTTP 基类统一读取。
        super().__init__(config)
        # OpenAI-compatible 服务还需要 Bearer Token，因此在本类补充 api_key。
        self.api_key = self._require_text(config, "api_key")

    def _extract_content(self, response_body: str) -> str:
        payload = self._decode_json_response(response_body)
        try:
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("assistant content must be a string")
            return content
        except (KeyError, IndexError, TypeError) as exc:
            self._raise_invalid_response(exc)

    def _build_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
