"""OpenAI-compatible chat completion provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from json import JSONDecodeError
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM, ChatMessage


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider request or response cannot be processed."""


class OpenAILLM(BaseLLM):
    """Call an OpenAI-compatible ``chat/completions`` endpoint."""

    provider_name: ClassVar[str] = "openai"
    default_base_url: ClassVar[str] = "https://api.openai.com/v1"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.model = self._require_text(config, "model")
        self.api_key = self._require_text(config, "api_key")
        self.base_url = self._resolve_base_url(config)
        self.timeout = self._resolve_timeout(config)

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Validate messages, send one request, and return assistant text."""
        normalized_messages = self._validate_messages(messages)
        payload = {"model": self.model, "messages": normalized_messages}
        request = Request(
            self._build_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_bytes = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise LLMProviderError(
                f"{self.provider_name} {type(exc).__name__}: chat request failed"
            ) from exc

        try:
            response_body = response_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LLMProviderError(
                f"{self.provider_name} UnicodeDecodeError: invalid chat response"
            ) from exc

        return self._extract_content(response_body)

    def _validate_messages(
        self, messages: Sequence[ChatMessage]
    ) -> list[dict[str, Any]]:
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

            normalized.append(dict(message))

        return normalized

    def _extract_content(self, response_body: str) -> str:
        try:
            payload = json.loads(response_body)
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("assistant content must be a string")
            return content
        except JSONDecodeError as exc:
            raise LLMProviderError(
                f"{self.provider_name} JSONDecodeError: invalid chat response"
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                f"{self.provider_name} {type(exc).__name__}: invalid chat response"
            ) from exc

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        value = config.get("base_url", self.default_base_url)
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration("base_url must be a non-empty string")
        return value.rstrip("/")

    def _resolve_timeout(self, config: Mapping[str, Any]) -> float:
        value = config.get("timeout", 30.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._raise_configuration("timeout must be a positive number")
        timeout = float(value)
        if timeout <= 0:
            self._raise_configuration("timeout must be a positive number")
        return timeout

    def _build_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _require_text(self, config: Mapping[str, Any], field: str) -> str:
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration(f"missing required field 'llm.{field}'")
        return value.strip()

    def _raise_configuration(self, detail: str) -> None:
        raise ValueError(f"{self.provider_name} configuration error: {detail}")

    def _raise_validation(self, detail: str) -> None:
        raise ValueError(f"{self.provider_name} validation error: {detail}")
