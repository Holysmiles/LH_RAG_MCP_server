"""DeepSeek's OpenAI-compatible chat completion provider."""

from __future__ import annotations

from typing import ClassVar

from libs.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """Use the shared OpenAI-compatible client with DeepSeek's API URL."""

    provider_name: ClassVar[str] = "deepseek"
    default_base_url: ClassVar[str] = "https://api.deepseek.com"
