"""Azure OpenAI chat completion provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import quote, urlencode

from libs.llm.openai_llm import OpenAILLM


class AzureLLM(OpenAILLM):
    """Adapt the shared chat client to Azure's URL and authentication shape."""

    provider_name: ClassVar[str] = "azure"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.deployment_name = self._require_text(config, "deployment_name")
        self.api_version = self._require_text(config, "api_version")
        super().__init__(config)

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        endpoint = config.get("azure_endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            self._raise_configuration("missing required field 'llm.azure_endpoint'")
        return self._normalize_base_url(endpoint)

    def _build_url(self) -> str:
        deployment = quote(self.deployment_name, safe="")
        query = urlencode({"api-version": self.api_version})
        return (
            f"{self.base_url}/openai/deployments/{deployment}/chat/completions"
            f"?{query}"
        )

    def _build_headers(self) -> dict[str, str]:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
