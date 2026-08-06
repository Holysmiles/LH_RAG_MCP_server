"""Azure OpenAI Embedding provider。

Azure 的 Embedding 请求体和 OpenAI 基本一致，所以本类只覆盖三个差异：
deployment URL、`api-version` 查询参数和 `api-key` 请求头。批量校验、响应
排序、超时和异常处理全部复用 ``OpenAIEmbedding``，避免两套代码逐渐产生差异。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import quote, urlencode

from libs.embedding.openai_embedding import (
    EmbeddingProviderError,
    OpenAIEmbedding,
)


__all__ = ["AzureEmbedding", "EmbeddingProviderError"]


class AzureEmbedding(OpenAIEmbedding):
    """通过 Azure OpenAI deployment 调用 Embedding 接口。"""

    provider_name: ClassVar[str] = "azure"

    def __init__(self, config: Mapping[str, Any]) -> None:
        # 先读取 Azure 独有字段，再调用父类初始化 model/密钥/超时等公共配置。
        self.deployment_name = self._require_text(config, "deployment_name")
        self.api_version = self._require_text(config, "api_version")
        super().__init__(config)

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        """Azure 使用 azure_endpoint，而不是 OpenAI 的 base_url。"""
        endpoint = config.get("azure_endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            self._raise_configuration(
                "missing required field 'embedding.azure_endpoint'"
            )
        return self._normalize_base_url(endpoint)

    def _build_url(self) -> str:
        """拼出 Azure deployment 专用的 Embedding endpoint。"""
        deployment = quote(self.deployment_name, safe="")
        query = urlencode({"api-version": self.api_version})
        return (
            f"{self.base_url}/openai/deployments/{deployment}/embeddings"
            f"?{query}"
        )

    def _build_headers(self) -> dict[str, str]:
        """Azure 使用 api-key 请求头，不使用 OpenAI 的 Authorization。"""
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
