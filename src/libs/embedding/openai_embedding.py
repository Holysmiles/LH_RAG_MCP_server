"""OpenAI Embedding provider with a small, dependency-free HTTP client.

最容易理解的调用流程是：

``EmbeddingFactory -> OpenAIEmbedding -> BaseEmbedding -> urllib.request``

输入是一组文本，输出是同样顺序的一组浮点向量。OpenAI 返回的 ``data``
可能不按输入顺序排列，因此实现会根据返回的 ``index`` 重新排序，保证上层
摄取流水线不会把向量和原文本错配。
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from json import JSONDecodeError
import math
from typing import Any, ClassVar, NoReturn, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class EmbeddingProviderError(RuntimeError):
    """表示 Embedding 服务调用失败，且不会回显密钥或原始响应。"""


__all__ = ["EmbeddingProviderError", "OpenAIEmbedding"]


class OpenAIEmbedding(BaseEmbedding):
    """调用 OpenAI ``/embeddings`` 接口，把文本批量转换成向量。"""

    provider_name: ClassVar[str] = "openai"
    default_base_url: ClassVar[str] = "https://api.openai.com/v1"

    def __init__(self, config: Mapping[str, Any]) -> None:
        # 复制配置，避免 provider 初始化时意外修改 Settings 中的原始字典。
        self.config = dict(config)
        self.model = self._require_text(config, "model")
        self.api_key = self._require_text(config, "api_key")
        self.base_url = self._resolve_base_url(config)
        self.timeout = self._resolve_timeout(config)

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """批量生成向量，并保持返回顺序与输入顺序一致。

        ``trace`` 是统一接口为后续可观测性预留的参数，本阶段不在 provider
        内写日志；这样单元测试仍然可以完全隔离网络和日志依赖。
        """
        normalized_texts = self._validate_texts(texts)
        payload = {"input": normalized_texts, "model": self.model}

        try:
            request_body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            # 当前输入已经限制为字符串，这里仍保留保护层，防止未来扩展字段时
            # 把底层 JSON 异常直接暴露给业务代码。
            raise ValueError(
                f"{self.provider_name} validation error: "
                "request payload must be JSON-serializable"
            ) from None

        request = Request(
            self._build_url(),
            data=request_body,
            headers=self._build_headers(),
            method="POST",
        )

        try:
            # timeout 防止 Embedding 服务无响应时，摄取流水线永久卡住。
            with urlopen(request, timeout=self.timeout) as response:
                response_bytes = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # 不拼接原始异常文本，避免 traceback 或代理错误意外暴露配置。
            raise EmbeddingProviderError(
                f"{self.provider_name} {type(exc).__name__}: embed request failed"
            ) from None

        try:
            response_body = response_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise EmbeddingProviderError(
                f"{self.provider_name} UnicodeDecodeError: invalid embedding response"
            ) from None

        return self._extract_vectors(response_body, expected_count=len(normalized_texts))

    def _validate_texts(self, texts: list[str]) -> list[str]:
        """校验批量输入，并明确空输入与长文本的处理策略。"""
        if not isinstance(texts, list):
            self._raise_validation("texts must be a list of strings")
        if not texts:
            self._raise_validation("texts must not be empty")

        normalized: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                self._raise_validation(f"texts[{index}] must be a string")
            if not text.strip():
                self._raise_validation(f"texts[{index}] must not be empty")

            # 本地不偷偷截断长文本，避免悄悄改变用户要表达的内容；原样交给
            # Embedding 服务，由模型自己的 token 限制返回明确错误。
            normalized.append(text)

        return normalized

    def _extract_vectors(
        self, response_body: str, *, expected_count: int
    ) -> list[list[float]]:
        """解析响应、校验向量数量，并按 index 恢复输入顺序。"""
        try:
            payload = json.loads(response_body)
        except JSONDecodeError:
            raise EmbeddingProviderError(
                f"{self.provider_name} JSONDecodeError: invalid embedding response"
            ) from None

        try:
            raw_items = payload["data"]
            if not isinstance(raw_items, list):
                raise TypeError("data must be a list")
            if len(raw_items) != expected_count:
                raise ValueError(
                    f"expected {expected_count} vectors, got {len(raw_items)}"
                )

            indexed_vectors: list[tuple[int, list[float]]] = []
            for position, item in enumerate(raw_items):
                if not isinstance(item, Mapping):
                    raise TypeError(f"data[{position}] must be a mapping")

                # OpenAI 标准响应包含 index。兼容省略 index 的 mock/网关时，
                # 暂时把返回位置当作 index；有 index 时仍严格按它排序。
                vector_index = item.get("index", position)
                if isinstance(vector_index, bool) or not isinstance(vector_index, int):
                    raise TypeError(f"data[{position}].index must be an integer")

                raw_vector = item["embedding"]
                if not isinstance(raw_vector, list) or not raw_vector:
                    raise TypeError(
                        f"data[{position}].embedding must be a non-empty list"
                    )

                vector: list[float] = []
                for value in raw_vector:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("embedding values must be numbers")
                    try:
                        numeric_value = float(value)
                    except (OverflowError, ValueError):
                        raise ValueError("embedding values must be finite numbers") from None
                    if not math.isfinite(numeric_value):
                        raise ValueError("embedding values must be finite numbers")
                    vector.append(numeric_value)
                indexed_vectors.append((vector_index, vector))

            indexes = sorted(index for index, _ in indexed_vectors)
            if indexes != list(range(expected_count)):
                raise ValueError("embedding indexes must cover every input position")

            indexed_vectors.sort(key=lambda pair: pair[0])
            return [vector for _, vector in indexed_vectors]
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            self._raise_invalid_response(exc)

    def _build_url(self) -> str:
        """返回 OpenAI 的批量 Embedding endpoint。"""
        return f"{self.base_url}/embeddings"

    def _build_headers(self) -> dict[str, str]:
        """OpenAI 使用 Bearer Token，Azure 子类会覆盖这个方法。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        value = config.get("base_url", self.default_base_url)
        return self._normalize_base_url(value)

    def _normalize_base_url(self, value: object) -> str:
        """校验绝对 HTTP(S) 地址，并清理末尾斜杠。"""
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration("base_url must be a non-empty string")

        normalized = value.strip().rstrip("/")
        try:
            parsed = urlsplit(normalized)
        except ValueError:
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
        return normalized

    def _resolve_timeout(self, config: Mapping[str, Any]) -> float:
        """读取有限且大于零的超时时间，默认 30 秒。"""
        value = config.get("timeout", 30.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._raise_configuration("timeout must be a positive number")
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            self._raise_configuration("timeout must be a positive number")
        return timeout

    def _require_text(self, config: Mapping[str, Any], field: str) -> str:
        """读取必填字符串配置，并在错误中指出具体字段。"""
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration(f"missing required field 'embedding.{field}'")
        return value.strip()

    def _raise_configuration(self, detail: str) -> NoReturn:
        raise ValueError(f"{self.provider_name} configuration error: {detail}")

    def _raise_validation(self, detail: str) -> NoReturn:
        raise ValueError(f"{self.provider_name} validation error: {detail}")

    def _raise_invalid_response(self, exc: Exception) -> NoReturn:
        raise EmbeddingProviderError(
            f"{self.provider_name} {type(exc).__name__}: invalid embedding response"
        ) from None
