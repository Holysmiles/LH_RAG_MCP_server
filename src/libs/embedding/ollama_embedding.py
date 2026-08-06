"""Ollama 本地 Embedding Provider。

Ollama 的当前 Embedding 接口是 ``POST /api/embed``。它可以一次接收一个
``input`` 字符串数组，并在 ``embeddings`` 字段中按输入顺序返回向量。因此
本模块不需要依赖第三方 SDK，调用链很直观：

``EmbeddingFactory -> OllamaEmbedding -> urllib.request -> Ollama /api/embed``

这里的实现刻意与 OpenAI 的响应格式分开处理：OpenAI 返回 ``data`` 列表，
而 Ollama 返回 ``embeddings`` 列表。两者最后都遵守 ``BaseEmbedding.embed``
的统一契约，即输入几段文本，就返回同样数量的浮点向量。
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
from libs.embedding.openai_embedding import EmbeddingProviderError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["EmbeddingProviderError", "OllamaEmbedding"]


class OllamaEmbedding(BaseEmbedding):
    """通过 Ollama 原生 ``/api/embed`` 接口批量生成文本向量。

    Ollama 通常运行在本机，所以不需要 API Key。服务地址和模型名称仍然
    由配置提供，方便把本地服务换成远程 Ollama 实例，或切换到
    ``nomic-embed-text``、``mxbai-embed-large`` 等不同模型。
    """

    provider_name: ClassVar[str] = "ollama"
    default_base_url: ClassVar[str] = "http://localhost:11434"

    def __init__(self, config: Mapping[str, Any]) -> None:
        """读取并校验配置，但不要在初始化阶段访问网络。

        延迟到 ``embed`` 才发请求，可以让工厂初始化、配置检查和单元测试
        在没有运行 Ollama 服务时也能正常进行。
        """
        if not isinstance(config, Mapping):
            self._raise_configuration("configuration must be a mapping")

        # 复制一份配置，避免调用者之后修改字典而悄悄改变 Provider 行为。
        self.config = dict(config)
        self.model = self._require_text(config, "model")
        self.base_url = self._resolve_base_url(config)
        self.timeout = self._resolve_timeout(config)

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """把一批文本发送给 Ollama，并按输入顺序返回向量。

        具体流程可以按四步理解：

        1. 先检查输入，发现空列表、空文本或错误类型时立即报错；
        2. 构造 ``{"model": ..., "input": [...]}`` JSON 请求体；
        3. 使用标准库发起一次非流式 HTTP POST 请求；
        4. 解析 ``embeddings``，检查数量和每个数字，再返回浮点列表。

        ``trace`` 是统一接口为后续链路追踪预留的参数，当前 Provider 不在
        这里写日志，因此安全地忽略它。长文本也不会被偷偷截断，模型的
        token 限制由 Ollama 服务返回清晰错误。
        """
        del trace  # 当前阶段没有本地 tracing 实现，但保留统一接口参数。
        normalized_texts = self._validate_texts(texts)

        payload = {"model": self.model, "input": normalized_texts}
        try:
            request_body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            # 目前 payload 只有字符串；保留这个保护层，防止未来扩展字段时
            # 把底层 JSON 异常直接暴露给调用方。
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
            # timeout 防止 Ollama 未启动或模型加载很慢时，摄取流程永久等待。
            with urlopen(request, timeout=self.timeout) as response:
                response_bytes = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # 不把原始异常文本带回去，因为其中可能包含私有 URL、代理信息等。
            raise EmbeddingProviderError(
                f"{self.provider_name} {type(exc).__name__}: embed request failed"
            ) from None

        try:
            response_body = response_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise EmbeddingProviderError(
                f"{self.provider_name} UnicodeDecodeError: "
                "invalid embedding response"
            ) from None

        return self._extract_vectors(
            response_body, expected_count=len(normalized_texts)
        )

    def _validate_texts(self, texts: list[str]) -> list[str]:
        """检查批量文本的基本形状，并保留原文内容。

        这里不做本地截断或自动拆分，因为 Provider 不知道所选模型的真实
        token 上限。原文交给 Ollama 后，由服务返回模型可理解的错误信息。
        """
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
            normalized.append(text)

        return normalized

    def _extract_vectors(
        self, response_body: str, *, expected_count: int
    ) -> list[list[float]]:
        """解析 Ollama 响应并把值校验成安全的浮点向量列表。"""
        try:
            payload = json.loads(response_body)
        except JSONDecodeError:
            raise EmbeddingProviderError(
                f"{self.provider_name} JSONDecodeError: invalid embedding response"
            ) from None

        try:
            # /api/embed 的标准响应是 {"embeddings": [[...], [...]]}。
            if not isinstance(payload, Mapping):
                raise TypeError("response body must be a mapping")

            raw_vectors = payload["embeddings"]
            if not isinstance(raw_vectors, list):
                raise TypeError("embeddings must be a list")
            if len(raw_vectors) != expected_count:
                raise ValueError(
                    f"expected {expected_count} vectors, got {len(raw_vectors)}"
                )

            vectors: list[list[float]] = []
            for position, raw_vector in enumerate(raw_vectors):
                if not isinstance(raw_vector, list) or not raw_vector:
                    raise TypeError(
                        f"embeddings[{position}] must be a non-empty list"
                    )

                vector: list[float] = []
                for value in raw_vector:
                    # bool 是 int 的子类，但 True/False 不是合法的向量维度。
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("embedding values must be numbers")
                    try:
                        numeric_value = float(value)
                    except (OverflowError, ValueError):
                        raise ValueError(
                            "embedding values must be finite numbers"
                        ) from None
                    if not math.isfinite(numeric_value):
                        raise ValueError("embedding values must be finite numbers")
                    vector.append(numeric_value)
                vectors.append(vector)

            return vectors
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            self._raise_invalid_response(exc)

    def _build_url(self) -> str:
        """拼接服务根地址和 Ollama 当前的批量 Embedding 路径。"""
        return f"{self.base_url}/api/embed"

    def _build_headers(self) -> dict[str, str]:
        """Ollama 本地接口不需要密钥，只声明 JSON 请求体。"""
        return {"Content-Type": "application/json"}

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        """读取服务地址，未配置时使用本机 Ollama 默认端口。"""
        value = config.get("base_url", self.default_base_url)
        return self._normalize_base_url(value)

    def _normalize_base_url(self, value: object) -> str:
        """校验 HTTP(S) 根地址并移除末尾斜杠，避免路径出现双斜杠。"""
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
        """读取一个有限且大于零的超时秒数，默认等待 30 秒。"""
        value = config.get("timeout", 30.0)
        # bool 虽然是 int 的子类，但 True/False 不是有意义的超时秒数。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._raise_configuration("timeout must be a positive number")
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            self._raise_configuration("timeout must be a positive number")
        return timeout

    def _require_text(self, config: Mapping[str, Any], field: str) -> str:
        """读取必填字符串配置，并在错误中指出 embedding 字段路径。"""
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
