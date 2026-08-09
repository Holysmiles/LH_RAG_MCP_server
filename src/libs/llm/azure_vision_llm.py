"""通过 Azure OpenAI 把图片转换为文字描述。

它在整体 RAG 架构中的位置：

``PDF Loader -> 图片引用 -> ImageCaptioner (C7) -> AzureVisionLLM -> caption``

输入分成两部分：

- ``text``：告诉 GPT-4o 要从图片中提取什么，例如“描述图表趋势”；
- ``image_path``：本地图片路径、base64/data URI 文本或原始图片字节。

输出是 Azure ``choices[0].message.content`` 中的纯文本。C7 会把这段文字写回
Chunk，使只存在于图片里的信息也能被后续 Embedding 和检索命中。

本模块负责图片读取、尺寸压缩、base64 编码和 Azure 多模态请求组装；共享的
HTTP POST、超时、安全错误包装和响应解析继续复用现有 AzureLLM 骨架。
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from PIL import Image, UnidentifiedImageError

from libs.llm.azure_llm import AzureLLM
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse, ImageInput

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["AzureVisionLLM"]


_DATA_URI = re.compile(
    r"^data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,(?P<data>.*)$",
    flags=re.DOTALL,
)
_AZURE_IMAGE_MIME_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class AzureVisionLLM(AzureLLM, BaseVisionLLM):
    """调用 Azure GPT-4o/GPT-4-Vision 的图片理解接口。

    ``AzureLLM`` 提供 deployment URL、``api-key`` 请求头和安全 HTTP 流程；
    ``BaseVisionLLM`` 提供 ImageCaptioner 依赖的多模态接口。本类只填补两者
    之间的图片预处理和请求体差异。
    """

    provider_name: ClassVar[str] = "azure_vision"
    default_max_image_size: ClassVar[int] = 2048

    def __init__(self, config: Mapping[str, Any]) -> None:
        """读取 Azure 和图片配置，此时不会访问网络或打开图片。"""
        super().__init__(config)
        self.max_image_size = self._read_positive_int(
            config,
            "max_image_size",
            self.default_max_image_size,
        )

    def chat_with_image(
        self,
        text: str,
        image_path: ImageInput,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """把文字和图片发给 Azure，并返回模型生成的文字。

        数据转换顺序：

        1. 校验非空文字指令；
        2. 读取并验证图片，必要时按比例压缩；
        3. 把图片编码成 ``data:image/...;base64,...``；
        4. 组装 Azure 多模态 ``messages``；
        5. 复用 HTTP 骨架发送请求并提取文本响应。
        """
        del trace
        if not isinstance(text, str) or not text.strip():
            self._raise_validation("text must be a non-empty string")

        image_data_uri = self.preprocess_image(image_path)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text.strip()},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                    ],
                }
            ]
        }
        response_body = self._send_json_payload(payload, operation="vision")
        return self._extract_content(response_body)

    def preprocess_image(self, image: ImageInput) -> str:
        """把路径/base64/bytes 统一转换为 Azure 可接收的 data URI。

        输入图片的长边大于 ``max_image_size`` 时使用高质量 Lanczos 算法按
        原宽高比缩小。小图片不会被二次压缩，只做格式校验和 base64 包装。

        Returns:
            形如 ``data:image/png;base64,iVBOR...`` 的字符串。
        """
        image_bytes = self._read_image_bytes(image)

        try:
            with Image.open(BytesIO(image_bytes)) as opened_image:
                # load() 在任何 HTTP 请求前完整解码图片，损坏文件会在这里失败。
                opened_image.load()
                image_format = (opened_image.format or "").upper()
                output_format = (
                    image_format
                    if image_format in _AZURE_IMAGE_MIME_TYPES
                    else "PNG"
                )
                must_resize = max(opened_image.size) > self.max_image_size
                must_convert_format = output_format != image_format

                if must_resize or must_convert_format:
                    prepared_image = opened_image.copy()
                    if must_resize:
                        prepared_image.thumbnail(
                            (self.max_image_size, self.max_image_size),
                            Image.Resampling.LANCZOS,
                        )
                    if output_format == "JPEG" and prepared_image.mode not in {
                        "L",
                        "RGB",
                    }:
                        prepared_image = prepared_image.convert("RGB")

                    output = BytesIO()
                    prepared_image.save(output, format=output_format)
                    image_bytes = output.getvalue()

                mime_type = _AZURE_IMAGE_MIME_TYPES[output_format]
        except (OSError, UnidentifiedImageError, ValueError):
            self._raise_validation("image input is not a supported image")

        encoded = b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _read_image_bytes(self, image: ImageInput) -> bytes:
        """读取三种公开输入形态，并在联网前拒绝空值或无效 base64。"""
        if isinstance(image, bytes):
            if not image:
                self._raise_validation("image bytes must not be empty")
            return image

        if not isinstance(image, str) or not image.strip():
            self._raise_validation(
                "image_path must be a non-empty path, base64 string, or bytes"
            )

        normalized = image.strip()
        data_uri_match = _DATA_URI.fullmatch(normalized)
        if data_uri_match is not None:
            return self._decode_base64(data_uri_match.group("data"))

        # 普通字符串优先按文件路径解释。路径不存在时再按原始 base64 解释，
        # 因此调用者不必为了本地文件手工编码，也可以直接传数据库中的 base64。
        path = Path(normalized)
        try:
            if path.exists():
                if not path.is_file():
                    self._raise_validation("image path must point to a file")
                try:
                    image_bytes = path.read_bytes()
                except OSError as exc:
                    raise RuntimeError(
                        f"{self.provider_name} {type(exc).__name__}: "
                        "image file could not be read"
                    ) from None
                if not image_bytes:
                    self._raise_validation("image file must not be empty")
                return image_bytes
        except OSError:
            # Windows 会拒绝超长或包含非法字符的“路径”。它仍可能是合法 base64，
            # 所以这里继续交给 base64 解码器判断。
            pass

        try:
            return self._decode_base64(normalized)
        except ValueError:
            self._raise_validation(
                "image_path must be an existing file path or valid base64 image"
            )

    def _decode_base64(self, value: str) -> bytes:
        """严格解码 base64；允许 YAML/JSON 中为了排版加入的空白字符。"""
        compact_value = "".join(value.split())
        if not compact_value:
            self._raise_validation("base64 image must not be empty")
        try:
            decoded = b64decode(compact_value, validate=True)
        except (Base64Error, ValueError):
            raise ValueError("invalid base64 image") from None
        if not decoded:
            self._raise_validation("base64 image must not be empty")
        return decoded

    def _resolve_base_url(self, config: Mapping[str, Any]) -> str:
        """Vision 配置从独立 ``vision_llm`` 段读取 Azure endpoint。"""
        endpoint = config.get("azure_endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            self._raise_configuration(
                "missing required field 'vision_llm.azure_endpoint'"
            )
        return self._normalize_base_url(endpoint)

    def _require_text(self, config: Mapping[str, Any], field: str) -> str:
        """读取 Vision 必填字段，并在错误中给出正确的配置路径。"""
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            self._raise_configuration(
                f"missing required field 'vision_llm.{field}'"
            )
        return value.strip()

    def _read_positive_int(
        self,
        config: Mapping[str, Any],
        field: str,
        default: int,
    ) -> int:
        """读取正整数图片尺寸，排除 Python 中属于 int 子类的 bool。"""
        value = config.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            self._raise_configuration(f"{field} must be a positive integer")
        return value

    def _raise_validation(self, detail: str) -> NoReturn:
        raise ValueError(f"{self.provider_name} validation error: {detail}")
