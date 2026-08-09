"""Vision LLM 的供应商无关契约。

它在整体架构中的位置：

``PDF/图片 -> ImageCaptioner (阶段 C7) -> BaseVisionLLM -> Azure/OpenAI 等具体服务``

ImageCaptioner 只需要知道“给模型一段指令和一张图片，可以得到文字描述”，
不需要知道 Azure、OpenAI 或本地模型各自的 HTTP 请求格式。这样更换 Vision
模型时，只替换 Libs 层的 Provider，不会改动摄取流水线。

输入：

- ``text``：告诉模型要如何理解图片的文字指令，例如“用一句话描述该图表”；
- ``image_path``：图片文件路径、已经编码好的 base64 文本，或原始图片字节；
- ``trace``：可选的链路追踪上下文，供后续可观测性阶段记录调用信息。

输出：

- ``ChatResponse``：当前与文字 LLM 一致，返回模型生成的文本；例如图片 caption。

本模块只定义调用双方必须遵守的形状，不读取图片、不压缩图片、也不访问网络。
具体 Provider 可以覆写 ``preprocess_image``，在真正发送前加入压缩、尺寸限制或
格式转换，避免把这些供应商差异泄漏到 ImageCaptioner。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


# 现有 BaseLLM.chat() 的返回值就是 str。起一个稳定名字，使调用 Vision 接口时
# 能看出这是模型响应；以后需要附带 usage、模型名等字段时可在此处演进。
ChatResponse: TypeAlias = str
ImageInput: TypeAlias = str | bytes


class BaseVisionLLM(ABC):
    """定义所有“能看图的语言模型”必须提供的最小能力。"""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: ImageInput,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """根据一段指令和一张图片生成文字响应。

        Args:
            text: 非空文字指令。调用者用它说明图片需要被描述、识别或总结什么。
            image_path: 图片路径、base64 文本或原始字节。Provider 负责识别和编码它。
            trace: 可选追踪对象；未启用追踪时传入 ``None``。

        Returns:
            模型生成的文本，例如可写回 Chunk metadata 的图片 caption。

        Raises:
            ValueError: 当实现方发现输入格式不符合其要求时。
            RuntimeError: 当模型服务、图片处理或网络调用失败时。
        """
        raise NotImplementedError

    def preprocess_image(self, image: ImageInput) -> ImageInput:
        """返回可发送给 Provider 的图片，默认原样返回。

        这是图片预处理的扩展点：B9 的 Azure Provider 可以在这里做尺寸压缩、
        PNG/JPEG 格式转换或 base64 编码。当前抽象层不改变图片，确保测试和未来
        本地模型都可以按自己的需求覆写该方法。

        Args:
            image: 来自 ``chat_with_image`` 的图片路径、base64 文本或原始字节。

        Returns:
            预处理后的同类图片输入，供具体 Provider 发送给模型服务。
        """
        return image
