"""文档加载器的最小抽象契约。

整体数据流只有一件事：

``本地源文件 -> BaseLoader.load(path) -> core.Document``

输入是一个文件路径，输出始终是项目自己的 ``Document``。这样后面的切块、
图片描述、向量化和存储只依赖统一数据结构，不需要知道上游使用的是 PDF、
Markdown 还是未来新增的网页加载器。本层不负责切块或入库，这些属于后续阶段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core import Document


class BaseLoader(ABC):
    """所有文件加载器必须实现的单一输入/输出接口。"""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """读取一个源文件并返回规范化文档。

        Args:
            path: 待读取的本地文件路径；具体实现负责校验支持的文件类型。

        Returns:
            ``Document``，其中 ``text`` 是后续切块所用正文，``metadata`` 至少
            包含 ``source_path``，``id`` 必须能稳定标识相同文件内容。
        """
        raise NotImplementedError


__all__ = ["BaseLoader"]
