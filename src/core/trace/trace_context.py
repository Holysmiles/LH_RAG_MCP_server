"""请求级追踪的最小数据容器。

C5 只需要让 Transform 能把“处理了多少 Chunk、走了哪种路径”记录下来，因此
本阶段提供最小接口：创建唯一 ``trace_id``，并按顺序追加阶段字典。F1 会在这个
稳定接口上继续增加开始/结束时间、耗时和序列化，不在 C5 提前混入持久化逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4


class TraceContext:
    """保存一次请求中按发生顺序记录的阶段数据。

    Inputs:
        无必填参数；创建时自动生成全局唯一 ``trace_id``。

    Outputs:
        ``stages`` 是按调用顺序保存的字典列表，每项至少包含 ``stage`` 名称。
    """

    def __init__(self) -> None:
        self.trace_id = str(uuid4())
        self.stages: list[dict[str, Any]] = []

    def record_stage(
        self,
        stage_name: str,
        details: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """追加一个阶段快照，不保留调用方可变字典的引用。"""
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string")
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("details must be a mapping or None")

        stage: dict[str, Any] = {}
        if details is not None:
            stage.update(deepcopy(dict(details)))
        stage.update(deepcopy(fields))
        # 调用方可以自由记录业务字段，但不能把阶段身份改成另一个名称。
        stage["stage"] = stage_name.strip()
        self.stages.append(stage)


__all__ = ["TraceContext"]
