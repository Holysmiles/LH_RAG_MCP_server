"""Recursive Splitter 的行为测试。

测试不会访问网络；它们既验证 LangChain 适配关系，也是第一次阅读
``RecursiveSplitter`` 时可以直接照着理解的使用示例。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


@pytest.mark.unit
def test_factory_creates_recursive_splitter_and_copies_config() -> None:
    """provider=recursive 时，工厂应返回内置实现并隔离配置字典。"""
    config = {
        "provider": "recursive",
        "chunk_size": 80,
        "chunk_overlap": 10,
    }

    splitter = SplitterFactory.create(Settings(splitter=config))

    assert isinstance(splitter, RecursiveSplitter)
    assert isinstance(splitter, BaseSplitter)
    assert splitter.config == config
    assert splitter.config is not config
    assert isinstance(
        splitter._langchain_splitter, RecursiveCharacterTextSplitter
    )


@pytest.mark.unit
def test_split_text_uses_recursive_boundaries_and_overlap() -> None:
    """没有自然分隔符时按字符递归切分，并保留配置的重叠字符。"""
    splitter = RecursiveSplitter({"chunk_size": 8, "chunk_overlap": 2})

    chunks = splitter.split_text("abcdefghijklmnop")

    assert chunks == ["abcdefgh", "ghijklmn", "mnop"]
    assert all(chunk for chunk in chunks)


@pytest.mark.unit
def test_heading_line_is_never_split_into_random_characters() -> None:
    """标题长度正常时，即使正文很长，标题行也必须保持完整。"""
    splitter = RecursiveSplitter({"chunk_size": 16, "chunk_overlap": 0})

    chunks = splitter.split_text("# Retrieval Design\nThis is the body.")

    assert any(chunk.startswith("# Retrieval Design") for chunk in chunks)
    assert "# Retrieval Design" not in "".join(
        chunk for chunk in chunks if not chunk.startswith("# Retrieval Design")
    )

    long_heading = "# This heading is longer than the configured chunk size"
    assert splitter.split_text(long_heading) == [long_heading]


@pytest.mark.unit
def test_langchain_respects_normal_markdown_heading_boundary() -> None:
    """普通标题由 LangChain 的 Markdown separators 作为完整边界保留。"""
    source = (
        "Preface text that fills space.\n\n"
        "## API\n"
        "Body text that also fills space."
    )

    chunks = RecursiveSplitter(
        {"chunk_size": 32, "chunk_overlap": 0}
    ).split_text(source)

    assert "## API" in chunks
    assert all("##" not in chunk or chunk == "## API" for chunk in chunks)


@pytest.mark.unit
def test_fenced_code_block_is_kept_as_one_chunk() -> None:
    """代码块的开闭围栏、换行和缩进不能被递归切分破坏。"""
    source = (
        "# Example\n\n"
        "Before the code.\n\n"
        "```python\n"
        "def greet(name):\n"
        "    return f\"Hello {name}\"\n"
        "```\n\n"
        "After the code."
    )
    splitter = RecursiveSplitter({"chunk_size": 20, "chunk_overlap": 4})

    chunks = splitter.split_text(source)
    code_chunks = [chunk for chunk in chunks if chunk.startswith("```python")]

    assert len(code_chunks) == 1
    assert code_chunks[0].endswith("```\n")
    assert "    return f\"Hello {name}\"" in code_chunks[0]
    assert len(code_chunks[0]) > splitter.chunk_size


@pytest.mark.unit
def test_image_placeholder_is_kept_as_one_chunk() -> None:
    """Loader 图片 ID 很长，但结构标记不能因 chunk_size 很小而失去完整性。"""
    placeholder = f"[IMAGE: {'a' * 64}_1_1]"
    source = f"Before image.\n\n{placeholder}\n\nAfter image."

    chunks = RecursiveSplitter(
        {"chunk_size": 20, "chunk_overlap": 0}
    ).split_text(source)

    assert any(chunk.strip() == placeholder for chunk in chunks)
    assert sum(placeholder in chunk for chunk in chunks) == 1


@pytest.mark.unit
def test_tilde_and_unclosed_fences_are_safe() -> None:
    """波浪线围栏和未闭合围栏都应作为完整代码片段保留。"""
    source = "~~~sql\nSELECT 1;\n~~~   \n\n~~~python\nprint('unfinished')\n"

    chunks = RecursiveSplitter({"chunk_size": 10, "chunk_overlap": 0}).split_text(
        source
    )

    assert any(
        chunk.startswith("~~~sql") and chunk.endswith("~~~   \n")
        for chunk in chunks
    )
    assert any(
        chunk.startswith("~~~python") and "print('unfinished')" in chunk
        for chunk in chunks
    )


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_returns_no_fake_chunk(text: str) -> None:
    """空文档没有可检索内容，因此返回空列表。"""
    assert RecursiveSplitter({}).split_text(text) == []


@pytest.mark.unit
def test_non_string_text_is_rejected() -> None:
    """错误类型应在真正切分前被指出。"""
    with pytest.raises(ValueError, match=r"recursive validation error:.*string"):
        RecursiveSplitter({}).split_text(42)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": True},
        {"chunk_overlap": -1},
        {"chunk_overlap": True},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"chunk_size": 10, "chunk_overlap": 11},
    ],
)
def test_invalid_chunk_configuration_is_rejected(config: dict[str, Any]) -> None:
    """非法尺寸会导致死循环或无意义结果，必须在初始化时失败。"""
    with pytest.raises(ValueError, match=r"recursive configuration error"):
        RecursiveSplitter(config)
