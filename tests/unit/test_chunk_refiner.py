"""C5 ChunkRefiner 的离线单元测试。

测试严格区分确定性规则和外部 LLM：规则样例来自 JSON fixture；模型路径注入
FakeLLM，不联网、不产生费用。每个输出还必须保留 Chunk 的 ID、offset 和父引用。
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import pytest

from core import Chunk
from core.settings import Settings
from core.trace import TraceContext
from ingestion.transform import BaseTransform, ChunkRefiner
from libs.llm.base_llm import BaseLLM, ChatMessage


FIXTURE_PATH = Path("tests/fixtures/noisy_chunks.json")


class FakeLLM(BaseLLM):
    """记录消息并返回固定文本的离线模型。"""

    def __init__(self, response: str = "LLM refined passage") -> None:
        self.response = response
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self.response


class FailingLLM(BaseLLM):
    """模拟 Provider 网络或模型失败。"""

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        raise RuntimeError("provider unavailable")


def settings(*, use_llm: bool) -> Settings:
    return Settings(
        llm={"provider": "openai", "model": "gpt-test", "api_key": "test"},
        ingestion={"chunk_refiner": {"use_llm": use_llm}},
    )


def make_chunk(text: str = "Raw    chunk") -> Chunk:
    return Chunk(
        id="doc-1_0000_abcd1234",
        text=text,
        metadata={"source_path": "reports/source.pdf", "chunk_index": 0},
        start_offset=10,
        end_offset=10 + len(text),
        source_ref="doc-1",
    )


def noisy_cases() -> list[dict[str, str]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_base_transform_is_abstract_and_refiner_implements_contract() -> None:
    with pytest.raises(TypeError):
        BaseTransform()
    assert isinstance(ChunkRefiner(settings(use_llm=False)), BaseTransform)


@pytest.mark.unit
@pytest.mark.parametrize("case", noisy_cases(), ids=lambda item: item["name"])
def test_rule_based_refine_matches_noise_fixtures(case: dict[str, str]) -> None:
    """八类样例固定规则清洗的允许范围，并防止清洗器越权改写事实。"""
    refiner = ChunkRefiner(settings(use_llm=False))

    assert refiner._rule_based_refine(case["input"]) == case["expected"]


@pytest.mark.unit
def test_rule_transform_returns_new_chunk_and_preserves_identity() -> None:
    original = make_chunk("Page 1 of 1\nUseful    facts")

    refined = ChunkRefiner(settings(use_llm=False)).transform([original])[0]

    assert refined is not original
    assert refined.text == "Useful facts"
    assert refined.id == original.id
    assert refined.start_offset == original.start_offset
    assert refined.end_offset == original.end_offset
    assert refined.source_ref == original.source_ref
    assert refined.metadata["refined_by"] == "rule"
    assert "fallback_reason" not in refined.metadata
    assert "refined_by" not in original.metadata


@pytest.mark.unit
def test_successful_retry_clears_stale_fallback_metadata() -> None:
    """同一 Chunk 重试成功后，不能继续携带上一次失败的诊断信息。"""
    chunk = make_chunk("Useful    facts")
    chunk.metadata.update(
        refined_by="original",
        fallback_reason="previous failure",
    )

    output = ChunkRefiner(settings(use_llm=False)).transform([chunk])[0]

    assert output.text == "Useful facts"
    assert output.metadata["refined_by"] == "rule"
    assert "fallback_reason" not in output.metadata


@pytest.mark.unit
def test_empty_batch_returns_empty_and_records_trace() -> None:
    """空批次不应调用外部模型，但仍可形成可观测的零计数阶段。"""
    llm = FakeLLM()
    trace = TraceContext()

    output = ChunkRefiner(settings(use_llm=True), llm=llm).transform([], trace)

    assert output == []
    assert llm.calls == []
    assert trace.stages == [
        {
            "input_count": 0,
            "output_count": 0,
            "llm_refined": 0,
            "rule_refined": 0,
            "original_preserved": 0,
            "fallback_count": 0,
            "stage": "chunk_refiner",
        }
    ]


@pytest.mark.unit
def test_code_block_protection_token_collision_is_reversible() -> None:
    """正文即使包含内部 token 字样，也不能覆盖或错位恢复代码块。"""
    source = (
        "C5PROTECTEDCODEBLOCK0TOKEN\n\n"
        "```python\n"
        "value    = 1\n"
        "```"
    )

    refined = ChunkRefiner(settings(use_llm=False))._rule_based_refine(source)

    assert refined == source


@pytest.mark.unit
def test_llm_receives_rule_cleaned_text_and_marks_metadata(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Clean this input:\n{text}", encoding="utf-8")
    llm = FakeLLM("Self-contained answer")
    refiner = ChunkRefiner(
        settings(use_llm=True),
        llm=llm,
        prompt_path=prompt_path,
    )

    output = refiner.transform([make_chunk("Page 1 of 1\nUseful    facts")])[0]

    assert output.text == "Self-contained answer"
    assert output.metadata["refined_by"] == "llm"
    assert "fallback_reason" not in output.metadata
    assert len(llm.calls) == 1
    assert llm.calls[0] == [
        {"role": "user", "content": "Clean this input:\nUseful facts"}
    ]


@pytest.mark.unit
def test_use_llm_false_never_calls_injected_model() -> None:
    llm = FakeLLM()

    output = ChunkRefiner(settings(use_llm=False), llm=llm).transform(
        [make_chunk()]
    )[0]

    assert llm.calls == []
    assert output.metadata["refined_by"] == "rule"


@pytest.mark.unit
def test_llm_exception_falls_back_to_rule_result() -> None:
    output = ChunkRefiner(
        settings(use_llm=True),
        llm=FailingLLM(),
    ).transform([make_chunk("Page 1 of 1\nUseful    facts")])[0]

    assert output.text == "Useful facts"
    assert output.metadata["refined_by"] == "rule"
    assert output.metadata["fallback_reason"] == "llm_call_failed:RuntimeError"


@pytest.mark.unit
@pytest.mark.parametrize("response", ["", "   ", "\n"])
def test_empty_llm_response_falls_back_to_rule(response: str) -> None:
    output = ChunkRefiner(
        settings(use_llm=True),
        llm=FakeLLM(response),
    ).transform([make_chunk("alpha    beta")])[0]

    assert output.text == "alpha beta"
    assert output.metadata["refined_by"] == "rule"
    assert output.metadata["fallback_reason"] == "llm_returned_empty_text"


@pytest.mark.unit
def test_llm_cannot_drop_loader_image_reference() -> None:
    """图片占位符是 C7 的外键；模型漏掉它时必须回退，不能接受损坏结果。"""
    source = "Chart follows.\n\n[IMAGE: report_2_1]"

    output = ChunkRefiner(
        settings(use_llm=True),
        llm=FakeLLM("A chart follows."),
    ).transform([make_chunk(source)])[0]

    assert output.text == source
    assert output.metadata["refined_by"] == "rule"
    assert output.metadata["fallback_reason"] == "llm_dropped_image_reference"


@pytest.mark.unit
def test_llm_factory_initialization_failure_is_deferred_to_fallback() -> None:
    broken_settings = Settings(
        llm={"provider": "missing", "model": "none"},
        ingestion={"chunk_refiner": {"use_llm": True}},
    )

    refiner = ChunkRefiner(broken_settings)
    output = refiner.transform([make_chunk("alpha    beta")])[0]

    assert output.text == "alpha beta"
    assert output.metadata["refined_by"] == "rule"
    assert output.metadata["fallback_reason"] == (
        "llm_initialization_failed:ValueError"
    )


@pytest.mark.unit
def test_rule_that_removes_everything_preserves_original_text() -> None:
    chunk = make_chunk("Page 1 of 1\n---")

    output = ChunkRefiner(settings(use_llm=False)).transform([chunk])[0]

    assert output.text == chunk.text
    assert output.metadata["refined_by"] == "rule"
    assert output.metadata["fallback_reason"] == "rule_produced_empty_text"


@pytest.mark.unit
def test_one_chunk_failure_does_not_block_following_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refiner = ChunkRefiner(settings(use_llm=False))
    original_rule = refiner._rule_based_refine

    def fail_selected(text: str) -> str:
        if text == "bad":
            raise ValueError("simulated rule failure")
        return original_rule(text)

    monkeypatch.setattr(refiner, "_rule_based_refine", fail_selected)

    outputs = refiner.transform([make_chunk("bad"), make_chunk("good    text")])

    assert outputs[0].text == "bad"
    assert outputs[0].metadata["refined_by"] == "original"
    assert outputs[0].metadata["fallback_reason"] == (
        "chunk_refinement_failed:ValueError"
    )
    assert outputs[1].text == "good text"
    assert outputs[1].metadata["refined_by"] == "rule"


@pytest.mark.unit
def test_trace_records_batch_outcomes() -> None:
    trace = TraceContext()
    refiner = ChunkRefiner(settings(use_llm=True), llm=FailingLLM())

    refiner.transform([make_chunk("one"), make_chunk("two")], trace)

    assert trace.stages == [
        {
            "stage": "chunk_refiner",
            "input_count": 2,
            "output_count": 2,
            "llm_refined": 0,
            "rule_refined": 2,
            "original_preserved": 0,
            "fallback_count": 2,
        }
    ]


@pytest.mark.unit
def test_prompt_falls_back_for_missing_or_invalid_file(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("No placeholder here", encoding="utf-8")
    missing_refiner = ChunkRefiner(
        settings(use_llm=False), prompt_path=tmp_path / "missing.txt"
    )
    invalid_refiner = ChunkRefiner(settings(use_llm=False), prompt_path=invalid)

    assert "{text}" in missing_refiner.prompt_template
    assert invalid_refiner.prompt_template == ChunkRefiner.default_prompt


@pytest.mark.unit
def test_config_and_public_input_validation() -> None:
    with pytest.raises(TypeError, match="use_llm must be a boolean"):
        ChunkRefiner(
            Settings(ingestion={"chunk_refiner": {"use_llm": "yes"}})
        )
    with pytest.raises(TypeError, match="chunks must be"):
        ChunkRefiner(settings(use_llm=False)).transform("not chunks")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"chunks\[0\]"):
        ChunkRefiner(settings(use_llm=False)).transform([object()])  # type: ignore[list-item]


@pytest.mark.unit
def test_trace_context_has_unique_ids_and_copies_stage_details() -> None:
    first = TraceContext()
    second = TraceContext()
    details: dict[str, Any] = {"items": [1]}

    first.record_stage("transform", {**details, "stage": "cannot-overwrite"})
    details["items"].append(2)

    assert first.trace_id != second.trace_id
    assert first.stages == [{"stage": "transform", "items": [1]}]
    with pytest.raises(ValueError, match="stage_name"):
        first.record_stage(" ")
