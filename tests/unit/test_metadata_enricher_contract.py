"""Contract tests for deterministic and LLM-backed metadata enrichment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from core import Chunk
from core.settings import Settings
from core.trace import TraceContext
from ingestion.transform import BaseTransform, MetadataEnricher
from libs.llm.base_llm import BaseLLM, ChatMessage


class FakeLLM(BaseLLM):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self.response


class FailingLLM(BaseLLM):
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        raise RuntimeError("provider unavailable")


def settings(*, use_llm: bool = False) -> Settings:
    return Settings(
        llm={"provider": "openai", "model": "test", "api_key": "test"},
        ingestion={"metadata_enricher": {"use_llm": use_llm}},
    )


def make_chunk(text: str = "# Azure Setup\nConfigure the endpoint and deployment.") -> Chunk:
    return Chunk(
        id="doc-1_0000_abcd1234",
        text=text,
        metadata={"source_path": "docs/guide.pdf", "chunk_index": 0},
        start_offset=5,
        end_offset=5 + len(text),
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_enricher_implements_transform_contract_and_rule_fields() -> None:
    enricher = MetadataEnricher(settings())
    assert isinstance(enricher, BaseTransform)
    output = enricher.transform([make_chunk()])[0]

    assert output is not None
    assert output.metadata["title"] == "Azure Setup"
    assert output.metadata["summary"] == "Configure the endpoint and deployment."
    assert output.metadata["tags"]
    assert output.metadata["enriched_by"] == "rule"


@pytest.mark.unit
def test_rule_mode_preserves_metadata_and_chunk_identity_without_mutation() -> None:
    original = make_chunk("Plain text about vector stores.")
    original.metadata["tags"] = ["existing"]

    output = MetadataEnricher(settings()).transform([original])[0]

    assert output is not original
    assert output.id == original.id
    assert output.start_offset == original.start_offset
    assert output.end_offset == original.end_offset
    assert output.source_ref == original.source_ref
    assert output.metadata["source_path"] == "docs/guide.pdf"
    assert output.metadata["tags"] == ["existing"]
    assert "title" not in original.metadata


@pytest.mark.unit
def test_llm_mode_accepts_structured_json_and_uses_real_injected_model() -> None:
    llm = FakeLLM('{"title":"Endpoint Setup","summary":"LLM summary","tags":["azure","config"]}')
    enricher = MetadataEnricher(settings(use_llm=True), llm=llm)

    output = enricher.transform([make_chunk()])[0]

    assert output.metadata["title"] == "Endpoint Setup"
    assert output.metadata["summary"] == "LLM summary"
    assert output.metadata["tags"] == ["azure", "config"]
    assert output.metadata["enriched_by"] == "llm"
    assert len(llm.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "response,reason",
    [
        ("", "llm_returned_empty_text"),
        ("not json", "llm_invalid_metadata:JSONDecodeError"),
        ('{"title":"Only title"}', "llm_invalid_metadata:ValueError"),
        ('{"title":"x","summary":"y","tags":[]}', "llm_invalid_metadata:ValueError"),
    ],
)
def test_invalid_llm_output_falls_back_to_rule_metadata(
    response: str, reason: str
) -> None:
    output = MetadataEnricher(
        settings(use_llm=True), llm=FakeLLM(response)
    ).transform([make_chunk()])[0]

    assert output.metadata["enriched_by"] == "rule"
    assert output.metadata["title"] == "Azure Setup"
    assert output.metadata["fallback_reason"] == reason


@pytest.mark.unit
def test_llm_exception_and_factory_failure_do_not_block_ingestion() -> None:
    failed_call = MetadataEnricher(
        settings(use_llm=True), llm=FailingLLM()
    ).transform([make_chunk()])[0]
    assert failed_call.metadata["fallback_reason"] == "llm_call_failed:RuntimeError"

    broken_settings = Settings(
        llm={"provider": "missing", "model": "none"},
        ingestion={"metadata_enricher": {"use_llm": True}},
    )
    factory_failure = MetadataEnricher(broken_settings).transform([make_chunk()])[0]
    assert factory_failure.metadata["fallback_reason"] == (
        "llm_initialization_failed:ValueError"
    )


@pytest.mark.unit
def test_json_code_fence_and_nested_metadata_are_supported() -> None:
    llm = FakeLLM('```json\n{"metadata":{"title":"T","summary":"S","tags":"a, b"}}\n```')
    output = MetadataEnricher(settings(use_llm=True), llm=llm).transform([make_chunk()])[0]

    assert output.metadata["title"] == "T"
    assert output.metadata["summary"] == "S"
    assert output.metadata["tags"] == ["a", "b"]


@pytest.mark.unit
def test_trace_records_batch_outcomes_and_input_validation() -> None:
    trace = TraceContext()
    output = MetadataEnricher(settings()).transform([make_chunk(), make_chunk("text")], trace)

    assert len(output) == 2
    assert trace.stages == [
        {
            "input_count": 2,
            "output_count": 2,
            "rule_enriched": 2,
            "llm_enriched": 0,
            "original_preserved": 0,
            "fallback_count": 0,
            "stage": "metadata_enricher",
        }
    ]
    with pytest.raises(TypeError, match="chunks must be"):
        MetadataEnricher(settings()).transform("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"chunks\[0\]"):
        MetadataEnricher(settings()).transform([object()])  # type: ignore[list-item]


@pytest.mark.unit
def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(TypeError, match="metadata_enricher.use_llm"):
        MetadataEnricher(
            Settings(ingestion={"metadata_enricher": {"use_llm": "yes"}})
        )


@pytest.mark.unit
def test_rule_failure_is_isolated_and_returns_minimum_safe_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enricher = MetadataEnricher(settings())

    def fail_rule(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("broken rule")

    monkeypatch.setattr(enricher, "_rule_metadata", fail_rule)

    output = enricher.transform([make_chunk(), make_chunk("second chunk")])

    assert len(output) == 2
    assert all(item.metadata["title"] for item in output)
    assert all(item.metadata["summary"] for item in output)
    assert all(item.metadata["tags"] for item in output)
    assert all(item.metadata["enriched_by"] == "original" for item in output)


@pytest.mark.unit
def test_existing_and_llm_text_fields_are_bounded() -> None:
    chunk = make_chunk()
    chunk.metadata.update({"title": "t" * 500, "summary": "s" * 800})
    rule_output = MetadataEnricher(settings()).transform([chunk])[0]

    llm = FakeLLM(
        '{"title":"' + "t" * 500 + '","summary":"' + "s" * 800 + '","tags":["' + "x" * 100 + '"]}'
    )
    llm_output = MetadataEnricher(settings(use_llm=True), llm=llm).transform(
        [make_chunk()]
    )[0]

    assert len(rule_output.metadata["title"]) <= 120
    assert len(rule_output.metadata["summary"]) <= 300
    assert len(llm_output.metadata["title"]) <= 120
    assert len(llm_output.metadata["summary"]) <= 300
    assert len(llm_output.metadata["tags"][0]) <= 64
