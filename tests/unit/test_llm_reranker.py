"""Offline behavior tests for the B7.7 LLM reranker.

The fake LLM below never accesses a network.  Tests can therefore inspect the
exact prompt, choose the returned JSON, and simulate failures deterministically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.ollama_llm import OllamaLLM
from libs.reranker.llm_reranker import LLMReranker, LLMRerankerError
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM(BaseLLM):
    """Return a configured response and remember the received messages."""

    def __init__(
        self,
        response: object,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response
        self.failure = failure
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if self.failure is not None:
            raise self.failure
        return self.response  # type: ignore[return-value]


@pytest.fixture
def candidates() -> list[dict[str, Any]]:
    """Candidates contain extra fields that must survive reranking unchanged."""
    return [
        {
            "chunk_id": "chunk-1",
            "text": "Python lists preserve insertion order.",
            "score": 0.8,
            "metadata": {"source": "python.pdf"},
        },
        {
            "chunk_id": "chunk-2",
            "text": "Reranking changes the final relevance order.",
            "score": 0.5,
            "metadata": {"source": "rag.pdf"},
        },
        {
            "chunk_id": "chunk-3",
            "text": "This passage is unrelated.",
            "score": 0.2,
            "metadata": {"source": "other.pdf"},
        },
    ]


@pytest.mark.unit
def test_factory_creates_builtin_llm_reranker() -> None:
    """provider=llm reuses the application's existing LLM configuration."""
    rerank_config = {
        "provider": "llm",
        "prompt_path": "config/prompts/rerank.txt",
    }
    llm_config = {"provider": "ollama", "model": "qwen2.5:7b"}

    reranker = RerankerFactory.create(
        Settings(llm=llm_config, rerank=rerank_config)
    )

    assert isinstance(reranker, LLMReranker)
    assert isinstance(reranker.llm, OllamaLLM)
    assert reranker.config == {**rerank_config, "llm": llm_config}
    assert reranker.config["llm"] is not llm_config


@pytest.mark.unit
def test_rerank_orders_original_objects_from_ranked_ids(
    candidates: list[dict[str, Any]],
) -> None:
    """The model chooses IDs; the adapter returns the original dictionaries."""
    fake_llm = FakeLLM(
        json.dumps({"ranked_ids": ["chunk-2", "chunk-1", "chunk-3"]})
    )
    reranker = LLMReranker({}, llm=fake_llm, prompt_text="Rank passages.")

    ranked = reranker.rerank("How does reranking work?", candidates)

    assert [item["chunk_id"] for item in ranked] == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]
    assert ranked[0] is candidates[1]
    assert ranked[1] is candidates[0]
    assert ranked[0]["metadata"] == {"source": "rag.pdf"}


@pytest.mark.unit
def test_prompt_contains_schema_query_ids_and_text(
    candidates: list[dict[str, Any]],
) -> None:
    """A reader can see exactly which local fields are sent to the model."""
    fake_llm = FakeLLM(
        json.dumps({"ranked_ids": ["chunk-1", "chunk-2", "chunk-3"]})
    )
    reranker = LLMReranker(
        {},
        llm=fake_llm,
        prompt_text="CUSTOM RERANK INSTRUCTION",
    )

    reranker.rerank("Which passage explains lists?", candidates)

    messages = fake_llm.calls[0]
    assert messages[0]["role"] == "system"
    assert "CUSTOM RERANK INSTRUCTION" in messages[0]["content"]
    assert '"ranked_ids"' in messages[0]["content"]

    user_payload = json.loads(messages[1]["content"])
    assert user_payload["query"] == "Which passage explains lists?"
    assert user_payload["candidates"][0] == {
        "chunk_id": "chunk-1",
        "text": "Python lists preserve insertion order.",
    }
    assert "metadata" not in user_payload["candidates"][0]
    assert "score" not in user_payload["candidates"][0]


@pytest.mark.unit
def test_empty_candidates_return_without_calling_llm() -> None:
    """No candidates means no cost, network request, or parsing work."""
    fake_llm = FakeLLM(json.dumps({"ranked_ids": []}))
    reranker = LLMReranker({}, llm=fake_llm, prompt_text="Rank passages.")

    assert reranker.rerank("valid query", []) == []
    assert fake_llm.calls == []


@pytest.mark.unit
def test_dataclass_candidates_are_supported_and_preserved() -> None:
    """The Libs layer also works with the future Core dataclass result type."""

    @dataclass
    class Candidate:
        chunk_id: str
        text: str
        score: float

    first = Candidate("one", "First passage", 0.8)
    second = Candidate("two", "Second passage", 0.7)
    fake_llm = FakeLLM(json.dumps({"ranked_ids": ["two", "one"]}))
    reranker = LLMReranker({}, llm=fake_llm, prompt_text="Rank passages.")

    ranked = reranker.rerank("query", [first, second])

    assert ranked == [second, first]
    assert ranked[0] is second


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not-json", "valid JSON"),
        ("[]", "exactly one field"),
        ('{"ranked_ids":[],"explanation":"extra"}', "exactly one field"),
        ('{"ranked_ids":"chunk-1"}', "list of strings"),
        ('{"ranked_ids":[1,2,3]}', "list of strings"),
        (
            '{"ranked_ids":["chunk-1","chunk-1","chunk-3"]}',
            "must not contain duplicates",
        ),
        (
            '{"ranked_ids":["chunk-1","chunk-2","invented"]}',
            "missing=.*chunk-3.*unknown=.*invented",
        ),
        (
            '{"ranked_ids":["chunk-1","chunk-2"]}',
            "contain every candidate",
        ),
    ],
)
def test_invalid_llm_response_raises_readable_fallback_signal(
    candidates: list[dict[str, Any]],
    response: str,
    message: str,
) -> None:
    """Bad model output is never used to partially reorder candidates."""
    reranker = LLMReranker(
        {},
        llm=FakeLLM(response),
        prompt_text="Rank passages.",
    )

    with pytest.raises(LLMRerankerError, match=message):
        reranker.rerank("query", candidates)


@pytest.mark.unit
def test_llm_failure_becomes_stable_fallback_signal(
    candidates: list[dict[str, Any]],
) -> None:
    """Core only needs one safe exception type to trigger its later fallback."""
    reranker = LLMReranker(
        {},
        llm=FakeLLM("", failure=TimeoutError("secret provider details")),
        prompt_text="Rank passages.",
    )

    with pytest.raises(
        LLMRerankerError,
        match="llm reranker failed: TimeoutError",
    ) as error:
        reranker.rerank("query", candidates)

    assert "secret provider details" not in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate_list", "message"),
    [
        ([{"text": "missing id"}], r"chunk_id"),
        ([{"chunk_id": "id", "text": 42}], r"text must be a string"),
        (
            [
                {"chunk_id": "same", "text": "first"},
                {"chunk_id": "same", "text": "second"},
            ],
            "duplicate chunk_id",
        ),
    ],
)
def test_invalid_candidates_fail_before_calling_llm(
    candidate_list: list[dict[str, Any]],
    message: str,
) -> None:
    """Malformed local data should fail locally and should cost no LLM call."""
    fake_llm = FakeLLM(json.dumps({"ranked_ids": []}))
    reranker = LLMReranker({}, llm=fake_llm, prompt_text="Rank passages.")

    with pytest.raises(ValueError, match=message):
        reranker.rerank("query", candidate_list)

    assert fake_llm.calls == []


@pytest.mark.unit
def test_prompt_file_is_read_once_during_construction(tmp_path: Path) -> None:
    """Changing the file later cannot alter an already-created reranker."""
    prompt_path = tmp_path / "rerank.txt"
    prompt_path.write_text("FIRST PROMPT", encoding="utf-8")
    fake_llm = FakeLLM(json.dumps({"ranked_ids": ["one"]}))
    reranker = LLMReranker(
        {"prompt_path": str(prompt_path)},
        llm=fake_llm,
    )
    prompt_path.write_text("CHANGED PROMPT", encoding="utf-8")

    reranker.rerank("query", [{"chunk_id": "one", "text": "text"}])

    assert "FIRST PROMPT" in fake_llm.calls[0][0]["content"]
    assert "CHANGED PROMPT" not in fake_llm.calls[0][0]["content"]
