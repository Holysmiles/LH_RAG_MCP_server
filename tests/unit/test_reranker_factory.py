"""Unit tests for reranker contracts, factory routing, and safe fallback."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.llm.base_llm import BaseLLM
from libs.reranker.base_reranker import BaseReranker, CandidateT
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import NoneReranker, RerankerFactory


class ScoreReranker(BaseReranker):
    """Deterministic fake backend that sorts mapping candidates by score."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def rerank(
        self,
        query: str,
        candidates: Sequence[CandidateT],
        trace: object | None = None,
    ) -> list[CandidateT]:
        score_key = str(self.config.get("score_key", "score"))
        return sorted(
            candidates,
            key=lambda candidate: float(candidate[score_key]),  # type: ignore[index]
            reverse=True,
        )


@pytest.fixture
def registered_score_provider() -> Iterator[None]:
    RerankerFactory.register("score", ScoreReranker)
    yield
    RerankerFactory.unregister("score")


@pytest.fixture
def candidates() -> list[dict[str, Any]]:
    return [
        {"chunk_id": "chunk-1", "score": 0.2},
        {"chunk_id": "chunk-2", "score": 0.9},
        {"chunk_id": "chunk-3", "score": 0.5},
    ]


@pytest.mark.unit
def test_base_reranker_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseReranker()


@pytest.mark.unit
def test_none_reranker_preserves_order_and_candidate_objects(
    candidates: list[dict[str, Any]],
) -> None:
    reranker = NoneReranker()

    ranked = reranker.rerank("query", candidates)

    assert ranked == candidates
    assert ranked is not candidates
    assert all(actual is original for actual, original in zip(ranked, candidates))


@pytest.mark.unit
def test_factory_creates_builtin_none_fallback(
    candidates: list[dict[str, Any]],
) -> None:
    settings = Settings(rerank={"provider": "none", "enabled": False})

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, NoneReranker)
    assert reranker.rerank("query", candidates) == candidates


@pytest.mark.unit
def test_factory_routes_to_registered_backend(
    registered_score_provider: None,
    candidates: list[dict[str, Any]],
) -> None:
    settings = Settings(rerank={"provider": "score", "score_key": "score"})

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, ScoreReranker)
    assert [item["chunk_id"] for item in reranker.rerank("query", candidates)] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]


@pytest.mark.unit
def test_factory_normalizes_provider_name() -> None:
    settings = Settings(rerank={"provider": "  NONE  "})

    assert isinstance(RerankerFactory.create(settings), NoneReranker)


@pytest.mark.unit
def test_factory_passes_a_copy_of_reranker_config() -> None:
    config = {"provider": "none", "enabled": False}
    settings = Settings(rerank=config)

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, NoneReranker)
    assert reranker.config == config
    assert reranker.config is not settings.rerank


@pytest.mark.unit
def test_factory_lists_builtin_none_provider() -> None:
    assert "none" in RerankerFactory.list_providers()


@pytest.mark.unit
def test_factory_reports_missing_provider() -> None:
    with pytest.raises(ValueError, match=r"rerank\.provider"):
        RerankerFactory.create(Settings(rerank={}))


@pytest.mark.unit
def test_factory_rejects_non_string_provider() -> None:
    settings = Settings(rerank={"provider": 42})

    with pytest.raises(TypeError, match=r"rerank\.provider.*string"):
        RerankerFactory.create(settings)


@pytest.mark.unit
def test_factory_rejects_non_mapping_config() -> None:
    settings = Settings(rerank=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"section 'rerank'.*mapping"):
        RerankerFactory.create(settings)


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    settings = Settings(rerank={"provider": "missing"})

    with pytest.raises(ValueError, match="Unsupported reranker provider 'missing'"):
        RerankerFactory.create(settings)


@pytest.mark.unit
def test_duplicate_registration_requires_explicit_replace(
    registered_score_provider: None,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        RerankerFactory.register("SCORE", ScoreReranker)


@pytest.mark.unit
def test_factory_rejects_builder_with_wrong_return_type() -> None:
    RerankerFactory.register(
        "invalid", lambda config: object()  # type: ignore[return-value]
    )
    try:
        settings = Settings(rerank={"provider": "invalid"})

        with pytest.raises(TypeError, match="must return BaseReranker"):
            RerankerFactory.create(settings)
    finally:
        RerankerFactory.unregister("invalid")


class ReverseLLM(BaseLLM):
    def chat(self, messages):
        data = json.loads(messages[-1]["content"])
        return json.dumps({"ranked_ids": [item["chunk_id"] for item in reversed(data["candidates"])]})


class AscendingScorer:
    def predict(self, sentences, **kwargs):
        return list(range(len(sentences)))


@pytest.fixture(params=["none", "llm", "cross_encoder"])
def builtin_reranker(request):
    if request.param == "none":
        return RerankerFactory.create(Settings(rerank={"provider": "none"})), False
    if request.param == "llm":
        return LLMReranker({}, llm=ReverseLLM(), prompt_text="Rank passages."), True
    return CrossEncoderReranker({}, scorer=AscendingScorer()), True


@pytest.mark.unit
@pytest.mark.parametrize("as_objects", [False, True], ids=["mappings", "retrieval-results"])
def test_builtin_rerankers_share_sequence_and_object_preservation_contract(builtin_reranker, as_objects):
    reranker, reverses = builtin_reranker
    items = [
        {"chunk_id": f"chunk-{index}", "text": f"passage {index}", "score": .9 - index / 10,
         "metadata": {"source_path": "guide.pdf", "page": index + 1, "tags": ["rag"]}}
        for index in range(3)
    ]
    if as_objects:
        items = [RetrievalResult(**item) for item in items]
    original = tuple(items)
    snapshot = deepcopy(original)
    ranked = reranker.rerank("query", original, trace=TraceContext())
    expected = list(reversed(original)) if reverses else list(original)
    assert isinstance(ranked, list) and len(ranked) == len(original)
    assert all(actual is expected_item for actual, expected_item in zip(ranked, expected, strict=True))
    assert original == snapshot
    ranked.clear()
    assert original == snapshot


@pytest.mark.unit
def test_builtin_rerankers_accept_empty_and_singleton_tuples(builtin_reranker):
    reranker, _ = builtin_reranker
    assert reranker.rerank("query", (), trace=TraceContext()) == []
    candidate = {"chunk_id": "only", "text": "only passage", "score": .1, "metadata": {}}
    ranked = reranker.rerank("query", (candidate,))
    assert ranked == [candidate] and ranked[0] is candidate


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["", " \t"])
def test_factory_rejects_blank_provider_names(provider):
    with pytest.raises(ValueError, match=r"rerank\.provider"):
        RerankerFactory.create(Settings(rerank={"provider": provider}))


@pytest.mark.unit
def test_registration_replacement_and_snapshot_do_not_corrupt_registry(monkeypatch):
    monkeypatch.setattr(RerankerFactory, "_providers", dict(RerankerFactory._providers))
    RerankerFactory.register(" Contract ", ScoreReranker)
    RerankerFactory.register("CONTRACT", NoneReranker, replace=True)
    assert isinstance(RerankerFactory.create(Settings(rerank={"provider": "contract"})), NoneReranker)
    providers = RerankerFactory.list_providers()
    assert providers == sorted(set(providers))
    providers.clear()
    assert {"none", "llm", "cross_encoder", "contract"} <= set(RerankerFactory.list_providers())
    RerankerFactory.unregister(" CONTRACT ")
    RerankerFactory.unregister("contract")
    assert "contract" not in RerankerFactory.list_providers()


@pytest.mark.unit
def test_registration_rejects_non_callable_builder():
    with pytest.raises(TypeError, match="callable"):
        RerankerFactory.register("not-callable", None)
    assert "not-callable" not in RerankerFactory.list_providers()
