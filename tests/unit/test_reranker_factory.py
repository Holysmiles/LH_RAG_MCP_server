"""Unit tests for reranker contracts, factory routing, and safe fallback."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, CandidateT
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
