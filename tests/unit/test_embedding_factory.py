"""Unit tests for the provider-independent embedding contract and factory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Deterministic provider used to exercise batch embedding and routing."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        dimension = int(self.config.get("dimension", 2))
        return [
            [float(len(text)), float(index)] + [0.0] * (dimension - 2)
            for index, text in enumerate(texts)
        ]


@pytest.fixture
def registered_fake_provider() -> Iterator[None]:
    EmbeddingFactory.register("fake", FakeEmbedding)
    yield
    EmbeddingFactory.unregister("fake")


@pytest.mark.unit
def test_base_embedding_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseEmbedding()


@pytest.mark.unit
def test_factory_routes_batch_to_registered_provider(
    registered_fake_provider: None,
) -> None:
    settings = Settings(
        embedding={"provider": "fake", "model": "test-model", "dimension": 3}
    )

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.embed(["one", "three"]) == [
        [3.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
    ]


@pytest.mark.unit
def test_fake_embedding_returns_stable_vectors(
    registered_fake_provider: None,
) -> None:
    embedding = EmbeddingFactory.create(
        Settings(embedding={"provider": "fake", "model": "test-model"})
    )

    first = embedding.embed(["stable", "batch"])
    second = embedding.embed(["stable", "batch"])

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 2 for vector in first)


@pytest.mark.unit
def test_factory_normalizes_provider_name(registered_fake_provider: None) -> None:
    settings = Settings(embedding={"provider": "  FAKE  ", "model": "test-model"})

    assert isinstance(EmbeddingFactory.create(settings), FakeEmbedding)


@pytest.mark.unit
def test_factory_passes_a_copy_of_provider_config(
    registered_fake_provider: None,
) -> None:
    config = {"provider": "fake", "model": "test-model", "dimension": 4}
    settings = Settings(embedding=config)

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.config == config
    assert embedding.config is not settings.embedding


@pytest.mark.unit
def test_factory_reports_missing_provider() -> None:
    with pytest.raises(ValueError, match=r"embedding\.provider"):
        EmbeddingFactory.create(Settings(embedding={"model": "test-model"}))


@pytest.mark.unit
def test_factory_rejects_non_string_provider() -> None:
    settings = Settings(embedding={"provider": 42, "model": "test-model"})

    with pytest.raises(TypeError, match=r"embedding\.provider.*string"):
        EmbeddingFactory.create(settings)


@pytest.mark.unit
def test_factory_rejects_non_mapping_embedding_config() -> None:
    settings = Settings(embedding=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"section 'embedding'.*mapping"):
        EmbeddingFactory.create(settings)


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    settings = Settings(embedding={"provider": "missing", "model": "test-model"})

    with pytest.raises(ValueError, match="Unsupported embedding provider 'missing'"):
        EmbeddingFactory.create(settings)


@pytest.mark.unit
def test_duplicate_registration_requires_explicit_replace(
    registered_fake_provider: None,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        EmbeddingFactory.register("FAKE", FakeEmbedding)


@pytest.mark.unit
def test_factory_rejects_builder_with_wrong_return_type() -> None:
    EmbeddingFactory.register(
        "invalid", lambda config: object()  # type: ignore[return-value]
    )
    try:
        settings = Settings(
            embedding={"provider": "invalid", "model": "test-model"}
        )

        with pytest.raises(TypeError, match="must return BaseEmbedding"):
            EmbeddingFactory.create(settings)
    finally:
        EmbeddingFactory.unregister("invalid")
