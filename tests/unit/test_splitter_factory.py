"""Unit tests for the provider-independent splitter contract and factory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class SeparatorSplitter(BaseSplitter):
    """Fake strategy that splits text using a configured separator."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        separator = str(self.config.get("separator", "|"))
        return text.split(separator)


class FixedWidthSplitter(BaseSplitter):
    """Fake strategy that divides text into deterministic fixed-width chunks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        width = int(self.config["chunk_size"])
        return [text[index : index + width] for index in range(0, len(text), width)]


@pytest.fixture
def registered_fake_providers() -> Iterator[None]:
    SplitterFactory.register("separator", SeparatorSplitter)
    SplitterFactory.register("fixed", FixedWidthSplitter)
    yield
    SplitterFactory.unregister("separator")
    SplitterFactory.unregister("fixed")


@pytest.mark.unit
def test_base_splitter_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseSplitter()


@pytest.mark.unit
def test_factory_routes_to_separator_strategy(
    registered_fake_providers: None,
) -> None:
    settings = Settings(
        splitter={"provider": "separator", "separator": "|"}
    )

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, SeparatorSplitter)
    assert splitter.split_text("alpha|beta|gamma") == ["alpha", "beta", "gamma"]


@pytest.mark.unit
def test_factory_routes_to_fixed_width_strategy(
    registered_fake_providers: None,
) -> None:
    settings = Settings(splitter={"provider": "fixed", "chunk_size": 3})

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, FixedWidthSplitter)
    assert splitter.split_text("abcdefgh") == ["abc", "def", "gh"]


@pytest.mark.unit
def test_factory_normalizes_provider_name(
    registered_fake_providers: None,
) -> None:
    settings = Settings(splitter={"provider": "  FIXED  ", "chunk_size": 2})

    assert isinstance(SplitterFactory.create(settings), FixedWidthSplitter)


@pytest.mark.unit
def test_factory_passes_a_copy_of_splitter_config(
    registered_fake_providers: None,
) -> None:
    config = {"provider": "fixed", "chunk_size": 4}
    settings = Settings(splitter=config)

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, FixedWidthSplitter)
    assert splitter.config == config
    assert splitter.config is not settings.splitter


@pytest.mark.unit
def test_factory_reports_missing_provider() -> None:
    with pytest.raises(ValueError, match=r"splitter\.provider"):
        SplitterFactory.create(Settings(splitter={"chunk_size": 1000}))


@pytest.mark.unit
def test_factory_rejects_non_string_provider() -> None:
    settings = Settings(splitter={"provider": 42})

    with pytest.raises(TypeError, match=r"splitter\.provider.*string"):
        SplitterFactory.create(settings)


@pytest.mark.unit
def test_factory_rejects_non_mapping_splitter_config() -> None:
    settings = Settings(splitter=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"section 'splitter'.*mapping"):
        SplitterFactory.create(settings)


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    settings = Settings(splitter={"provider": "missing"})

    with pytest.raises(ValueError, match="Unsupported splitter provider 'missing'"):
        SplitterFactory.create(settings)


@pytest.mark.unit
def test_duplicate_registration_requires_explicit_replace(
    registered_fake_providers: None,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        SplitterFactory.register("FIXED", FixedWidthSplitter)


@pytest.mark.unit
def test_factory_rejects_builder_with_wrong_return_type() -> None:
    SplitterFactory.register(
        "invalid", lambda config: object()  # type: ignore[return-value]
    )
    try:
        settings = Settings(splitter={"provider": "invalid"})

        with pytest.raises(TypeError, match="must return BaseSplitter"):
            SplitterFactory.create(settings)
    finally:
        SplitterFactory.unregister("invalid")
