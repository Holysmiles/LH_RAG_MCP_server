"""Unit tests for the provider-independent LLM contract and factory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    """Small deterministic provider used to exercise factory behavior."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        return f"{self.config['model']}:{messages[-1]['content']}"


@pytest.fixture
def registered_fake_provider() -> Iterator[None]:
    LLMFactory.register("fake", FakeLLM)
    yield
    LLMFactory.unregister("fake")


@pytest.mark.unit
def test_base_llm_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseLLM()


@pytest.mark.unit
def test_factory_routes_to_registered_provider(
    registered_fake_provider: None,
) -> None:
    settings = Settings(llm={"provider": "fake", "model": "local-test-model"})

    llm = LLMFactory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == (
        "local-test-model:hello"
    )


@pytest.mark.unit
def test_factory_normalizes_provider_name(registered_fake_provider: None) -> None:
    settings = Settings(llm={"provider": "  FAKE  ", "model": "test-model"})

    assert isinstance(LLMFactory.create(settings), FakeLLM)


@pytest.mark.unit
def test_factory_lists_normalized_providers_in_sorted_order() -> None:
    LLMFactory.register("Zulu", FakeLLM)
    LLMFactory.register("alpha", FakeLLM)
    try:
        providers = LLMFactory.list_providers()

        assert providers == sorted(providers)
        assert {"alpha", "zulu"}.issubset(providers)
    finally:
        LLMFactory.unregister("Zulu")
        LLMFactory.unregister("alpha")


@pytest.mark.unit
def test_factory_passes_a_copy_of_provider_config(
    registered_fake_provider: None,
) -> None:
    config = {"provider": "fake", "model": "test-model", "temperature": 0.2}
    settings = Settings(llm=config)

    llm = LLMFactory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.config == config
    assert llm.config is not settings.llm


@pytest.mark.unit
def test_factory_reports_missing_provider() -> None:
    with pytest.raises(ValueError, match=r"llm\.provider"):
        LLMFactory.create(Settings(llm={"model": "test-model"}))


@pytest.mark.unit
def test_factory_rejects_non_string_provider() -> None:
    settings = Settings(llm={"provider": 42, "model": "test-model"})

    with pytest.raises(TypeError, match=r"llm\.provider.*string"):
        LLMFactory.create(settings)


@pytest.mark.unit
def test_factory_rejects_non_mapping_llm_config() -> None:
    settings = Settings(llm=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"section 'llm'.*mapping"):
        LLMFactory.create(settings)


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    settings = Settings(llm={"provider": "missing", "model": "test-model"})

    with pytest.raises(ValueError, match=r"Unsupported LLM provider 'missing'"):
        LLMFactory.create(settings)


@pytest.mark.unit
def test_duplicate_registration_requires_explicit_replace(
    registered_fake_provider: None,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        LLMFactory.register("FAKE", FakeLLM)


@pytest.mark.unit
def test_explicit_replace_updates_provider_builder(
    registered_fake_provider: None,
) -> None:
    class ReplacementLLM(FakeLLM):
        """Distinct implementation used to verify explicit replacement."""

    LLMFactory.register("FAKE", ReplacementLLM, replace=True)

    settings = Settings(llm={"provider": "fake", "model": "test-model"})

    assert isinstance(LLMFactory.create(settings), ReplacementLLM)


@pytest.mark.unit
def test_register_rejects_empty_provider_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LLMFactory.register(" ", FakeLLM)


@pytest.mark.unit
def test_register_rejects_non_callable_builder() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        LLMFactory.register("invalid", object())  # type: ignore[arg-type]


@pytest.mark.unit
def test_factory_rejects_builder_with_wrong_return_type() -> None:
    LLMFactory.register("invalid", lambda config: object())  # type: ignore[return-value]
    try:
        settings = Settings(llm={"provider": "invalid", "model": "test-model"})

        with pytest.raises(TypeError, match="must return BaseLLM"):
            LLMFactory.create(settings)
    finally:
        LLMFactory.unregister("invalid")
