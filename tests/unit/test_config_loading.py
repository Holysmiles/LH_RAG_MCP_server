from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import main as application
from core.settings import Settings, SettingsError, load_settings, validate_settings


VALID_CONFIG = {
    "llm": {"provider": "openai", "model": "gpt-4o-mini"},
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "${TEST_OPENAI_API_KEY}",
    },
    "splitter": {
        "provider": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "vector_store": {"provider": "chroma"},
    "retrieval": {
        "dense_top_k": 20,
        "sparse_top_k": 20,
        "fusion_top_k": 10,
    },
    "rerank": {"provider": "none"},
    "evaluation": {"provider": "custom"},
    "observability": {"log_level": "INFO"},
}


def write_config(tmp_path: Path, config: object = VALID_CONFIG) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.unit
def test_load_settings_returns_structured_settings(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path))

    assert isinstance(settings, Settings)
    assert settings.llm["provider"] == "openai"
    assert settings.splitter["chunk_size"] == 1000
    assert settings.retrieval["fusion_top_k"] == 10


@pytest.mark.unit
def test_load_settings_expands_available_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-secret")

    settings = load_settings(write_config(tmp_path))

    assert settings.embedding["api_key"] == "test-secret"


@pytest.mark.unit
def test_load_settings_preserves_unset_environment_placeholders(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path))

    assert settings.embedding["api_key"] == "${TEST_OPENAI_API_KEY}"


@pytest.mark.unit
def test_load_settings_reports_missing_nested_field(tmp_path: Path) -> None:
    config = deepcopy(VALID_CONFIG)
    del config["embedding"]["provider"]

    with pytest.raises(SettingsError, match=r"embedding\.provider"):
        load_settings(write_config(tmp_path, config))


@pytest.mark.unit
def test_load_settings_rejects_non_mapping_section(tmp_path: Path) -> None:
    config = deepcopy(VALID_CONFIG)
    config["embedding"] = "openai"

    with pytest.raises(SettingsError, match=r"embedding.*mapping"):
        load_settings(write_config(tmp_path, config))


@pytest.mark.unit
def test_load_settings_rejects_empty_document(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="top level"):
        load_settings(write_config(tmp_path, None))


@pytest.mark.unit
def test_load_settings_reports_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("llm: [unterminated", encoding="utf-8")

    with pytest.raises(SettingsError, match="Invalid YAML"):
        load_settings(path)


@pytest.mark.unit
def test_validate_settings_reports_empty_required_value() -> None:
    settings = Settings(**deepcopy(VALID_CONFIG))
    settings.llm["provider"] = "  "

    with pytest.raises(SettingsError, match=r"llm\.provider"):
        validate_settings(settings)


@pytest.mark.unit
def test_main_loads_valid_settings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = application.main(write_config(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Modular RAG MCP Server is ready." in captured.out


@pytest.mark.unit
def test_main_fails_fast_for_invalid_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = deepcopy(VALID_CONFIG)
    del config["embedding"]["provider"]

    exit_code = application.main(write_config(tmp_path, config))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "embedding.provider" in captured.err
