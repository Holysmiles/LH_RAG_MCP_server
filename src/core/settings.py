"""Configuration loading and fail-fast validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import re
from typing import Any

import yaml


class SettingsError(ValueError):
    """Raised when the project configuration cannot be parsed or validated."""


@dataclass(slots=True)
class Settings:
    """Structured project settings without initializing external services."""

    project: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    # Vision 模型独立于文字 LLM：没有图片任务时保持空字典，不会创建外部服务。
    # 启用 C7/B9 后，这里保存 provider、model、endpoint 和图片尺寸等配置。
    vision_llm: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    splitter: dict[str, Any] = field(default_factory=dict)
    vector_store: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    dashboard: dict[str, Any] = field(default_factory=dict)
    ingestion: dict[str, Any] = field(default_factory=dict)


REQUIRED_FIELDS = (
    "llm.provider",
    "llm.model",
    "embedding.provider",
    "embedding.model",
    "splitter.provider",
    "vector_store.provider",
    "retrieval.dense_top_k",
    "retrieval.sparse_top_k",
    "retrieval.fusion_top_k",
    "rerank.provider",
    "evaluation.provider",
    "observability.log_level",
)

_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_settings(path: str | Path) -> Settings:
    """Read a YAML file, expand available environment variables, and validate it."""
    settings_path = Path(path)

    try:
        raw_config = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML in settings file '{settings_path}': {exc}") from exc

    if not isinstance(raw_config, Mapping):
        raise SettingsError("Settings file must contain a YAML mapping at the top level")

    expanded_config = _expand_environment_variables(raw_config)
    settings_values: dict[str, dict[str, Any]] = {}

    for settings_field in fields(Settings):
        section_name = settings_field.name
        section = expanded_config.get(section_name, {})
        if not isinstance(section, Mapping):
            raise SettingsError(f"Configuration field '{section_name}' must be a mapping")
        settings_values[section_name] = dict(section)

    settings = Settings(**settings_values)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Ensure every required dotted field exists and has a non-empty value."""
    for field_path in REQUIRED_FIELDS:
        section_name, field_name = field_path.split(".", maxsplit=1)
        section = getattr(settings, section_name)
        if field_path == "evaluation.provider" and "backends" in section:
            backends = section["backends"]
            if not isinstance(backends, (list, tuple)) or not backends:
                raise SettingsError("evaluation.backends must be a nonempty list of providers")
            continue
        value = section.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise SettingsError(f"Missing required configuration field: {field_path}")


def _expand_environment_variables(value: Any) -> Any:
    if isinstance(value, str):
        return _ENVIRONMENT_VARIABLE.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)), value
        )
    if isinstance(value, Mapping):
        return {
            key: _expand_environment_variables(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return value
