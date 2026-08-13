"""C5 使用项目真实 LLM 配置的集成验收。

运行前必须设置 ``config/settings.yaml`` 对应的 API key/endpoint 环境变量。
测试会产生真实模型调用和可能的费用；没有凭据时明确 skip，不把 Fake 当成验收。
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from core import Chunk
from core.settings import Settings, load_settings
from ingestion.transform import ChunkRefiner
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory


LOCAL_PROVIDER = "c5-local-transformers"


class LocalTransformersLLM(BaseLLM):
    """仅供集成验收使用的真实本地生成模型 Provider。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = str(config["model"])
        self.max_new_tokens = int(config.get("max_new_tokens", 80))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
        )

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        return self.tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()


@pytest.fixture(autouse=True)
def local_provider() -> Iterator[None]:
    """注册测试专用 Provider，不污染生产 LLM 注册表。"""
    LLMFactory.register(LOCAL_PROVIDER, LocalTransformersLLM)
    yield
    LLMFactory.unregister(LOCAL_PROVIDER)


def _real_llm_settings() -> Settings:
    local_model = os.environ.get("C5_LOCAL_MODEL_PATH")
    if local_model:
        model_path = Path(local_model)
        if not model_path.is_dir():
            pytest.skip(f"C5_LOCAL_MODEL_PATH is not a directory: {model_path}")
        return Settings(
            llm={
                "provider": LOCAL_PROVIDER,
                "model": str(model_path),
                "max_new_tokens": 80,
            },
            ingestion={"chunk_refiner": {"use_llm": True}},
        )

    settings = load_settings(Path("config/settings.yaml"))
    provider = str(settings.llm.get("provider", "")).lower()
    required_env = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
    }.get(provider)
    if required_env and not os.environ.get(required_env):
        pytest.skip(f"real LLM integration requires {required_env}")
    if provider == "ollama" and not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("real Ollama integration requires OLLAMA_BASE_URL")
    settings.ingestion["chunk_refiner"] = {"use_llm": True}
    return settings


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="integration-chunk",
        text=text,
        metadata={"source_path": "integration/noisy.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="integration-document",
    )


@pytest.mark.integration
def test_configured_real_llm_refines_without_losing_facts() -> None:
    noisy = "Page 1 of 1\n\nRAG    combines retrieval with generation.\n---"

    refined = ChunkRefiner(_real_llm_settings()).transform([_chunk(noisy)])[0]

    assert refined.metadata["refined_by"] == "llm"
    assert "RAG" in refined.text
    assert "retrieval" in refined.text.lower()
    assert "generation" in refined.text.lower()
    assert "Page 1 of 1" not in refined.text


@pytest.mark.integration
def test_invalid_real_model_configuration_falls_back_to_rules() -> None:
    settings = _real_llm_settings()
    invalid_model = "model-that-does-not-exist-c5"
    settings.llm["model"] = invalid_model
    if str(settings.llm.get("provider", "")).lower() == "azure":
        # Azure 的请求 URL 使用 deployment_name，而不是通用 model 字段。
        settings.llm["deployment_name"] = invalid_model
    settings.llm["timeout"] = 10
    noisy = "Page 1 of 1\nUseful    facts"

    refined = ChunkRefiner(settings).transform([_chunk(noisy)])[0]

    assert refined.text == "Useful facts"
    assert refined.metadata["refined_by"] == "rule"
    assert refined.metadata["fallback_reason"].startswith(
        ("llm_initialization_failed:", "llm_call_failed:")
    )
