"""Load Settings and expose only fields intended for the overview cards."""

from pathlib import Path

from core.settings import load_settings


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "config/settings.yaml"


class ConfigService:
    def __init__(self, path: str | Path = DEFAULT_SETTINGS_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self.settings = load_settings(self.path)

    def component_configs(self) -> list[dict[str, str]]:
        """An allowlist prevents API keys and arbitrary provider settings leaking."""
        settings = self.settings
        cards = []
        for name, config in (("LLM", settings.llm), ("Embedding", settings.embedding),
                             ("Splitter", settings.splitter), ("Vector Store", settings.vector_store),
                             ("Reranker", settings.rerank), ("Evaluator", settings.evaluation)):
            enabled = config.get("enabled", True) and config.get("provider", "none") != "none"
            details = []
            for key, label in (("dimensions", "维度"), ("chunk_size", "块大小"), ("chunk_overlap", "重叠")):
                if key in config:
                    details.append(f"{label}：{config[key]}")
            cards.append({"name": name, "provider": str(config.get("provider", "未配置")),
                          "model": str(config.get("model", "")), "details": " · ".join(details),
                          "status": "已配置" if enabled else "未启用"})
        return cards
