"""Executable entry point for the Modular RAG MCP Server."""

from pathlib import Path

from core.settings import SettingsError, load_settings
from observability.logger import get_logger

PROJECT_NAME = "Modular RAG MCP Server"
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "config" / "settings.yaml"


def main(settings_path: str | Path | None = None) -> int:
    """Load and validate configuration before starting the application."""
    path = settings_path or DEFAULT_SETTINGS_PATH

    try:
        settings = load_settings(path)
    except (OSError, SettingsError) as exc:
        get_logger(__name__).error("Failed to load settings: %s", exc)
        return 1

    logger = get_logger(__name__, settings.observability["log_level"])
    logger.info("Configuration loaded from %s", path)
    print(f"{PROJECT_NAME} is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
