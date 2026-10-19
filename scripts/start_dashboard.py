"""Start the dashboard with this interpreter and the configured port."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.settings import SettingsError, load_settings
from observability.logger import get_logger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start the RAG Streamlit dashboard.")
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "config/settings.yaml")
    parser.add_argument("--port", type=int, help="Override dashboard.port.")
    args = parser.parse_args(argv)
    logger = get_logger("scripts.start_dashboard")
    try:
        settings_path = args.settings.expanduser().resolve()
        settings = load_settings(settings_path)
        if settings.dashboard.get("enabled", True) is not True:
            raise ValueError("dashboard.enabled must be true")
        port = args.port if args.port is not None else settings.dashboard.get("port", 8501)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("dashboard.port must be an integer from 1 to 65535")
    except (OSError, SettingsError, ValueError):
        logger.error("Cannot start dashboard: check settings, dashboard.enabled and dashboard.port")
        return 2
    command = [sys.executable, "-m", "streamlit", "run",
               str(PROJECT_ROOT / "src/observability/dashboard/app.py"),
               "--server.port", str(port), "--server.address", "127.0.0.1",
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    try:
        return subprocess.call(command, cwd=PROJECT_ROOT,
                               env=dict(os.environ, RAG_SETTINGS_PATH=str(settings_path)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
