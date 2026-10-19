"""The dashboard starts with the same interpreter and configured settings."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("start_dashboard", ROOT / "scripts/start_dashboard.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
pytestmark = pytest.mark.unit


def test_launcher_honors_settings_and_port_without_mutating_parent_environment(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["dashboard"]["port"] = 8651
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []
    before = os.environ.get("RAG_SETTINGS_PATH")
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    assert launcher.main(["--settings", str(path)]) == 0
    command, kwargs = calls[0]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[command.index("--server.port") + 1] == "8651"
    assert kwargs["env"]["RAG_SETTINGS_PATH"] == str(path)
    assert os.environ.get("RAG_SETTINGS_PATH") == before
    calls.clear()
    assert launcher.main(["--settings", str(path), "--port", "8652"]) == 0
    assert calls[0][0][calls[0][0].index("--server.port") + 1] == "8652"


@pytest.mark.parametrize("port", [0, 65536, True, "8501"])
def test_invalid_port_is_rejected_before_launch(tmp_path, port, monkeypatch):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["dashboard"]["port"] = port
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(launcher.subprocess, "call", lambda *args, **kwargs: pytest.fail("Must not launch"))
    assert launcher.main(["--settings", str(path)]) == 2
