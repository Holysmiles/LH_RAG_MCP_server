"""CLI startup validation does not create indexes or successful reports."""

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evaluate_cli", ROOT / "scripts/evaluate.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["missing-index", "bad-golden", "bad-top-k", "bad-settings"])
def test_startup_failures_do_not_create_data_or_leak_config(tmp_path, capsys, mode):
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    config["vector_store"]["persist_directory"] = str(tmp_path / "chroma")
    config["llm"]["api_key"] = "private-api-key"
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    golden = ROOT / "tests/fixtures/golden_test_set.json"
    if mode == "bad-golden":
        golden = tmp_path / "golden.json"
        golden.write_text('{"private-api-key":', encoding="utf-8")
    if mode == "bad-settings":
        path.write_text("llm: [private-api-key", encoding="utf-8")
    output = tmp_path / "report.json"
    args = ["--settings", str(path), "--test-set", str(golden), "--output", str(output)]
    if mode == "bad-top-k":
        args += ["--top-k", "0"]
    assert script.main(args) == 2
    stdout, stderr = capsys.readouterr()
    assert not stdout and "Cannot start evaluation" in stderr
    assert "private-api-key" not in stderr
    assert not output.exists() and not (tmp_path / "chroma").exists()
