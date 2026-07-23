from importlib import import_module

import pytest


TOP_LEVEL_PACKAGES = (
    "mcp_server",
    "core",
    "ingestion",
    "libs",
    "observability",
)


@pytest.mark.unit
@pytest.mark.parametrize("package_name", TOP_LEVEL_PACKAGES)
def test_top_level_package_is_importable(package_name: str) -> None:
    assert import_module(package_name).__name__ == package_name
