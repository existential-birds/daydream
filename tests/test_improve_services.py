"""Tests for improve-flow service enumeration."""

from pathlib import Path

from daydream.config_file import DaydreamFileConfig
from daydream.services import enumerate_services


def test_manifestless_directory_under_conventional_root_stays_undiscovered(tmp_path: Path) -> None:
    (tmp_path / "apps" / "docs-only").mkdir(parents=True)
    (tmp_path / "apps" / "docs-only" / "README.md").write_text("# docs\n")
    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []
