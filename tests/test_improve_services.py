"""Tests for improve-flow service enumeration."""

from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.improve.services import enumerate_services, filter_scope


@pytest.fixture
def tmp_path_repo(tmp_path: Path) -> Path:
    """Build the customer-style ``apps/<service>`` monorepo layout."""
    for service in ("billing", "catalog"):
        root = tmp_path / "apps" / service
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f"[project]\nname='{service}'\n")
    return tmp_path


def test_config_declared_roots_win(tmp_path_repo: Path) -> None:
    cfg = DaydreamFileConfig(improve_service_roots=["apps/*"])
    services = enumerate_services(tmp_path_repo, cfg)
    assert [s.name for s in services] == ["billing", "catalog"]
    assert services[0].root == Path("apps/billing")


def test_heuristics_detect_manifest_under_conventional_root(tmp_path_repo: Path) -> None:
    # no config: apps/billing + apps/catalog each hold their own pyproject.toml
    services = enumerate_services(tmp_path_repo, DaydreamFileConfig())
    assert {s.name for s in services} == {"billing", "catalog"}


def test_single_package_repo_yields_no_services(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='solo'\n")
    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []


def test_absent_conventional_roots_yield_no_services(tmp_path: Path) -> None:
    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []


def test_scope_filters_search_not_read(tmp_path_repo: Path) -> None:
    services = enumerate_services(tmp_path_repo, DaydreamFileConfig())
    scoped = filter_scope(services, "apps/billing")
    assert [s.name for s in scoped] == ["billing"]
