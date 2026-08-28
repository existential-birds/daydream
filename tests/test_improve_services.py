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


@pytest.mark.parametrize(
    "manifest",
    ["requirements.txt", "requirements.in", "setup.py", "setup.cfg", "tox.ini"],
)
def test_pre_pep621_python_service_is_discovered(tmp_path: Path, manifest: str) -> None:
    root = tmp_path / "apps" / "ledger"
    root.mkdir(parents=True)
    (root / manifest).write_text("httpx==0.27.0\n")

    services = enumerate_services(tmp_path, DaydreamFileConfig())

    assert [(s.name, s.root.as_posix()) for s in services] == [("ledger", "apps/ledger")]
    assert filter_scope(services, "apps/ledger")[0].name == "ledger"


def test_manifestless_directory_under_conventional_root_stays_undiscovered(tmp_path: Path) -> None:
    (tmp_path / "apps" / "docs-only").mkdir(parents=True)
    (tmp_path / "apps" / "docs-only" / "README.md").write_text("# docs\n")

    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []


def test_single_package_repo_yields_no_services(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='solo'\n")
    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []


def test_absent_conventional_roots_yield_no_services(tmp_path: Path) -> None:
    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []


def test_scope_filters_search_not_read(tmp_path_repo: Path) -> None:
    services = enumerate_services(tmp_path_repo, DaydreamFileConfig())
    scoped = filter_scope(services, "apps/billing")
    assert [s.name for s in scoped] == ["billing"]
