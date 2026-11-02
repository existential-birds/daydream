"""Shared service discovery, ownership, and explicit service_roots contracts."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.services import (
    enumerate_services,
    owning_services,
)


@pytest.fixture
def monorepo(tmp_path: Path) -> Path:
    """Two conventional services plus a third that no heuristic root covers."""
    for service in ("billing", "catalog"):
        root = tmp_path / "apps" / service
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f"[project]\nname='{service}'\n")
    docs = tmp_path / "apps/docs-only"
    docs.mkdir()
    (docs / "README.md").write_text("# docs\n")
    edge = tmp_path / "edge" / "gateway"
    edge.mkdir(parents=True)
    (edge / "go.mod").write_text("module gateway\n")
    return tmp_path

def test_explicit_service_roots_replace_the_improve_config_list(monorepo: Path) -> None:
    """#1113: the diagram flow's own ``service_roots`` wins over the improve
    list, so a repo can scope diagram participants differently from audits."""
    inferred = enumerate_services(monorepo, DaydreamFileConfig())
    assert [(service.root.as_posix(), service.source) for service in inferred] == [
        ("apps/billing", "heuristic:conventional-root"), ("apps/catalog", "heuristic:conventional-root"),
    ]
    cfg = DaydreamFileConfig(improve_service_roots=["apps/*"])
    services = enumerate_services(monorepo, cfg, service_roots=["edge/*"])
    assert [service.root.as_posix() for service in services] == ["edge/gateway"]
    assert [service.source for service in services] == ["config"]


def test_policy_arguments_carry_no_defaults() -> None:
    """M4: no caller may inherit a rule it did not state."""
    parameters = inspect.signature(owning_services).parameters
    assert list(parameters) == ["path", "services", "match", "repo_root", "match_root_equal"]
    assert all(p.default is inspect.Parameter.empty for p in parameters.values())
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in parameters.values()
        if p.name not in {"path", "services"}
    )
