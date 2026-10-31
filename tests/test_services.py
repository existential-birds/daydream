"""Shared service discovery, ownership, and explicit service_roots contracts."""

from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Sequence

import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.services import (
    RepoRootPolicy,
    Service,
    ServiceMatch,
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



def test_explicit_roots_with_no_match_yield_no_services(monorepo: Path) -> None:
    services = enumerate_services(monorepo, DaydreamFileConfig(), service_roots=["nope/*"])
    assert services == []

#: One root/api/inner ownership matrix. A tuple so the shared value cannot be
#: mutated by a caller that sorts or filters it.
_NESTED_SERVICES: tuple[Service, ...] = (
    Service("root", Path("."), "config"),
    Service("api", Path("services/api"), "config"),
    Service("inner", Path("services/api/inner"), "config"),
)


def _owners(
    path: str, services: Sequence[Service], *,
    match: ServiceMatch, repo_root: RepoRootPolicy, match_root_equal: bool,
) -> tuple[str, ...]:
    """Owning service names for ``path`` under one explicitly stated policy triple."""
    return tuple(service.name
        for service in owning_services(
            path, services, match=match, repo_root=repo_root, match_root_equal=match_root_equal
        )
    )


def test_deepest_match_skips_a_repo_root_service_and_accepts_a_path_equal_to_a_root() -> None:
    owners = partial(_owners, services=_NESTED_SERVICES, match=ServiceMatch.DEEPEST,
                     repo_root=RepoRootPolicy.SKIP, match_root_equal=True)
    assert owners("services/api/inner/main.py") == ("inner",)
    assert owners("services/api/inner") == ("inner",)
    assert owners("scripts/tool.py") == ()
    assert owners(".") == ()

def test_first_match_in_the_callers_order_catch_alls_a_repo_root_service() -> None:
    services = sorted(_NESTED_SERVICES, key=lambda service: (-len(service.root.parts), service.root.as_posix()))
    owners = partial(_owners, services=services, match=ServiceMatch.FIRST,
                     repo_root=RepoRootPolicy.CATCH_ALL, match_root_equal=False)
    assert owners("services/api/inner/main.py") == ("inner",)
    assert owners("services/api/inner") == ("api",)
    assert owners("services/api") == ("root",)
    assert owners("README.md") == ("root",)

def test_all_matching_owners_keep_input_order_under_the_ordinary_repo_root_rule() -> None:
    owners = partial(_owners, services=_NESTED_SERVICES, match=ServiceMatch.ALL,
                     repo_root=RepoRootPolicy.ORDINARY, match_root_equal=True)
    assert owners("services/api/inner/main.py") == ("api", "inner")
    assert owners(".") == ("root",)
    assert owners("README.md") == ()
    assert owners("scripts/tool.py") == ()


def test_policy_arguments_carry_no_defaults() -> None:
    """M4: no caller may inherit a rule it did not state."""
    parameters = inspect.signature(owning_services).parameters
    assert list(parameters) == ["path", "services", "match", "repo_root", "match_root_equal"]
    assert all(p.default is inspect.Parameter.empty for p in parameters.values())
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in parameters.values()
        if p.name not in {"path", "services"}
    )
