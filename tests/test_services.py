"""Tests for the shared service-root discovery module.

``daydream.services`` is the single service-discovery implementation after the
move out of ``daydream/improve/services.py`` (issue #1113). This file covers the
parts that are new at package root: the parameterized ``owning_services``
containment predicate, the explicit ``service_roots`` override the
grounded-diagram flow passes, and the shim's re-export identity. The improve
flow's own behavioral coverage stays in ``tests/test_improve_services.py``.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

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
    edge = tmp_path / "edge" / "gateway"
    edge.mkdir(parents=True)
    (edge / "go.mod").write_text("module gateway\n")
    return tmp_path


def test_explicit_service_roots_replace_the_improve_config_list(monorepo: Path) -> None:
    """#1113: the diagram flow's own ``service_roots`` wins over the improve
    list, so a repo can scope diagram participants differently from audits."""
    cfg = DaydreamFileConfig(improve_service_roots=["apps/*"])
    services = enumerate_services(monorepo, cfg, service_roots=["edge/*"])
    assert [service.root.as_posix() for service in services] == ["edge/gateway"]
    assert [service.source for service in services] == ["config"]


def test_absent_explicit_roots_fall_back_to_the_improve_config_list(monorepo: Path) -> None:
    cfg = DaydreamFileConfig(improve_service_roots=["apps/*"])
    assert [s.root.as_posix() for s in enumerate_services(monorepo, cfg)] == [
        "apps/billing",
        "apps/catalog",
    ]
    assert [
        s.root.as_posix() for s in enumerate_services(monorepo, cfg, service_roots=None)
    ] == ["apps/billing", "apps/catalog"]


def test_empty_explicit_roots_mean_nothing_declared_not_no_services(monorepo: Path) -> None:
    """An empty list is "the caller declared nothing", so the improve list still
    applies — it must not be read as "this repo has no services"."""
    cfg = DaydreamFileConfig(improve_service_roots=["apps/*"])
    services = enumerate_services(monorepo, cfg, service_roots=[])
    assert [service.root.as_posix() for service in services] == [
        "apps/billing",
        "apps/catalog",
    ]


def test_explicit_roots_short_circuit_layout_inference(monorepo: Path) -> None:
    """Declared roots are authoritative: the conventional ``apps/*`` services
    are not appended to an explicit ``edge/*`` request."""
    services = enumerate_services(monorepo, DaydreamFileConfig(), service_roots=["edge/*"])
    assert [service.root.as_posix() for service in services] == ["edge/gateway"]


def test_explicit_roots_with_no_match_yield_no_services(monorepo: Path) -> None:
    services = enumerate_services(monorepo, DaydreamFileConfig(), service_roots=["nope/*"])
    assert services == []


def test_improve_shim_re_exports_the_same_objects() -> None:
    """The historical import path stays valid and identical, not a copy."""
    from daydream.improve import services as shim

    assert shim.enumerate_services is enumerate_services
    assert shim.Service is Service
    assert shim.__all__ == ["Service", "enumerate_services", "filter_scope"]


def test_deepest_match_skips_a_repo_root_service_and_accepts_a_path_equal_to_a_root() -> None:
    services = [
        Service("root", Path("."), "config"),
        Service("api", Path("services/api"), "config"),
        Service("inner", Path("services/api/inner"), "config"),
    ]

    def owners(path: str) -> tuple[str, ...]:
        return tuple(
            s.name
            for s in owning_services(
                path, services, match=ServiceMatch.DEEPEST, repo_root=RepoRootPolicy.SKIP, match_root_equal=True
            )
        )

    assert owners("services/api/inner/main.py") == ("inner",)
    assert owners("services/api/inner") == ("inner",)
    assert owners("scripts/tool.py") == ()
    assert owners(".") == ()


def test_first_match_in_the_callers_order_catch_alls_a_repo_root_service() -> None:
    services = sorted(
        [
            Service("root", Path("."), "config"),
            Service("api", Path("services/api"), "config"),
            Service("inner", Path("services/api/inner"), "config"),
        ],
        key=lambda service: (-len(service.root.parts), service.root.as_posix()),
    )

    def owners(path: str) -> tuple[str, ...]:
        return tuple(
            s.name
            for s in owning_services(
                path, services, match=ServiceMatch.FIRST, repo_root=RepoRootPolicy.CATCH_ALL, match_root_equal=False
            )
        )

    assert owners("services/api/inner/main.py") == ("inner",)
    assert owners("services/api/inner") == ("api",)
    assert owners("services/api") == ("root",)
    assert owners("README.md") == ("root",)


def test_all_matching_owners_keep_input_order_under_the_ordinary_repo_root_rule() -> None:
    services = [
        Service("root", Path("."), "config"),
        Service("api", Path("services/api"), "config"),
        Service("inner", Path("services/api/inner"), "config"),
    ]

    def owners(path: str) -> tuple[str, ...]:
        return tuple(
            s.name
            for s in owning_services(
                path, services, match=ServiceMatch.ALL, repo_root=RepoRootPolicy.ORDINARY, match_root_equal=True
            )
        )

    assert owners("services/api/inner/main.py") == ("api", "inner")
    assert owners(".") == ("root",)
    assert owners("README.md") == ()
    assert owners("scripts/tool.py") == ()


def test_the_repo_root_spellings_are_one_input() -> None:
    """``Path("")`` and ``Path(".")`` are the same root; the predicate decides it."""
    assert Path("").as_posix() == "."
    services = [Service("root", Path(""), "config"), Service("api", Path("services/api"), "config")]

    assert tuple(
        s.name
        for s in owning_services(
            "README.md", services, match=ServiceMatch.FIRST, repo_root=RepoRootPolicy.CATCH_ALL, match_root_equal=False
        )
    ) == ("root",)


def test_policy_arguments_carry_no_defaults() -> None:
    """M4: no caller may inherit a rule it did not state."""
    parameters = inspect.signature(owning_services).parameters
    assert list(parameters) == ["path", "services", "match", "repo_root", "match_root_equal"]
    assert all(p.default is inspect.Parameter.empty for p in parameters.values())
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in parameters.values()
        if p.name not in {"path", "services"}
    )


def test_service_field_order_is_positional_stable() -> None:
    """``Service`` is constructed positionally by existing tests, so its field
    order is load-bearing."""
    service = Service("gateway", Path("edge/gateway"), "config")
    assert (service.name, service.root, service.source) == (
        "gateway",
        Path("edge/gateway"),
        "config",
    )


# These two patterns are the same shapes the issue #1216 M7 acceptance search
# uses. Keep them in sync with the M7 command rather than treating them as
# unrelated.
_CONTAINMENT_SHAPES = (
    re.compile(r'\.startswith\(f"\{[a-z_.]*root[a-z_.]*\}/"\)'),
    re.compile(r"\b(path|file|relative|target|name|p)\s*[!=]=\s*([a-z_]+\.)?root\b"),
)


def test_no_service_containment_shape_lives_outside_services_py() -> None:
    """The acceptance search (issue #1216 M7/S1), executable.

    ``_owning_partition`` in the improve orchestrator answers the same *shape* of
    question about partitions; it is enumerated Out of Scope and is the one
    allowed exception.
    """
    repo_root = Path(__file__).resolve().parents[1]
    owner = repo_root / "daydream" / "services.py"
    offenders: list[str] = []
    for source in sorted((repo_root / "daydream").rglob("*.py")):
        if source == owner:
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not any(shape.search(line) for shape in _CONTAINMENT_SHAPES):
                continue
            if re.search(r"partition\.root\b", line):
                continue
            offenders.append(f"{source.relative_to(repo_root)}:{number}: {line.strip()}")

    assert offenders == []
