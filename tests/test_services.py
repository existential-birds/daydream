"""Shared service ownership and explicit service_roots override contracts.

Improve discovery behavior is covered separately in test_improve_services.py.
"""

from __future__ import annotations

import ast
import inspect
import re
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



def test_explicit_roots_short_circuit_layout_inference(monorepo: Path) -> None:
    """Declared roots are authoritative: the conventional ``apps/*`` services
    are not appended to an explicit ``edge/*`` request."""
    services = enumerate_services(monorepo, DaydreamFileConfig(), service_roots=["edge/*"])
    assert [service.root.as_posix() for service in services] == ["edge/gateway"]

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



# The first two patterns are the shapes the issue #1216 M7 acceptance search
# uses; keep them in sync with the M7 command rather than treating them as
# unrelated. The third is a superset that also catches a `startswith` arguing on
# `root` without the exact f-string spelling (`path.startswith(service.root...)`),
# which the M7 pair alone would let through.
_CONTAINMENT_SHAPES = (
    re.compile(r'\.startswith\(f"\{[^}]*root[^}]*\}/"\)'),
    re.compile(r"\b(path|file|relative|target|name|p)\s*[!=]=\s*([a-z_]+\.)?root\b"),
    re.compile(r"\.startswith\([^)]*\broot\b"),
)


# ``_owning_partition`` in Improve's audit scope answers the same *shape* of
# question about partitions. It is enumerated Out of Scope and is the one allowed
# exception, so the exemption is scoped to that function's module and name rather
# than to any line that happens to mention ``partition.root``.
_PARTITION_EXCEPTION = Path("daydream") / "improve" / "audit_scope.py"


def _function_line_span(path: Path, name: str) -> set[int]:
    """The inclusive source lines occupied by the named function, so an
    exemption names the function rather than any line that mentions its state."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return set()

def test_no_service_containment_shape_lives_outside_services_py() -> None:
    """The acceptance search (issue #1216 M7/S1), executable.

    ``_owning_partition`` in Improve's audit scope answers the same *shape* of question about partitions; it is
    enumerated Out of Scope and is the one allowed exception."""
    repo_root = Path(__file__).resolve().parents[1]
    owner = repo_root / "daydream" / "services.py"
    partition_exception = repo_root / _PARTITION_EXCEPTION
    partition_exempt_lines = _function_line_span(partition_exception, "_owning_partition")
    offenders: list[str] = []
    for source in sorted((repo_root / "daydream").rglob("*.py")):
        if source == owner:
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not any(shape.search(line) for shape in _CONTAINMENT_SHAPES):
                continue
            if source == partition_exception and number in partition_exempt_lines:
                continue
            offenders.append(f"{source.relative_to(repo_root)}:{number}: {line.strip()}")

    assert offenders == []
