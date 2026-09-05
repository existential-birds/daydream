"""Tests for the shared service-root discovery module.

``daydream.services`` is the single service-discovery implementation after the
move out of ``daydream/improve/services.py`` (issue #1113). This file covers the
parts that are new at package root: the explicit ``service_roots`` override the
grounded-diagram flow passes, and the shim's re-export identity. The improve
flow's own behavioral coverage stays in ``tests/test_improve_services.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.services import Service, enumerate_services


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


def test_service_field_order_is_positional_stable() -> None:
    """``Service`` is constructed positionally by existing tests, so its field
    order is load-bearing."""
    service = Service("gateway", Path("edge/gateway"), "config")
    assert (service.name, service.root, service.source) == (
        "gateway",
        Path("edge/gateway"),
        "config",
    )
