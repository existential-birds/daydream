"""Privacy-safe Daydream Harbor review agent (issue #780) tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def test_spike_task_toml_env_carries_case_key(tmp_path, fake_gh):
    """Task 0 spike: a per-case task.toml [environment].env block transmits the
    opaque case key through a real compile, stays byte-deterministic, and is
    accepted by Harbor's Task model (skip-guarded: Harbor is an optional extra)."""
    pytest.importorskip("harbor")
    from daydream.benchmark.harbor import build, package as pkg
    from tests.test_benchmark_harbor_build import _seed_ready_workspace

    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    ver = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    pkg.build_harbor(ws, wheel=wheel)      # real compile -> per-case task.toml

    case = ws / "harbor" / build.derive_task_key(case_id)
    toml = (case / "task.toml").read_text()
    key = build.derive_task_key(case_id)
    assert f"DAYDREAM_REVIEW_CASE_ID = \"{key}\"" in toml
    assert "[environment]" in toml
    # determinism: re-rendering identical bytes
    assert pkg.render_task_toml(key) == (case / "task.toml").read_bytes()
    # Harbor validates the enriched task.toml (same-interpreter Task model)
    try:
        from harbor.models.task import Task  # noqa: PLC0415 - namespace fallback as in package.py
    except ImportError:  # Harbor 0.21 wheel exposes task as a namespace package.
        from harbor.models.task.task import Task  # noqa: PLC0415
    assert Task(str(case), disable_verification=True) is not None