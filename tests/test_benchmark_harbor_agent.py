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

# ---------------------------------------------------------------------------
# Task 1: candidate findings builder (merged items -> findings)
# ---------------------------------------------------------------------------


def test_build_candidate_findings_maps_and_skips():
    from daydream.benchmark.harbor import candidate
    from daydream.benchmark.harbor import verifier_core as vc

    items = [
        {"file": "src/cache.py", "line": 42, "description": "Cache key not tenant-scoped",
         "rationale": "Key can collide across tenants.", "severity": "high", "confidence": "HIGH"},
        {"file": "", "line": 1, "description": "empty-file item is skipped", "rationale": "", "severity": "low"},
        {"file": "src/a.py", "line": 3, "description": "Dup A", "rationale": "r", "severity": "medium", "confidence": "LOW"},
        {"file": "src/a.py", "line": 9, "description": "Dup A", "rationale": "r", "severity": "medium", "confidence": "LOW"},
    ]
    case_id = "case-abc123def456"
    findings = candidate.build_candidate_findings(items, case_id=case_id)

    assert len(findings) == 3                       # empty-file item skipped
    assert [f["title"] for f in findings] == ["Cache key not tenant-scoped", "Dup A", "Dup A"]
    f0 = findings[0]
    assert f0["path"] == "src/cache.py" and f0["start_line"] == 42 and f0["end_line"] == 42
    assert f0["severity"] == "high" and "Key can collide" in f0["body"]
    # byte-for-byte candidate-id contract incl. duplicate ordinals: the verifier
    # re-derives every id and would raise on any drift; the parsed findings must
    # carry the exact ids the builder derived.
    parsed = vc.validate_candidate_artifact(
        {"schema_version": 1, "case_id": case_id, "base_ref": "base", "head_ref": "head",
         "findings": findings}
    )
    assert [p.candidate_id for p in parsed] == [f["candidate_id"] for f in findings]
    assert findings[1]["candidate_id"] != findings[2]["candidate_id"]
