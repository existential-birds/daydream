"""Packaging and Harbor 0.25 integration tests for compiled benchmarks."""
import hashlib
import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import package as pkg
from daydream.benchmark.harbor.build import TEMPLATE_VERSION
from tests.test_benchmark_harbor_build import _stub_wheel

# The runtime-requirements lock is committed, not generated at package time: nothing in the
# production packaging path runs `uv export`. These helpers ARE that regeneration recipe, so
# they live with the only test that uses them and with the committed lock they must reproduce.
_GENERATION_COMMAND = "uv export --frozen --no-dev --no-emit-project --format requirements-txt"


def _strip_uv_header(text: str) -> str:
    """Strip uv's leading generated-comment block while preserving requirement comments."""
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines) and (lines[index].startswith("#") or not lines[index].strip()):
        index += 1
    return "".join(lines[index:])


def _uv_export_body(uv_lock_path: Path) -> str:
    """Export hash-pinned runtime requirements from *uv_lock_path*."""
    command = _GENERATION_COMMAND.split()
    try:
        result = subprocess.run(
            command, cwd=uv_lock_path.resolve().parent, capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise pkg.PackageError(f"cannot run `{_GENERATION_COMMAND}`: {exc}") from exc
    if result.returncode != 0:
        raise pkg.PackageError(
            f"`{_GENERATION_COMMAND}` failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    body = _strip_uv_header(result.stdout)
    if "--hash=sha256:" not in body:
        raise pkg.PackageError(f"`{_GENERATION_COMMAND}` did not produce hash-pinned requirements")
    return body


def _render_runtime_lock(uv_lock_path: Path, *, daydream_version: str) -> tuple[str, str]:
    """Return the deterministic header and exported requirements body."""
    source_sha256 = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    body = _uv_export_body(uv_lock_path)
    if f"daydream=={daydream_version}" in body:
        raise pkg.PackageError("runtime lock unexpectedly includes the Daydream project")
    header = (
        "# Daydream Harbor runtime requirements (generated; do not edit)\n"
        f"# daydream_version: {daydream_version}\n"
        f"# source_uv_lock_sha256: {source_sha256}\n"
        f"# generation_command: {_GENERATION_COMMAND}\n"
        f"# template_version: {TEMPLATE_VERSION}\n"
        "\n"
    )
    return header, body


def test_runtime_lock_regeneration_is_noop_on_unchanged(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    uv_lock = repo / "uv.lock"
    uv_lock.write_bytes((root / "uv.lock").read_bytes())
    (repo / "pyproject.toml").write_bytes((root / "pyproject.toml").read_bytes())
    ver = importlib.metadata.version("daydream")
    committed = (root / "daydream/benchmark/harbor/runtime-requirements.lock").read_bytes()
    # Byte-identity is the whole header contract: every field, in order, plus the pinned body.
    assert "".join(_render_runtime_lock(uv_lock, daydream_version=ver)).encode() == committed
    uv_lock.write_bytes(uv_lock.read_bytes() + b"\n# drift\n")
    assert "".join(_render_runtime_lock(uv_lock, daydream_version=ver)).encode() != committed


def test_validate_wheel_accepts_matching_and_rejects_mismatch(tmp_path: Path) -> None:
    good, ver = _stub_wheel(tmp_path)
    info = pkg.validate_wheel(good, daydream_version=ver)
    assert info.distribution == "daydream" and info.version == ver
    assert len(info.sha256) == 64

    bad = tmp_path / "daydream-0.99.0-py3-none-any.whl"
    bad.write_bytes(b"x")
    with pytest.raises(pkg.PackageError) as mismatch:
        pkg.validate_wheel(bad, daydream_version=ver)
    assert bad.name in str(mismatch.value)
    assert "daydream-" + ver in str(mismatch.value)

    with pytest.raises(pkg.PackageError) as missing:
        pkg.validate_wheel(tmp_path / "absent.whl", daydream_version=ver)
    assert "absent.whl" in str(missing.value)

def test_resolve_harbor_checks_same_interpreter_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("harbor")

    exe = pkg.resolve_harbor()
    assert exe == str(Path(sys.executable).parent / "harbor")

    def absent(distribution: Any) -> None:
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    with pytest.raises(pkg.PackageError) as missing:
        pkg.resolve_harbor()
    assert "pip install 'daydream[benchmark]'" in str(missing.value)

    for unsupported in ("0.24.0", "0.26.0", "invalid"):
        monkeypatch.setattr(importlib.metadata, "version", lambda d, version=unsupported: version)
        with pytest.raises(pkg.PackageError) as wrong:
            pkg.resolve_harbor()
        assert "[0.25, 0.26)" in str(wrong.value)

def test_render_task_toml_normalizes_hosts_and_fails_closed() -> None:
    doc = tomllib.loads(pkg.render_task_toml(
        "case-abc", reviewer_hosts=["HTTPS://B.Example.com", "a.example.com"], judge_hosts=["openrouter.ai"],
    ).decode())
    assert doc["metadata"] == {"benchmark_case_key": "case-abc", "source_kind": "historic-github-pr"}
    # Network boundary: offline reviewer image, allowlisted agent, separate allowlisted verifier.
    assert doc["agent"] == {"timeout_sec": 1800.0, "network_mode": "allowlist",
                            "allowed_hosts": ["a.example.com", "b.example.com"]}
    assert doc["environment"] == {
        "network_mode": "no-network", "build_timeout_sec": 1200.0, "workdir": "/workspace/repo", "cpus": 2,
        "memory_mb": 4096, "storage_mb": 10240,
        "env": {"DAYDREAM_REVIEW_CASE_ID": "case-abc", "DAYDREAM_REVIEW_BASE_REF": "base",
                "DAYDREAM_REVIEW_HEAD_REF": "head"},
    }
    assert doc["verifier"] == {"timeout_sec": 900.0, "environment_mode": "separate", "environment": {
        "network_mode": "allowlist", "allowed_hosts": ["openrouter.ai"], "build_timeout_sec": 1200.0, "cpus": 1,
        "memory_mb": 2048, "storage_mb": 4096}}
    with pytest.raises(pkg.PackageError):
        pkg.render_task_toml("case-abc", reviewer_hosts=["not-a-host"], judge_hosts=["openrouter.ai"])
    with pytest.raises(pkg.PackageError):
        pkg.render_task_toml("case-abc", reviewer_hosts=["api.anthropic.com"], judge_hosts=["*.example.com"])
    bad_cases: tuple[dict[str, Any], ...] = (
        {"reviewer_hosts": [], "judge_hosts": ["h.example"]}, {"reviewer_hosts": ["h.example"], "judge_hosts": []},
        {"reviewer_hosts": [], "judge_hosts": []},
    )
    for kw in bad_cases:
        with pytest.raises(pkg.PackageError):
            pkg.render_task_toml("case-abc", **kw)
    with pytest.raises(pkg.PackageError):
        pkg.render_task_toml("case-abc")   # missing required kwargs


def test_validate_compiled_rejects_missing_harbor_with_remediation(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(distribution: Any) -> None:
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.validate_compiled(None)
    assert "pip install 'daydream[benchmark]'" in str(rejected.value)
