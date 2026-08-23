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


# ---------------------------------------------------------------------------
# Task 2: candidate artifact assembly + caps + atomic write
# ---------------------------------------------------------------------------


def test_artifact_caps_fail_closed_and_write_is_atomic(tmp_path):
    from daydream.benchmark.harbor import candidate
    from daydream.benchmark.harbor import verifier_core as vc

    case_id = "case-abc123def456"
    over = [{"title": f"t{i}", "body": "b", "severity": "low",
             "path": "src/f.py", "start_line": i, "end_line": i}
            for i in range(1, vc.MAX_CANDIDATE_FINDINGS + 2)]
    with pytest.raises(candidate.CandidateError) as too_many:
        candidate.build_candidate_artifact(case_id, over)
    assert too_many.value.kind == "over_limit"

    dest = tmp_path / "logs" / "artifacts" / "review.json"
    art = candidate.build_candidate_artifact(case_id, [])
    assert art == {"schema_version": 1, "case_id": case_id,
                   "base_ref": "base", "head_ref": "head", "findings": []}
    candidate.write_candidate_artifact_atomic(dest, art)
    loaded = json.loads(dest.read_text())
    assert loaded == art and loaded["findings"] == []      # clean review round-trips
    assert vc.validate_candidate_artifact(loaded) == []     # schema-valid
    # no stray temp file is observable at the destination
    assert list(dest.parent.glob("review.json*")) == [dest]


def test_artifact_write_failure_raises(tmp_path):
    from daydream.benchmark.harbor import candidate

    dest = tmp_path / "adir"                                # a directory -> replace fails
    dest.mkdir()
    with pytest.raises(candidate.CandidateError) as write_fail:
        candidate.write_candidate_artifact_atomic(dest, {"schema_version": 1, "findings": []})
    assert write_fail.value.kind == "write_failure"


# ---------------------------------------------------------------------------
# Task 3: render_task_toml — reviewer-only host policy + case env threading
# ---------------------------------------------------------------------------


def test_render_task_toml_host_policy_and_case_env():
    from daydream.benchmark.harbor import package as pkg

    toml = pkg.render_task_toml("case-abc123def456").decode("utf-8")
    assert 'allowed_hosts = ["api.anthropic.com"]' in toml       # reviewer-only host
    assert '"github.com"' not in toml and '"huggingface.co"' not in toml
    assert 'DAYDREAM_REVIEW_CASE_ID = "case-abc123def456"' in toml
    assert 'DAYDREAM_REVIEW_BASE_REF = "base"' in toml
    assert 'DAYDREAM_REVIEW_HEAD_REF = "head"' in toml
    # deterministic
    assert pkg.render_task_toml("case-abc123def456") == pkg.render_task_toml("case-abc123def456")
    # no judge vars or archive/target config reach the agent surface
    assert "DAYDREAM_JUDGE" not in toml and "HF_TOKEN" not in toml and "GITHUB_TOKEN" not in toml


# ---------------------------------------------------------------------------
# Task 4: entrypoint — controlled in-container runner
# ---------------------------------------------------------------------------


def test_entrypoint_build_run_config_is_controlled():
    from daydream.benchmark.harbor import entrypoint
    from daydream.config_file import DaydreamFileConfig

    cfg = entrypoint.build_run_config(
        repo_dir="/workspace/repo",
        trajectory_path="/logs/agent/trajectory.json",
        backend="claude",
        model="sonnet",
    )
    assert cfg.output_mode == "review"
    assert cfg.base == "base"
    assert cfg.non_interactive is True
    assert cfg.archive is False and cfg.run_eval is False
    assert cfg.findings_out is None                     # NO --findings-out (live PR lookup)
    assert cfg.trajectory_path == Path("/logs/agent/trajectory.json")
    assert cfg.backend == "claude" and cfg.model == "sonnet"
    assert isinstance(cfg.file_config, DaydreamFileConfig)
    # controlled empty: the target repo's .daydream.toml is never loaded
    assert cfg.file_config == DaydreamFileConfig()


def test_entrypoint_backend_fail_closed(monkeypatch):
    from daydream.benchmark.harbor import entrypoint

    monkeypatch.setenv("DAYDREAM_REVIEW_BACKEND", "codex")
    with pytest.raises(entrypoint.EntrypointError) as exc:
        entrypoint.require_supported_backend()
    assert "claude" in str(exc.value)


# ---------------------------------------------------------------------------
# Task 5: entrypoint — publish candidate artifact + failure modes
# ---------------------------------------------------------------------------


def test_entrypoint_publish_failure_modes(tmp_path):
    from daydream.benchmark.harbor import candidate, entrypoint

    # missing merged output -> fail-closed, never a silent clean review
    with pytest.raises(candidate.CandidateError) as missing:
        entrypoint.publish_review(
            repo_dir=tmp_path,
            artifact_path=tmp_path / "review.json",
            case_id="case-abc123def456",
        )
    assert missing.value.kind == "missing_merged"

    # corrupt merged output (items not a list) -> fail-closed
    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True)
    (deep / "merged-items.json").write_text('{"items": "nope"}')
    with pytest.raises(candidate.CandidateError) as corrupt:
        entrypoint.publish_review(
            repo_dir=tmp_path,
            artifact_path=tmp_path / "review.json",
            case_id="case-abc123def456",
        )
    assert corrupt.value.kind == "corrupt_merged"

    # 101 parseable items -> artifact build raises over_limit
    items = [
        {"file": "src/f.py", "line": i, "description": f"d{i}",
         "rationale": "r", "severity": "low", "confidence": "LOW"}
        for i in range(1, 102)
    ]
    (deep / "merged-items.json").write_text(json.dumps({"items": items}))
    with pytest.raises(candidate.CandidateError) as over:
        entrypoint.publish_review(
            repo_dir=tmp_path,
            artifact_path=tmp_path / "review.json",
            case_id="case-abc123def456",
        )
    assert over.value.kind == "over_limit"

    # a clean review round-trips to a schema-valid empty artifact
    (deep / "merged-items.json").write_text(json.dumps({"items": []}))
    entrypoint.publish_review(
        repo_dir=tmp_path,
        artifact_path=tmp_path / "review.json",
        case_id="case-abc123def456",
    )
    loaded = json.loads((tmp_path / "review.json").read_text())
    assert loaded["findings"] == []


# ---------------------------------------------------------------------------
# Task 6: DaydreamReviewAgent — lifecycle + network-free setup
# ---------------------------------------------------------------------------


import subprocess  # noqa: E402


def test_agent_package_import_does_not_pull_harbor():
    """Importing the daydream.benchmark package must not import Harbor (a lazy,
    optional extra); ``daydream/benchmark/__init__.py`` keeps exporting only stable
    schema/service types."""
    probe = (
        "import sys; import daydream.benchmark; "
        "assert not any(m == 'harbor' or m.startswith('harbor.') for m in sys.modules), "
        "[m for m in sys.modules if m == 'harbor' or m.startswith('harbor.')]"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, f"harbor imported eagerly:\n{out.stdout}{out.stderr}"


def test_agent_lifecycle_and_lazy_harbor():
    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    assert DaydreamReviewAgent.SUPPORTS_ATIF is True
    assert callable(DaydreamReviewAgent.name) and callable(DaydreamReviewAgent.version)
    assert isinstance(DaydreamReviewAgent.name(), str) and DaydreamReviewAgent.name()
    assert isinstance(DaydreamReviewAgent.version(), str)


def test_agent_setup_confirms_version_and_backend(tmp_path):
    import pytest

    pytest.importorskip("harbor")
    from daydream.benchmark.harbor.agent import DaydreamReviewAgent
    from harbor.environments.base import ExecResult

    agent = DaydreamReviewAgent(
        logs_dir=tmp_path, extra_env={"DAYDREAM_REVIEW_BACKEND": "claude"}
    )

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            self.captured = command
            return ExecResult(return_code=0, stdout="ok", stderr="")

    env = Env()
    import asyncio

    asyncio.run(agent.setup(env))
    assert agent.version() in env.captured        # setup checks the packaged version
    assert "claude_agent_sdk" in env.captured      # and the required backend SDK


def test_agent_setup_nonzero_exec_fails(tmp_path):
    """A failed setup probe surfaces as a typed failure, never a silent pass."""
    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult

    from daydream.benchmark.harbor.agent import AgentError, DaydreamReviewAgent

    agent = DaydreamReviewAgent(logs_dir=tmp_path)

    class Env:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            self.captured = command
            return ExecResult(return_code=1, stdout="", stderr="boom")

    import asyncio

    with pytest.raises(AgentError):
        asyncio.run(agent.setup(Env()))
