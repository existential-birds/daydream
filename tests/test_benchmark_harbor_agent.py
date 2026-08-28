"""Privacy-safe Daydream Harbor review agent (issue #780) tests."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.harness.fake_gh import FakeGh


def test_spike_task_toml_env_carries_case_key(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Task 0 spike: a per-case task.toml [environment].env block transmits the
    opaque case key through a real compile, stays byte-deterministic, and is
    accepted by Harbor's Task model (skip-guarded: Harbor is an optional extra)."""
    pytest.importorskip("harbor")
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import package as pkg
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
    # re-render with the same workspace-seeded privacy allowlists so the bytes
    # are identical to the compiled task.toml (reviewer h1/judge h2, not a
    # different reviewer policy)
    assert pkg.render_task_toml(
        key, reviewer_hosts=["h1.example.com"], judge_hosts=["h2.example.com"]
    ) == (case / "task.toml").read_bytes()
    # Harbor validates the enriched task.toml (same-interpreter Task model)
    try:
        from harbor.models.task import Task  # noqa: PLC0415 - namespace fallback as in package.py
    except ImportError:  # Harbor exposes task as a namespace package in some wheels.
        from harbor.models.task.task import Task  # noqa: PLC0415
    assert Task(str(case), disable_verification=True) is not None

# ---------------------------------------------------------------------------
# Task 1: candidate findings builder (merged items -> findings)
# ---------------------------------------------------------------------------


def test_build_candidate_findings_maps_and_skips() -> None:
    from daydream.benchmark.harbor import candidate
    from daydream.benchmark.harbor import verifier_core as vc

    items = [
        {"file": "src/cache.py", "line": 42,
         "description": "Cache key not tenant-scoped",
         "rationale": "Key can collide across tenants.", "severity": "high", "confidence": "HIGH"},
        {"file": "", "line": 1, "description": "empty-file item is skipped",
         "rationale": "", "severity": "low"},
        {"file": "src/a.py", "line": 3, "description": "Dup A", "rationale": "r",
         "severity": "medium", "confidence": "LOW"},
        {"file": "src/a.py", "line": 9, "description": "Dup A", "rationale": "r",
         "severity": "medium", "confidence": "LOW"},
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


def test_build_candidate_findings_enforces_verifier_bounds_fail_closed() -> None:
    from daydream.benchmark.harbor import candidate

    case_id = "case-abc123def456"
    # A verbose canonical description is preserved in the body while the
    # display title is deterministically bounded for the verifier schema.
    overlong_title = [{"file": "src/a.py", "line": 1,
                       "description": "t" * 501, "rationale": "", "severity": "low"}]
    [bounded] = candidate.build_candidate_findings(overlong_title, case_id=case_id)
    assert len(bounded["title"]) <= 500
    assert "t" * 501 in bounded["body"]

    # over-long body (>8 KiB) is a typed failure
    overlong_body = [{"file": "src/a.py", "line": 1,
                      "description": "header", "rationale": "r" * 9000, "severity": "low"}]
    with pytest.raises(candidate.CandidateError) as bad_body:
        candidate.build_candidate_findings(overlong_body, case_id=case_id)
    assert bad_body.value.kind == "invalid_finding"

    # non-enum severity is a typed failure
    bad_severity = [{"file": "src/a.py", "line": 1,
                     "description": "x", "rationale": "", "severity": "CRITICAL"}]
    with pytest.raises(candidate.CandidateError) as bad_sev:
        candidate.build_candidate_findings(bad_severity, case_id=case_id)
    assert bad_sev.value.kind == "invalid_finding"


# ---------------------------------------------------------------------------
# Task 2: candidate artifact assembly + caps + atomic write
# ---------------------------------------------------------------------------


def test_artifact_caps_fail_closed_and_write_is_atomic(tmp_path: Path) -> None:
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


def test_artifact_write_failure_raises(tmp_path: Path) -> None:
    from daydream.benchmark.harbor import candidate

    dest = tmp_path / "adir"                                # a directory -> replace fails
    dest.mkdir()
    with pytest.raises(candidate.CandidateError) as write_fail:
        candidate.write_candidate_artifact_atomic(dest, {"schema_version": 1, "findings": []})
    assert write_fail.value.kind == "write_failure"


# ---------------------------------------------------------------------------
# Task 3: render_task_toml — reviewer-only host policy + case env threading
# ---------------------------------------------------------------------------


def test_render_task_toml_host_policy_and_case_env() -> None:
    from daydream.benchmark.harbor import package as pkg

    toml = pkg.render_task_toml(
        "case-abc123def456",
        reviewer_hosts=["openrouter.ai"],
        judge_hosts=["openrouter.ai"],
    ).decode("utf-8")
    assert 'allowed_hosts = ["openrouter.ai"]' in toml       # reviewer-only host
    assert '"github.com"' not in toml and '"huggingface.co"' not in toml
    assert 'DAYDREAM_REVIEW_CASE_ID = "case-abc123def456"' in toml
    assert 'DAYDREAM_REVIEW_BASE_REF = "base"' in toml
    assert 'DAYDREAM_REVIEW_HEAD_REF = "head"' in toml
    # deterministic
    assert pkg.render_task_toml(
        "case-abc123def456",
        reviewer_hosts=["openrouter.ai"],
        judge_hosts=["openrouter.ai"],
    ) == pkg.render_task_toml(
        "case-abc123def456",
        reviewer_hosts=["openrouter.ai"],
        judge_hosts=["openrouter.ai"],
    )
    # no judge vars or archive/target config reach the agent surface
    assert "DAYDREAM_JUDGE" not in toml and "HF_TOKEN" not in toml and "GITHUB_TOKEN" not in toml


def test_render_task_toml_keeps_agent_verifier_host_boundaries() -> None:
    from daydream.benchmark.harbor import package as pkg

    toml = pkg.render_task_toml(
        "case-abc123def456",
        reviewer_hosts=["reviewer.example.com"],
        judge_hosts=["openrouter.ai"],
    ).decode("utf-8")
    agent_block = toml.split("[agent]", 1)[1].split("[environment]", 1)[0]
    verifier_block = toml.split("[verifier.environment]", 1)[1]
    assert 'allowed_hosts = ["reviewer.example.com"]' in agent_block
    assert "openrouter.ai" not in agent_block
    assert 'allowed_hosts = ["openrouter.ai"]' in verifier_block
    assert "reviewer.example.com" not in verifier_block


# ---------------------------------------------------------------------------
# Task 4: entrypoint — controlled in-container runner
# ---------------------------------------------------------------------------


def test_entrypoint_build_run_config_is_controlled() -> None:
    from daydream.benchmark.harbor import entrypoint
    from daydream.config_file import DaydreamFileConfig

    cfg = entrypoint.build_run_config(
        repo_dir="/workspace/repo",
        trajectory_path="/logs/agent/trajectory.json",
        backend="pi",
        model="deepseek/deepseek-v4-flash-0731",
    )
    assert cfg.output_mode == "review"
    assert cfg.base == "base"
    assert cfg.non_interactive is True
    assert cfg.archive is False and cfg.run_eval is False
    assert cfg.findings_out is None                     # NO --findings-out (live PR lookup)
    assert cfg.trajectory_path == Path("/logs/agent/trajectory.json")
    assert cfg.backend == "pi" and cfg.model == "deepseek/deepseek-v4-flash-0731"
    assert isinstance(cfg.file_config, DaydreamFileConfig)
    # controlled empty: the target repo's .daydream.toml is never loaded
    assert cfg.file_config == DaydreamFileConfig()


def test_entrypoint_backend_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.benchmark.harbor import entrypoint

    monkeypatch.setenv("DAYDREAM_REVIEW_BACKEND", "codex")
    with pytest.raises(entrypoint.EntrypointError) as exc:
        entrypoint.require_supported_backend()
    assert "pi" in str(exc.value)


def test_entrypoint_backend_allowlist_pi_and_claude_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.benchmark.harbor import entrypoint

    for value in ("pi", "claude", "CLAUDE", " claude "):
        monkeypatch.setenv("DAYDREAM_REVIEW_BACKEND", value)
        assert entrypoint.require_supported_backend() == value.strip().lower()


def test_entrypoint_backend_allowlist_rejects_others_and_defaults_pi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream.benchmark.harbor import entrypoint

    for value in ("codex", "opencode"):
        monkeypatch.setenv("DAYDREAM_REVIEW_BACKEND", value)
        with pytest.raises(entrypoint.EntrypointError) as exc:
            entrypoint.require_supported_backend()
        assert "'pi'" in str(exc.value) and "'claude'" in str(exc.value)
    monkeypatch.delenv("DAYDREAM_REVIEW_BACKEND", raising=False)
    assert entrypoint.require_supported_backend() == "pi"


# ---------------------------------------------------------------------------
# Task 5: entrypoint — publish candidate artifact + failure modes
# ---------------------------------------------------------------------------


def test_entrypoint_publish_failure_modes(tmp_path: Path) -> None:
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


def test_agent_package_import_does_not_pull_harbor() -> None:
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


def test_agent_lifecycle_and_lazy_harbor() -> None:
    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    assert DaydreamReviewAgent.SUPPORTS_ATIF is True
    assert callable(DaydreamReviewAgent.name) and callable(DaydreamReviewAgent.version)
    assert isinstance(DaydreamReviewAgent.name(), str) and DaydreamReviewAgent.name()
    assert isinstance(DaydreamReviewAgent.version(), str)


def test_agent_setup_confirms_version_and_backend(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult

    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    agent = DaydreamReviewAgent(
        logs_dir=tmp_path, extra_env={"DAYDREAM_REVIEW_BACKEND": "pi"}
    )

    class Env:
        async def exec(self, command: Any, cwd: Any=None, env: Any=None, timeout_sec: Any=None, user: Any=None) -> Any:
            self.captured = command
            return ExecResult(return_code=0, stdout="ok", stderr="")

    env = Env()
    import asyncio

    asyncio.run(agent.setup(env))
    assert agent.version() in env.captured        # setup checks the packaged version
    assert "shutil.which('pi')" in env.captured      # and the required Pi CLI


def test_agent_setup_probe_branches_on_backend(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult

    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    class Env:
        def __init__(self) -> None:
            self.captured = ""
        async def exec(self, command: Any, cwd: Any=None, env: Any=None, timeout_sec: Any=None, user: Any=None) -> Any:
            self.captured = command
            return ExecResult(return_code=0, stdout="ok", stderr="")

    pi_agent = DaydreamReviewAgent(
        logs_dir=tmp_path, extra_env={"DAYDREAM_REVIEW_BACKEND": "pi"}
    )
    env_pi = Env()
    import asyncio
    asyncio.run(pi_agent.setup(env_pi))
    assert "shutil.which('pi')" in env_pi.captured
    assert "claude_agent_sdk" not in env_pi.captured

    claude_agent = DaydreamReviewAgent(
        logs_dir=tmp_path, extra_env={"DAYDREAM_REVIEW_BACKEND": "claude"}
    )
    env_claude = Env()
    asyncio.run(claude_agent.setup(env_claude))
    assert "import claude_agent_sdk" in env_claude.captured
    assert "shutil.which('pi')" not in env_claude.captured
    assert claude_agent.version() in env_claude.captured          # version assert kept for both


def test_agent_setup_nonzero_exec_fails(tmp_path: Path) -> None:
    """A failed setup probe surfaces as a typed failure, never a silent pass."""
    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult

    from daydream.benchmark.harbor.agent import AgentError, DaydreamReviewAgent

    agent = DaydreamReviewAgent(logs_dir=tmp_path)

    class Env:
        async def exec(self, command: Any, cwd: Any=None, env: Any=None, timeout_sec: Any=None, user: Any=None) -> Any:
            self.captured = command
            return ExecResult(return_code=1, stdout="", stderr="boom")

    import asyncio

    with pytest.raises(AgentError):
        asyncio.run(agent.setup(Env()))


# ---------------------------------------------------------------------------
# Task 7: agent run() — Pi guarantee + allowlist child env + isolation
# ---------------------------------------------------------------------------


_BANNED = [
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "DAYDREAM_APP_ID",
    "DAYDREAM_APP_PRIVATE_KEY",
    "HF_TOKEN",
    "DAYDREAM_TRAJECTORY_HUB_REPO",
    "DAYDREAM_ARCHIVE_DIR",
    "DAYDREAM_JUDGE_API_KEY",
    "DAYDREAM_JUDGE_MODEL",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "PI_API_KEY",
]


def test_build_child_env_is_exact_allowlist() -> None:
    from daydream.benchmark.harbor.agent import build_child_env

    parent = {
        **{k: "secret" for k in _BANNED},
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "review-key",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        "PATH": "/usr/bin",
        "HOME": "/root",
        "LANG": "C.UTF-8",
        "RANDOM_SECRET": "ignore-me",
    }
    child = build_child_env(parent)
    assert child["DAYDREAM_REVIEW_API_KEY"] == "review-key"  # reviewer credential kept
    for banned in _BANNED:
        assert banned not in child                              # fail-closed: absent
    assert child["PATH"] == "/usr/bin" and child["HOME"] == "/root"
    # required process vars survive; nothing arbitrary leaks through.
    assert set(child).issubset(
        {
            "DAYDREAM_REVIEW_BACKEND",
            "DAYDREAM_REVIEW_API_KEY",
            "DAYDREAM_REVIEW_BASE_URL",
            "PATH",
            "HOME",
            "LANG",
        }
    )


def test_build_child_env_keeps_anthropic_for_claude_scrubs_for_pi() -> None:
    from daydream.benchmark.harbor.agent import build_child_env

    parent = {
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_AUTH_TOKEN": "tok",
        "OPENROUTER_API_KEY": "sk-or",
        "PI_API_KEY": "pi-key",
        "GH_TOKEN": "gh",
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "DAYDREAM_REVIEW_API_KEY": "review-key",
        "PATH": "/usr/bin",
        "HOME": "/root",
        "LANG": "C.UTF-8",
    }
    child = build_child_env(parent, backend="claude")
    assert child["ANTHROPIC_API_KEY"] == "sk-ant"
    assert child["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert child["ANTHROPIC_AUTH_TOKEN"] == "tok"
    assert "OPENROUTER_API_KEY" not in child and "PI_API_KEY" not in child
    assert "GH_TOKEN" not in child                              # non-credential bans stay

    child_pi = build_child_env(parent, backend="pi")
    assert not any(k.startswith("ANTHROPIC_") for k in child_pi)
    assert "OPENROUTER_API_KEY" not in child_pi
    assert child_pi["DAYDREAM_REVIEW_API_KEY"] == "review-key"


def test_agent_run_refuses_unsupported_backend_and_invokes_entrypoint(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult
    from harbor.models.agent.context import AgentContext

    from daydream.benchmark.harbor.agent import AgentError, DaydreamReviewAgent

    agent = DaydreamReviewAgent(
        logs_dir=tmp_path,
        extra_env={"DAYDREAM_REVIEW_BACKEND": "codex"},
    )
    with pytest.raises(AgentError) as refused:                     # before any reviewing
        import asyncio

        asyncio.run(agent.run("instruction", object(), AgentContext()))
    assert "pi" in str(refused.value)

    agent_ok = DaydreamReviewAgent(
        logs_dir=tmp_path,
        extra_env={
            "DAYDREAM_REVIEW_BACKEND": "pi",
            "DAYDREAM_REVIEW_API_KEY": "k",
            "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        },
    )

    class Env:
        async def exec(self, command: Any, cwd: Any=None, env: Any=None, timeout_sec: Any=None, user: Any=None) -> Any:
            self.captured = (command, cwd, env)
            return ExecResult(return_code=0, stdout="", stderr="")

    env = Env()
    import asyncio

    asyncio.run(agent_ok.run("instruction", env, AgentContext()))
    cmd, cwd, child = env.captured
    assert "daydream.benchmark.harbor.entrypoint" in cmd
    assert cwd == "/workspace/repo"
    assert "ANTHROPIC_API_KEY" not in child and "DAYDREAM_REVIEW_API_KEY" in child
    assert "--findings-out" not in cmd                     # no live-PR emission path


# ---------------------------------------------------------------------------
# Task 8: populate_context_post_run — trajectory metrics -> AgentContext
# ---------------------------------------------------------------------------


def test_populate_context_from_trajectory_final_metrics(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.models.agent.context import AgentContext

    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    traj_dir = tmp_path / "agent"
    traj_dir.mkdir(parents=True)
    (traj_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "final_metrics": {
                    "total_prompt_tokens": 1200,
                    "total_cached_tokens": 300,
                    "total_completion_tokens": 800,
                    "total_cost_usd": 0.42,
                },
            }
        )
    )
    agent = DaydreamReviewAgent(logs_dir=tmp_path)
    ctx = AgentContext()
    agent.populate_context_post_run(ctx)
    assert ctx.n_input_tokens == 1200
    assert ctx.n_cache_tokens == 300
    assert ctx.n_output_tokens == 800
    assert ctx.cost_usd == 0.42


def test_populate_context_absent_trajectory_leaves_metrics_unset(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.models.agent.context import AgentContext

    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    agent = DaydreamReviewAgent(logs_dir=tmp_path)  # no agent/trajectory.json
    ctx = AgentContext()
    agent.populate_context_post_run(ctx)
    assert ctx.is_empty()  # metrics stay unset; no fabricated values


def test_populate_context_malformed_trajectory_leaves_metrics_unset(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.models.agent.context import AgentContext

    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    traj_dir = tmp_path / "agent"
    traj_dir.mkdir(parents=True)
    (traj_dir / "trajectory.json").write_text("not json at all")
    agent = DaydreamReviewAgent(logs_dir=tmp_path)
    ctx = AgentContext()
    agent.populate_context_post_run(ctx)
    assert ctx.is_empty()


# ---------------------------------------------------------------------------
# Task 9: validate --compiled agent-import preflight
# ---------------------------------------------------------------------------


def test_validate_compiled_imports_agent_path_same_interpreter(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import pytest

    pytest.importorskip("harbor")
    from daydream.benchmark.harbor import package as pkg
    from tests.test_benchmark_harbor_build import _seed_ready_workspace

    ws, _, _ = _seed_ready_workspace(tmp_path, fake_gh)
    ver = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    pkg.build_harbor(ws, wheel=wheel)

    assert pkg.validate_compiled(ws) == 0  # agent class imports in the same interpreter

    # A separate/missing environment fails before a trial with remediation.
    real_import = importlib.import_module

    def broken(name: str, *a: Any, **k: Any) -> Any:
        if name == "daydream.benchmark.harbor.agent":
            raise ModuleNotFoundError("no module named 'daydream.benchmark.harbor.agent'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.validate_compiled(ws)
    assert "daydream.benchmark.harbor.agent" in str(rejected.value)
    assert "pip install 'daydream[benchmark]'" in str(rejected.value)


# ---------------------------------------------------------------------------
# Task 10: end-to-end real-path test — exact findings + explicit empty review
# ---------------------------------------------------------------------------


_KNOWN_ITEMS = [
    {"id": 1, "lens": "per-stack", "file": "api.py", "line": 1,
     "severity": "medium", "description": "Sample issue",
     "confidence": "MEDIUM", "rationale": "stub", "evidence": "api.py:1"},
    {"id": 2, "lens": "per-stack", "file": "api.py", "line": 1,
     "severity": "medium", "description": "Sample issue",
     "confidence": "MEDIUM", "rationale": "stub", "evidence": "api.py:1"},
]

# The stub-produced findings are the deterministic output of the real deep
# pipeline over this single-python-file diff: one per-stack Record schema item,
# normalized and rendered twice as duplicate-content findings.
_EXPECTED_TITLES = ["Sample issue", "Sample issue"]


def _seed_defect_repo(tmp_path: Path) -> Path:
    """Build a real temp git repo with a ``base`` ref and a single-python diff.

    Mirrors the deep-orchestrator's real-worktree fixtures, adding a ref literally
    named ``base`` (what the entrypoint's ``RunConfig.base="base"`` resolves
    against).
    """
    from tests.harness.git_helpers import commit as _commit
    from tests.harness.git_helpers import git as _git
    from tests.harness.git_helpers import init_repo as _init_repo

    project = tmp_path / "fixture"
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'world'\n")
    _init_repo(project)
    _git(project, "add", ".")
    _commit(project, "prime")
    _git(project, "branch", "base")
    _git(project, "checkout", "-b", "feature")
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    _git(project, "add", ".")
    _commit(project, "change")
    return project


def _end_env(repo: Path, tmp: Path, case_id: str) -> dict[str, str]:
    return {
        "DAYDREAM_REVIEW_REPO_DIR": str(repo),
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(tmp / "logs" / "artifacts" / "review.json"),
        "DAYDREAM_REVIEW_TRAJECTORY_PATH": str(tmp / "logs" / "agent" / "trajectory.json"),
        "DAYDREAM_REVIEW_CASE_ID": case_id,
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
    }


def test_end_to_end_findings_and_clean_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real temp git repo/task plus a fake backend drives the production entrypoint
    through the in-process runner and publishes the exact findings and an explicit
    empty review (AC 3 / AC 5 gate)."""
    import os

    from daydream.benchmark.harbor import entrypoint
    from daydream.benchmark.harbor import verifier_core as vc
    from tests.harness.stub_backend import install_stub_backend

    repo = _seed_defect_repo(tmp_path)
    install_stub_backend(monkeypatch, repo)
    case_id = "case-abc123def456"
    env = _end_env(repo, tmp_path, case_id)

    saved = {key: os.environ.get(key) for key in env}
    try:
        import asyncio

        rc = asyncio.run(entrypoint.main(monkeypatch_env=env))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert rc == 0
    artifact = json.loads((tmp_path / "logs" / "artifacts" / "review.json").read_text())
    parsed = vc.validate_candidate_artifact(artifact)
    assert [p.candidate_id for p in parsed] == [f["candidate_id"] for f in artifact["findings"]]
    assert [f["title"] for f in artifact["findings"]] == _EXPECTED_TITLES
    assert len({f["candidate_id"] for f in artifact["findings"]}) == len(artifact["findings"])
    assert artifact["case_id"] == case_id
    assert artifact["base_ref"] == "base" and artifact["head_ref"] == "head"

    # explicit empty review: a genuinely-empty deep review writes no merged output
    # (it fail-closes), so the clean-review contract is exercised through the very
    # same production publish step main() runs — a present, schema-valid artifact
    # whose findings list is explicitly empty.
    (repo / ".daydream" / "deep" / "merged-items.json").write_text(
        json.dumps({"items": []})
    )
    dest = tmp_path / "logs" / "artifacts" / "review-clean.json"
    entrypoint.publish_review(
        repo_dir=repo, artifact_path=dest, case_id=case_id
    )
    empty = json.loads(dest.read_text())
    assert empty["findings"] == []
    assert vc.validate_candidate_artifact(empty) == []


class Executed:
    """Captured results of the real agent lifecycle calls on the fake env."""

    def __init__(self) -> None:
        self.setup = ""
        self.command = ""
        self.cwd: str | None = None
        self.child: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Task 11: one local fake-backend Harbor task + make check
# ---------------------------------------------------------------------------


def test_local_harbor_task_with_fake_backend(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 5 gate: compile a real Harbor task with the custom agent, validate it
    in the same interpreter, then execute a local fake-backend Harbor trial
    end-to-end. The production :class:`DaydreamReviewAgent` ``setup()`` and
    ``run()`` drive a fake Harbor environment whose ``exec`` injects the allow-
    listed child env (``build_child_env``) into the entrypoint's ``os.environ``
    and the real runner + publisher complete to a candidate artifact -- so the
    env-injection, allowlist child-env traversal, and setup probe are all
    executed, not skipped. Only the in-docker nftables sandbox itself needs a
    Harbor-capable runtime this host does not provide (plan §14); that half is
    documented, but the runnable gate is a genuine executed pass."""
    import asyncio
    import importlib
    import os

    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult
    from harbor.models.agent.context import AgentContext

    from daydream.benchmark.harbor import build, entrypoint
    from daydream.benchmark.harbor import package as pkg
    from daydream.benchmark.harbor import verifier_core as vc
    from daydream.benchmark.harbor.agent import (
        _BANNED_PREFIXES,
        _BANNED_VARS,
        DaydreamReviewAgent,
    )
    from tests.harness.stub_backend import install_stub_backend
    from tests.test_benchmark_harbor_build import _seed_ready_workspace

    # Compile the wheel + validate the compiled tree, including the custom-agent
    # same-interpreter preflight.
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    ver = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    pkg.build_harbor(ws, wheel=wheel)
    assert pkg.validate_compiled(ws) == 0
    key = build.derive_task_key(case_id)
    assert (ws / "harbor" / key / "task.toml").is_file()
    assert f'DAYDREAM_REVIEW_CASE_ID = "{key}"' in (ws / "harbor" / key / "task.toml").read_text()

    # Freeze a real temp git repo (base/head) to review; the fake backend keeps
    # the in-process runner deterministic.
    repo = _seed_defect_repo(tmp_path)
    install_stub_backend(monkeypatch, repo)
    task_env = {
        "DAYDREAM_REVIEW_CASE_ID": key,
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        "DAYDREAM_REVIEW_REPO_DIR": str(repo),
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(
            tmp_path / "logs" / "artifacts" / "review.json"
        ),
        "DAYDREAM_REVIEW_TRAJECTORY_PATH": str(
            tmp_path / "logs" / "agent" / "trajectory.json"
        ),
    }
    # Host secrets present in the parent env must never reach the child env.
    for banned in _BANNED_VARS:
        monkeypatch.setenv(banned, "super-secret")

    agent = DaydreamReviewAgent(logs_dir=tmp_path / "logs", extra_env=task_env)

    executed = Executed()

    class Env:
        async def exec(
            self,
            command: Any,
            cwd: Any=None,
            env: Any=None,
            timeout_sec: Any=None,
            user: Any=None,
        ) -> ExecResult:
            if "entrypoint" in command:
                # Harbor injects the per-case child env into the container;
                # capture it so we can assert it is exactly the allowlist and
                # then really execute the entrypoint against it below.
                executed.command = command
                executed.cwd = cwd
                executed.child = env or {}
            else:
                executed.setup = command
            return ExecResult(return_code=0, stdout="", stderr="")

    env = Env()
    asyncio.run(agent.setup(env))
    asyncio.run(agent.run("review the frozen snapshot", env, AgentContext()))

    # setup() executed: the in-container probe asserts this packaged release + SDK.
    assert agent.version() in executed.setup
    assert "shutil.which('pi')" in executed.setup
    # run() executed with the allowlisted child env traversing to the entrypoint only.
    assert "daydream.benchmark.harbor.entrypoint" in executed.command
    assert executed.cwd == str(repo)  # cwd tracks DAYDREAM_REVIEW_REPO_DIR
    assert "--findings-out" not in executed.command  # no live-PR emission path
    assert executed.child["DAYDREAM_REVIEW_CASE_ID"] == key
    assert executed.child["DAYDREAM_REVIEW_BACKEND"] == "pi"
    assert set(executed.child) <= {"PATH", "HOME", "LANG"} | {
        k for k in executed.child if k.startswith("DAYDREAM_REVIEW_")
    }
    for banned in _BANNED_VARS:
        assert banned not in executed.child  # fail-closed: host secrets never reach the child
    for prefix in _BANNED_PREFIXES:
        assert not any(k.startswith(prefix) for k in executed.child)

    # Only the allowlisted child env reaches the executed entrypoint: drop the
    # host secrets we planted so the in-process runner sees the container env,
    # then really run the entrypoint against the captured child env.
    for banned in _BANNED_VARS:
        os.environ.pop(banned, None)
    saved = {env_key: os.environ.get(env_key) for env_key in executed.child}
    try:
        rc = asyncio.run(entrypoint.main(monkeypatch_env=executed.child))
    finally:
        for env_key, value in saved.items():
            if value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = value
    assert rc == 0

    # The child env carried the per-case task key through to a genuinely
    # completed review; the artifact is the exact frozen-snapshot candidate.
    artifact = json.loads(
        (tmp_path / "logs" / "artifacts" / "review.json").read_text()
    )
    parsed = vc.validate_candidate_artifact(artifact)
    assert [p.candidate_id for p in parsed] == [
        f["candidate_id"] for f in artifact["findings"]
    ]
    assert [f["title"] for f in artifact["findings"]] == _EXPECTED_TITLES
    assert artifact["case_id"] == key
    assert artifact["base_ref"] == "base" and artifact["head_ref"] == "head"


def test_agent_run_accepts_claude_and_invokes_entrypoint(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("harbor")
    from harbor.environments.base import ExecResult
    from harbor.models.agent.context import AgentContext

    from daydream.benchmark.harbor.agent import DaydreamReviewAgent

    agent = DaydreamReviewAgent(
        logs_dir=tmp_path,
        extra_env={
            "DAYDREAM_REVIEW_BACKEND": "claude",
            "DAYDREAM_REVIEW_API_KEY": "k",
        },
    )

    class Env:
        async def exec(self, command: Any, cwd: Any=None, env: Any=None, timeout_sec: Any=None, user: Any=None) -> Any:
            self.captured = (command, cwd, env)
            return ExecResult(return_code=0, stdout="", stderr="")

    env = Env()
    import asyncio

    asyncio.run(agent.run("instruction", env, AgentContext()))
    cmd, cwd, child = env.captured
    assert "daydream.benchmark.harbor.entrypoint" in cmd
    assert child["DAYDREAM_REVIEW_BACKEND"] == "claude"
