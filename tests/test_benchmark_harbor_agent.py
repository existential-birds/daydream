"""Privacy-safe Daydream Harbor review agent (issue #780) tests."""
from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import candidate, package as pkg, verifier_core as vc
from daydream.benchmark.harbor.agent import AgentError, DaydreamReviewAgent
from tests.harness.fake_gh import FakeGh
from tests.test_benchmark_harbor_build import _seed_ready_workspace, _stub_wheel


def test_artifact_caps_fail_closed_and_write_is_atomic(tmp_path: Path) -> None:
    case_id = "case-abc123def456"
    over = [{"title": f"t{i}", "body": "b", "severity": "low", "path": "src/f.py", "start_line": i, "end_line": i}
            for i in range(1, vc.MAX_CANDIDATE_FINDINGS + 2)]
    with pytest.raises(candidate.CandidateError) as too_many:
        candidate.build_candidate_artifact(case_id, over)
    assert too_many.value.kind == "over_limit"

    dest = tmp_path / "logs" / "artifacts" / "review.json"
    art = candidate.build_candidate_artifact(case_id, [])
    assert art == {"schema_version": 1, "case_id": case_id, "base_ref": "base", "head_ref": "head", "findings": []}
    candidate.write_candidate_artifact_atomic(dest, art)
    loaded = json.loads(dest.read_text())
    assert loaded == art and loaded["findings"] == []      # clean review round-trips
    assert vc.validate_candidate_artifact(loaded) == []     # schema-valid
    assert list(dest.parent.glob("review.json*")) == [dest]

    for field, value in (("title", ""), ("body", "bad\x00body"), ("path", "../escape")):
        finding: dict[str, object] = {"title": "Title", "body": "Body", "severity": "low",
            "path": "src/a.py", "start_line": 1, "end_line": 1, field: value}
        finding["candidate_id"] = vc.derive_candidate_id("case-key", finding, 0)
        with pytest.raises(candidate.CandidateError) as rejected:
            candidate.build_candidate_artifact("case-key", [finding])
        assert rejected.value.kind == "invalid_finding"


def test_agent_package_import_does_not_pull_harbor() -> None:
    """Importing daydream.benchmark must not import the optional Harbor extra."""
    probe = ("import sys; import daydream.benchmark; "
        "assert not any(m == 'harbor' or m.startswith('harbor.') for m in sys.modules), "
        "[m for m in sys.modules if m == 'harbor' or m.startswith('harbor.')]"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, f"harbor imported eagerly:\n{out.stdout}{out.stderr}"


def test_agent_setup_fails_on_probe_error_and_refuses_unsupported_backend_before_probe(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    with pytest.raises(AgentError):
        asyncio.run(DaydreamReviewAgent(logs_dir=tmp_path).setup(_capturing_env(Executed(return_code=1))()))
    agent = DaydreamReviewAgent(logs_dir=tmp_path, extra_env={"DAYDREAM_REVIEW_BACKEND": "codex"})
    executed = Executed()
    with pytest.raises(AgentError, match="pi"):
        asyncio.run(agent.setup(_capturing_env(executed)()))
    assert executed.setup == "" and executed.command == ""

def test_agent_run_refuses_unsupported_backend_and_invokes_entrypoint(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from harbor.models.agent.context import AgentContext

    agent = DaydreamReviewAgent(logs_dir=tmp_path, extra_env={"DAYDREAM_REVIEW_BACKEND": "codex"})
    with pytest.raises(AgentError) as refused:                     # before any reviewing

        asyncio.run(agent.run("instruction", object(), AgentContext()))
    assert "pi" in str(refused.value)

    agent_ok = DaydreamReviewAgent(logs_dir=tmp_path,
        extra_env={"DAYDREAM_REVIEW_BACKEND": "pi", "DAYDREAM_REVIEW_API_KEY": "k",
            "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        },
    )

    executed = Executed()

    asyncio.run(agent_ok.run("instruction", _capturing_env(executed)(), AgentContext()))
    assert "daydream.benchmark.harbor.entrypoint" in executed.command
    assert executed.cwd == "/workspace/repo"
    assert "ANTHROPIC_API_KEY" not in executed.child and "DAYDREAM_REVIEW_API_KEY" in executed.child
    assert "--findings-out" not in executed.command         # no live-PR emission path

def test_populate_context_from_trajectory_final_metrics(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from harbor.models.agent.context import AgentContext

    traj_dir = tmp_path / "agent"
    traj_dir.mkdir(parents=True)
    (traj_dir / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.7",
                "final_metrics": {
                    "total_prompt_tokens": 1200, "total_cached_tokens": 300, "total_completion_tokens": 800,
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

    absent = AgentContext()
    DaydreamReviewAgent(logs_dir=tmp_path / "absent").populate_context_post_run(absent)
    assert absent.is_empty()

def test_validate_compiled_imports_agent_path_same_interpreter(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("harbor")

    ws, _, _ = _seed_ready_workspace(tmp_path, fake_gh)
    wheel, _ = _stub_wheel(tmp_path)
    pkg.build_harbor(ws, wheel=wheel)

    assert pkg.validate_compiled(ws) == 0  # agent class imports in the same interpreter

    # A separate/missing environment fails before a trial with remediation.
    real_import = importlib.import_module

    def broken(name: str, *a: Any, **k: Any) -> Any:
        if name == "daydream.benchmark.harbor.agent":
            raise ModuleNotFoundError("no module named 'daydream.benchmark.harbor.agent'")
        return real_import(name, *a, **k)

    with monkeypatch.context() as separate, pytest.raises(pkg.PackageError) as rejected:
        separate.setattr(importlib, "import_module", broken)
        pkg.validate_compiled(ws)
    assert "daydream.benchmark.harbor.agent" in str(rejected.value)
    assert "pip install 'daydream[benchmark]'" in str(rejected.value)

    # Any byte change after compile breaks the recorded inventory.
    with (ws / "harbor/metric.py").open("a") as stream:
        stream.write("# edited after compile\n")
    with pytest.raises(pkg.PackageError, match="hash mismatch for metric.py"):
        pkg.validate_compiled(ws)

class Executed:
    """Captured results of the real agent lifecycle calls on the fake env."""

    def __init__(self, *, return_code: int = 0) -> None:
        self.setup = ""
        self.command = ""
        self.cwd: str | None = None
        self.child: dict[str, str] = {}
        self.return_code = return_code


def _capturing_env(executed: Executed) -> type:
    """Fake Harbor Environment whose exec captures entrypoint command/cwd/child env."""
    from harbor.environments.base import ExecResult

    class Env:
        async def exec(self, command: Any, cwd: Any = None, env: Any = None, timeout_sec: Any = None, user: Any = None,
        ) -> ExecResult:
            if "entrypoint" in command:
                executed.command = command
                executed.cwd = cwd
                executed.child = env or {}
            else:
                executed.setup = command
            return ExecResult(return_code=executed.return_code, stdout="", stderr="")

    return Env
