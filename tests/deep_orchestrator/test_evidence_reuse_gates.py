"""Real deep-flow tests binding finalized host-test evidence to the commit gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.artifacts import DeepArtifact
from daydream.phases import TestAttemptEvidence, phase_commit_push
from daydream.runner import run
from daydream.test_execution import TestExecutionIdentity
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import (
    bare_remote as _bare_remote,
    git as _git,
    seed_feature_branch as _seed_feature_branch,
)
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import MakeConfig, _silence


@pytest.mark.asyncio
async def test_a_half_formed_offer_is_refused(tmp_path: Path, make_work: Any, make_config: MakeConfig) -> None:
    """One half of the evidence/tree pair is a caller bug, not a silent fallback."""
    work = make_work(tmp_path / "half-formed")
    config = make_config(work.repo)
    evidence = TestAttemptEvidence(
        session_id="s", kind="host", command=("true",), passed=True, input_tree_key="t", output_tree_key="t",
    )
    with pytest.raises(ValueError, match="together"):
        await phase_commit_push(ScriptedBackend(), work, config=config, evidence=evidence)
    with pytest.raises(ValueError, match="together"):
        await phase_commit_push(ScriptedBackend(), work, config=config, retained_tree_key="t")

@pytest.mark.asyncio
async def test_a_tree_key_mismatch_is_refused(tmp_path: Path, make_work: Any, make_config: MakeConfig) -> None:
    """A retained key that disagrees with the evidence names no valid offer."""
    work = make_work(tmp_path / "mismatch")
    config = make_config(work.repo)
    identity = TestExecutionIdentity(session_id="s", argv=("true",), cwd_relative=".", runner=None, interpreter=None,
        config_digest=None, absent_components=(), input_tree_key="t", output_tree_key="t",
        head_sha="a" * 40, branch="feature", kind="host", outcome="passed",
    )
    evidence = TestAttemptEvidence(session_id="s", kind="host", command=("true",), passed=True,
        input_tree_key="t", output_tree_key="t", identity=identity,
    )
    with pytest.raises(ValueError, match="does not match"):
        await phase_commit_push(ScriptedBackend(), work, config=config, evidence=evidence, retained_tree_key="other",)

@pytest.mark.asyncio
async def test_real_flow_skips_the_pre_push_suite_run_but_still_runs_the_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, no_ci_remote: NoCIRemote,
) -> None:
    """Reuse skips the redundant proactive suite run while the hook and real push execute.

    The host-command counter must show exactly one TEST-phase execution.
    """
    repo = tmp_path / "hook-reuse-flow"
    _seed_feature_branch(repo, base={"api.py": "A = 1\n"}, feature={"api.py": "A = 2\n"})
    remote = _bare_remote(tmp_path / "origin.git")
    no_ci_remote.connect(repo, remote)

    hook_log = tmp_path / "hooks.log"
    hook = repo / ".git" / "hooks" / "pre-push"
    # Preserve the seeding hook's SHA publication and ready-state verification.
    with hook.open("a") as stream:
        stream.write(f"echo pre-push >> '{hook_log}'\n")

    counter = tmp_path / "host-runs"
    script = tmp_path / "record.py"
    script.write_text(
        "import pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "path.write_text(str(int(path.read_text()) + 1) if path.exists() else '1')\n"
    )

    observed: list[TestAttemptEvidence | None] = []
    retained_keys: list[str | None] = []
    real_phase = phase_commit_push

    async def spy(*args: Any, **kwargs: Any) -> Any:
        observed.append(kwargs.get("evidence"))
        retained_keys.append(kwargs.get("retained_tree_key"))
        return await real_phase(*args, **kwargs)

    monkeypatch.setattr("daydream.deep.fix_steps.phase_commit_push", spy)

    backend = StubBackend(repo)
    backend.fix_edit_line = "# repaired\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _silence(monkeypatch)

    assert await run(make_config(
        repo, assume="yes", output_mode="loop", test_command=f"{sys.executable} {script} {counter}",
        pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
    )) == 0

    assert counter.read_text() == "1", ("the TEST phase is the only orchestrator suite run; the commit gate "
        "reused the matching evidence instead of re-running it"
    )
    assert hook_log.read_text().splitlines() == ["pre-push"]
    assert _git(remote, "rev-parse", "refs/heads/feature") == _git(repo, "rev-parse", "HEAD")

    assert len(observed) == 1, "one finalized fix reaches the commit gate"
    evidence = observed[0]
    assert evidence is not None and evidence.identity is not None
    assert evidence.identity.outcome == "passed"
    assert evidence.identity.output_tree_key
    assert retained_keys == [evidence.identity.output_tree_key]

    record = json.loads(DeepArtifact.EVIDENCE_REUSE.at(repo / ".daydream" / "deep").read_text())
    gate = record["gates"]["pre-push"]
    assert gate["gate"] == "pre-push"
    assert gate["format_version"] == 1
    assert gate["result"] == "reused"
    assert gate["reused"] is True
    assert gate["post_commit_verification_used"] is True
