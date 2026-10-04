# tests/test_repair_job.py
"""The durable repair checkpoint and the repair job record (issue #1210).

A repair turn's authorized work is captured from the live tree *before* any
restoration runs, because cleanup can be the only thing that erases it. These
tests pin the payload contract, the atomic versioned write, cross-process
readability, the corrupt-is-a-blocker policy, and the real capture ordering.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import phases
from daydream.backends import ResultEvent, TextEvent
from daydream.deep.repair_job import merge_repair_job, read_repair_job, record_diagnostic
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.phases.repair_checkpoint import (
    REPAIR_CHECKPOINT_FORMAT,
    RepairCheckpoint,
    read_repair_checkpoint,
    write_repair_checkpoint,
)
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import commit as git_commit, git, init_repo

REPO_ROOT = Path(__file__).resolve().parent.parent

_RESULT = ResultEvent(structured_output=None, continuation=None)


def _checkpoint(**overrides: Any) -> RepairCheckpoint:
    """Build one checkpoint; every field except the identity is a keyword default."""
    fields: dict[str, Any] = {
        "job_id": "repair-job-0001",
        "execution_id": "repair-job-0001:execution:1",
        "candidate_patch": "diff --git a/allowed.py b/allowed.py\n+repair edit\n",
        "base_tree_key": "tree-base",
        "retained_tree_key": "tree-retained",
        "authorized_scope": ("allowed.py", "helper.py"),
        "policy_revision": 3,
        "failure_identity": "sha256:0123456789abcdef",
        "test_command": "pytest -q",
        "test_cwd": ".",
        "focused_results": ("1 failed, 0 passed",),
        "completed_experiments": (),
        "disproven_hypotheses": ("missing import",),
        "next_experiment": None,
        "backend_name": "scripted",
        "model": "test-model",
        "consumed_budget": {"wall_budget_s": 1800.0, "elapsed_s": 0.3},
    }
    fields.update(overrides)
    return RepairCheckpoint(**fields)


def test_checkpoint_payload_carries_every_required_field() -> None:
    """Requirement 25: the field list is the contract, not a summary."""
    payload = _checkpoint().payload()
    for field in ("job_id", "execution_id", "candidate_patch", "base_tree_key",
                  "retained_tree_key", "authorized_scope", "policy_revision",
                  "failure_identity", "test_command", "test_cwd", "focused_results",
                  "completed_experiments", "disproven_hypotheses", "next_experiment",
                  "backend_name", "model", "consumed_budget"):
        assert field in payload, field


def test_checkpoint_written_atomically_with_digests_and_format_version(tmp_path: Path) -> None:
    """Requirement 27: format_version + digests, written atomically."""
    path = write_repair_checkpoint(tmp_path, _checkpoint())
    stored = json.loads(path.read_text())
    assert stored["format_version"] == REPAIR_CHECKPOINT_FORMAT
    assert stored["patch_digest"] == hashlib.sha256(
        stored["candidate_patch"].encode()).hexdigest()
    assert not list(tmp_path.glob("*.tmp")), "atomic write must leave no staging file behind"


def test_checkpoint_is_readable_from_a_separate_interpreter(tmp_path: Path) -> None:
    """Requirements 27/48: cross-process readability, no inherited in-memory state."""
    write_repair_checkpoint(tmp_path, _checkpoint())
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys;from pathlib import Path;"
         "sys.path.insert(0,'.');"
         "from daydream.phases.repair_checkpoint import read_repair_checkpoint;"
         "print(read_repair_checkpoint(Path(sys.argv[1])).checkpoint.payload()['job_id'])",
         str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "repair-job-0001"


def test_corrupt_checkpoint_is_a_recovery_blocker_not_an_empty_job(tmp_path: Path) -> None:
    """Requirement 6/47: corrupt must never read as 'no repair happened'."""
    (tmp_path / "repair-checkpoint.json").write_text("{not json")
    result = read_repair_checkpoint(tmp_path)
    assert result.blocked is True
    assert result.reason is not None and "malformed" in result.reason


def test_absent_checkpoint_is_not_a_blocker(tmp_path: Path) -> None:
    """A repair job that never started has no checkpoint, which is not corruption."""
    result = read_repair_checkpoint(tmp_path)
    assert result.blocked is False
    assert result.checkpoint is None
    assert result.reason is None


def test_unwritable_checkpoint_is_a_blocker_not_a_silent_clean_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 28: a checkpoint the host cannot persist blocks the repair."""
    monkeypatch.setattr(
        "daydream.phases.repair_checkpoint.atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        write_repair_checkpoint(tmp_path, _checkpoint())
    assert not list(tmp_path.glob("*.tmp")), "a failed write leaves no staging file behind"


def test_job_record_merges_a_diagnostic_instead_of_clobbering_state(tmp_path: Path) -> None:
    """The read-modify-write shape: a later merge keeps an earlier execution's evidence."""
    first = merge_repair_job(tmp_path, {
        "job_id": "repair-s1", "execution_count": 1, "consumed_budget": {"elapsed_s": 1.5},
    })
    assert first.state == "active"
    second = record_diagnostic(tmp_path, "repair-s1", "checkpoint_write_failed: OSError: disk full")
    assert second is not None
    assert second.job_id == "repair-s1"
    assert second.execution_count == 1, "the diagnostic merge dropped the earlier execution"
    assert second.consumed_budget == {"elapsed_s": 1.5}
    assert second.diagnostics == ("checkpoint_write_failed: OSError: disk full",)
    assert read_repair_job(tmp_path) == second


def test_corrupt_job_record_is_not_read_as_an_empty_job(tmp_path: Path) -> None:
    """A foreign or corrupt record is reported as absent, never as fresh state."""
    assert read_repair_job(tmp_path) is None
    (tmp_path / "repair-job.json").write_text("{not json")
    assert read_repair_job(tmp_path) is None


@pytest.mark.asyncio
async def test_phase_blocks_the_repair_when_the_checkpoint_cannot_be_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    silence_console: Callable[..., None],
) -> None:
    """Requirement 28: an uncaptured repair is a repair that never happened."""
    silence_console("daydream.ui")
    init_repo(tmp_path)
    (tmp_path / "allowed.py").write_text("-- original\n")
    git(tmp_path, "add", "allowed.py")
    git_commit(tmp_path, "test: seed authorized file")

    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")
    monkeypatch.setattr(
        "daydream.phases.testing.write_repair_checkpoint",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    fail_turn = [TextEvent(text="1 failed, 0 passed"), _RESULT]

    def responder(_cwd: Path, prompt: str, *_rest: Any) -> Any:
        return tuple(fail_turn) if prompt.lower().startswith("the tests failed") else None

    backend = ScriptedBackend(responder=responder)
    footprint = AuthorizedFixFootprint(
        run_allowed_paths=frozenset({"allowed.py"}), policy_revision=1,
    )
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), session_id="s1",
        capture_tree_key=lambda: "tree-1", footprint=footprint, allow_standalone=True,
    )

    assert (result.passed, result.proceed, result.retries) == (False, False, 1)
    assert result.repairs[0].checkpoint_ref is None
    assert any("checkpoint_write_failed" in diagnostic and "OSError" in diagnostic
        for diagnostic in result.repairs[0].diagnostics)
    # The blocker is named in the job record too, so a resuming job sees it.
    job = read_repair_job(tmp_path / ".daydream" / "deep")
    assert job is not None
    assert any("checkpoint_write_failed" in diagnostic for diagnostic in job.diagnostics)


@pytest.mark.asyncio
async def test_checkpoint_captures_authorized_work_before_the_guard_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    silence_console: Callable[..., None],
) -> None:
    """Requirements 25/28: cleanup can never erase the only copy."""
    silence_console("daydream.ui")
    init_repo(tmp_path)
    (tmp_path / "allowed.py").write_text("-- original\n")
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "0001_init.sql").write_text("-- generated original\n")
    git(tmp_path, "add", "allowed.py", "migrations/0001_init.sql")
    git_commit(tmp_path, "test: seed authorized and generated files")

    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")

    async def repair(**_kwargs: Any) -> AsyncIterator[TextEvent]:
        # The spike's confirmed sequence: authorized edit, authorized new file,
        # then a stall to the wall budget with no terminal ResultEvent.
        (tmp_path / "allowed.py").write_text("-- repair edit\n")
        (tmp_path / "helper.py").write_text("# new helper\n")
        while True:
            yield TextEvent(text="PARTIAL-DIAGNOSIS")
            await anyio.sleep(0)

    backend = ScriptedBackend(responder=lambda cwd, prompt, *_: repair()
                              if prompt.lower().startswith("the tests failed") else None)
    footprint = AuthorizedFixFootprint(
        run_allowed_paths=frozenset({"allowed.py", "helper.py", "migrations/0001_init.sql"}),
        policy_revision=1,
    )
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), session_id="s1",
        capture_tree_key=lambda: "tree-1", footprint=footprint,
        allow_standalone=True,
    )

    assert (result.passed, result.proceed) == (False, False)
    read = read_repair_checkpoint(tmp_path / ".daydream" / "deep")
    assert read.blocked is False, read.reason
    checkpoint = read.checkpoint
    assert checkpoint is not None
    # Both authorized artifacts the interrupted turn produced are in the capture,
    # read from the live tree before the guard could restore anything.
    assert "repair edit" in checkpoint.candidate_patch
    assert "helper.py" in checkpoint.candidate_patch
    # The capture is the interrupted turn's own identity, not a later attempt's.
    assert checkpoint.execution_id == f"repair-s1:execution:{result.retries}"
    assert checkpoint.authorized_scope == ("allowed.py", "helper.py", "migrations/0001_init.sql")
    assert checkpoint.backend_name == "scripted"
    assert checkpoint.base_tree_key == "tree-1"
    assert checkpoint.next_experiment, "a bounded follow-up must be named for the resuming job"
    # The stored payload is verifiable on its own terms.
    stored = json.loads(
        (tmp_path / ".daydream" / "deep" / "repair-checkpoint.json").read_text(encoding="utf-8")
    )
    assert stored["patch_digest"] == hashlib.sha256(
        stored["candidate_patch"].encode()).hexdigest()
    assert result.repairs[0].checkpoint_ref == "repair-checkpoint.json"
