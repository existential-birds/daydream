#!/usr/bin/env python3
"""Tests for replay._is_transient.

Standalone (stdlib unittest) — the bench scripts are not wired into the package
pytest suite. Run directly:  python3 test_replay.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import replay
from replay import _findings_complete, _is_transient, replay_one

# A real `PiError: terminated` Backend-Execution-Error panel as captured in
# daydream's stdout when the z.ai/GLM provider drops the streaming connection.
PIERROR_TERMINATED_STDOUT = """\
Reviewing changed files...

╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ PiError: terminated
╚═══════════════════════════════════════════════════════════════════════════

╔═ ⚠️  Fatal Error ═════════════════════════════════════════════════════════
║ terminated
╚═══════════════════════════════════════════════════════════════════════════
"""

ECONNRESET_STDOUT = """\
╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ PiError: read ECONNRESET
╚═══════════════════════════════════════════════════════════════════════════
"""

SOCKET_HANG_UP_STDOUT = """\
╔═ ⚠️  Fatal Error ═════════════════════════════════════════════════════════
║ socket hang up
╚═══════════════════════════════════════════════════════════════════════════
"""

OVERLOADED_429_STDOUT = """\
╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ 429 service may be temporarily overloaded, please try again later
╚═══════════════════════════════════════════════════════════════════════════
"""

# A clean, successful review with NO error. Deliberately mentions "error" in
# benign prose ("error handling", "ECONNRESET") to guard against over-broad
# matching that would retry a perfectly good review forever.
CLEAN_SUCCESS_STDOUT = """\
Review complete. 2 findings written to findings.json.

  - [medium] The new fetch() call has no error handling; a dropped
    connection (e.g. ECONNRESET) would crash the worker. Wrap it in
    try/except and log the failure.
  - [low] The "terminated" status string is hardcoded; prefer an enum.

Done.
"""


class IsTransientTest(unittest.TestCase):
    def test_pierror_terminated_is_transient(self) -> None:
        # Regression guard: the stream-drop that motivated this fix MUST retry.
        self.assertTrue(_is_transient(PIERROR_TERMINATED_STDOUT))

    def test_econnreset_is_transient(self) -> None:
        self.assertTrue(_is_transient(ECONNRESET_STDOUT))

    def test_socket_hang_up_is_transient(self) -> None:
        self.assertTrue(_is_transient(SOCKET_HANG_UP_STDOUT))

    def test_429_overload_still_transient(self) -> None:
        self.assertTrue(_is_transient(OVERLOADED_429_STDOUT))

    def test_clean_success_is_not_transient(self) -> None:
        # Mentions "error"/"ECONNRESET"/"terminated" in prose but did NOT error.
        self.assertFalse(_is_transient(CLEAN_SUCCESS_STDOUT))

    def test_empty_is_not_transient(self) -> None:
        self.assertFalse(_is_transient(""))


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


# A complete review: findings artifact with a findings list + a trajectory whose
# final_metrics is populated. This is exactly what's on disk after a tail-end
# stream-drop (the review finished; only the closing socket died).
COMPLETE_FINDINGS = {"schema_version": 1, "findings": [{"id": "a"}, {"id": "b"}]}
COMPLETE_TRAJ = {"final_metrics": {"total_prompt_tokens": 100, "total_completion_tokens": 50,
                                   "total_cached_tokens": 0, "total_cost_usd": 0.0}}


class FindingsCompleteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.f = self.tmp / "findings.json"
        self.t = self.tmp / "traj.json"

    def test_complete_when_both_present(self) -> None:
        _write(self.f, COMPLETE_FINDINGS)
        _write(self.t, COMPLETE_TRAJ)
        self.assertTrue(_findings_complete(self.f, self.t))

    def test_incomplete_when_findings_missing(self) -> None:
        _write(self.t, COMPLETE_TRAJ)
        self.assertFalse(_findings_complete(self.f, self.t))

    def test_incomplete_when_findings_has_no_list(self) -> None:
        _write(self.f, {"schema_version": 1})  # no "findings" array
        _write(self.t, COMPLETE_TRAJ)
        self.assertFalse(_findings_complete(self.f, self.t))

    def test_incomplete_when_traj_has_no_metrics(self) -> None:
        _write(self.f, COMPLETE_FINDINGS)
        _write(self.t, {"steps": []})  # trajectory written but no final_metrics
        self.assertFalse(_findings_complete(self.f, self.t))


def _fake_git(source, args, *, check=True, timeout=None):
    # Every git call succeeds; merge-base / rev-parse yield a usable sha.
    return types.SimpleNamespace(returncode=0, stdout="deadbeef\n", stderr="")


def _arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


class ReplayRetryTest(unittest.TestCase):
    """Drive the real replay_one retry loop; stub only git/ensure_commit/run."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "src"
        self.source.mkdir()
        self.out = self.tmp / "out"
        self.record = {"pr_number": 1295, "review_commit_id": "deadbeef", "base_ref": "main"}
        self._orig = (replay.git, replay.ensure_commit, replay.run)
        replay.git = _fake_git
        replay.ensure_commit = lambda *a, **k: None

    def tearDown(self) -> None:
        replay.git, replay.ensure_commit, replay.run = self._orig

    def test_tail_drop_with_complete_findings_does_not_retry(self) -> None:
        calls = {"n": 0}

        def fake_run(cmd, *, cwd=None, check=False, timeout=None, env=None):
            # daydream "completes" (writes findings + metrics) but the closing
            # stream dies — exits non-zero with a PiError: terminated panel.
            calls["n"] += 1
            _write(Path(_arg(cmd, "--findings-out")), COMPLETE_FINDINGS)
            _write(Path(_arg(cmd, "--trajectory")), COMPLETE_TRAJ)
            return types.SimpleNamespace(
                returncode=1, stdout=PIERROR_TERMINATED_STDOUT, stderr="")

        replay.run = fake_run
        res = replay_one(self.record, self.source, self.out, "pi", False, 10,
                         retries=3, backoff=0)

        self.assertEqual(calls["n"], 1, "must NOT re-run a completed review")
        self.assertEqual(res["attempts"], 1)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["stream_dropped_at_tail"])
        self.assertEqual(res["n_findings"], 2)
        self.assertEqual(res["completion_tokens"], 50)

    def test_resume_skips_run_when_findings_already_on_disk(self) -> None:
        # Pre-seed a complete review (as a prior killed run would leave behind).
        _write((self.out / "findings" / "pr-1295.json"), COMPLETE_FINDINGS)
        _write((self.out / "traj" / "pr-1295.json"), COMPLETE_TRAJ)

        def fail_run(*a, **k):
            raise AssertionError("run() must not be called when findings exist")

        replay.run = fail_run
        res = replay_one(self.record, self.source, self.out, "pi", False, 10,
                         retries=3, backoff=0)

        self.assertTrue(res["reused_existing"])
        self.assertEqual(res["attempts"], 0)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_findings"], 2)

    def test_worktree_removed_when_run_raises(self) -> None:
        # run(..., timeout=) raises TimeoutExpired straight through replay_one.
        # The detached worktree must still be reclaimed in the finally block,
        # or leaked worktrees accumulate across a long batch.
        removes = []

        def tracking_git(source, args, *, check=True, timeout=None):
            if args[:2] == ["worktree", "remove"]:
                removes.append(args)
            return types.SimpleNamespace(returncode=0, stdout="deadbeef\n", stderr="")

        def boom_run(cmd, *, cwd=None, check=False, timeout=None, env=None):
            raise subprocess.TimeoutExpired(cmd, 10)

        replay.git = tracking_git
        replay.run = boom_run
        with self.assertRaises(subprocess.TimeoutExpired):
            replay_one(self.record, self.source, self.out, "pi", False, 10,
                       retries=0, backoff=0)

        # Two removes: the pre-add cleanup AND the post-failure finally cleanup.
        # Pre-fix there was only the pre-add remove (1) on the exception path.
        self.assertEqual(len(removes), 2, "worktree must be removed in finally")

    def test_transient_drop_without_findings_does_retry(self) -> None:
        # No findings ever written -> the loop must exhaust its retries.
        calls = {"n": 0}

        def fake_run(cmd, *, cwd=None, check=False, timeout=None, env=None):
            calls["n"] += 1
            return types.SimpleNamespace(
                returncode=1, stdout=PIERROR_TERMINATED_STDOUT, stderr="")

        replay.run = fake_run
        res = replay_one(self.record, self.source, self.out, "pi", False, 10,
                         retries=2, backoff=0)

        self.assertEqual(calls["n"], 3, "1 initial + 2 retries")
        self.assertEqual(res["status"], "daydream-exit-1")
        self.assertTrue(res["transient_error"])


if __name__ == "__main__":
    unittest.main()
