"""Real-path tests for the ``daydream post-findings`` verb.

Every test enters from ``cli.main`` (sys.argv patched — the production
entrypoint) with ``gh`` faked in-process at the ``subprocess.run`` boundary
(``tests/harness/fake_gh.py``), so ``git_ops._run_gh``, the ``gh_api``
tempfile-``--input`` path, and JSON parsing all run for real. Only the
GitHub network boundary (the ``gh`` process) is faked — synchronously, with
no fork and no clock, so these tests are deterministic under any host load.

Assertions are on observable outcomes: exit codes, the review payloads that
crossed the ``gh`` boundary, and the GraphQL mutations issued — never on
in-process bookkeeping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from daydream import cli
from daydream.findings import FINDINGS_SCHEMA_VERSION, write_findings_artifact
from daydream.pr_review import parse_finding_markers


def cli_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    raise AssertionError("cli.main() must exit via sys.exit")


def _post_argv(artifact: Path, *, pr: int = 7) -> list[str]:
    """The ``post-findings`` argv for *artifact*; override only what a test varies."""
    return ["post-findings", str(artifact), "--pr", str(pr), "--head-sha", "h" * 40, "--repo", "o/r"]


def _finding(fingerprint: str, *, path: str, line: int | None, placement: str, title: str,
             severity: str = "high") -> dict:
    return {
        "fingerprint": fingerprint,
        "path": path,
        "line": line,
        "placement": placement,
        "title": title,
        "body": "Body text",
        "severity": severity,
        "confidence": "HIGH",
        "is_cross_stack": False,
    }


def _write_artifact(path: Path, findings: list[dict]) -> Path:
    """Build a valid artifact via write_findings_artifact."""
    write_findings_artifact(
        path,
        {
            "schema_version": FINDINGS_SCHEMA_VERSION,
            "repo": "o/r",
            "pr_number": 7,
            "head_sha": "h" * 40,
            "run_info": "test run info",
            "findings": findings,
        },
    )
    return path


@pytest.fixture
def artifact_on_disk(tmp_path: Path) -> Path:
    """One inline + one body-only finding (both marker paths exercised)."""
    return _write_artifact(
        tmp_path / "findings.json",
        [
            _finding("a" * 64, path="a.py", line=3, placement="inline", title="Inline finding"),
            _finding("b" * 64, path="b.py", line=None, placement="body", title="Body finding"),
        ],
    )


@pytest.fixture
def artifact_on_disk_v2(tmp_path: Path) -> Path:
    """A later run: the prior ``a``-finding is gone, one new finding appears."""
    return _write_artifact(
        tmp_path / "findings_v2.json",
        [
            _finding("c" * 64, path="c.py", line=5, placement="inline", title="New finding"),
        ],
    )


def test_fresh_post_then_idempotent_repost(fake_gh, artifact_on_disk) -> None:
    argv = _post_argv(artifact_on_disk) + ["--bot-login", "daydream"]
    assert cli_main(argv) == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert parse_finding_markers(json.dumps(posts[0].payload))  # markers shipped
    # Replay the ACTUAL posted review as prior state: inline comments become
    # GraphQL threads, the review body becomes a REST review. Author is set to
    # the bot login so both harvest paths trust it via the [bot]-tolerant
    # comparator. This proves the wire-format round trip (poster emits ->
    # harvester reads) through the real CLI, on the real posted payload —
    # not just unit-level fabricated markers.
    fake_gh.serve_prior_threads_from(posts[0], author="daydream[bot]")
    assert cli_main(argv) == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1  # no dup review


def test_stale_finding_resolved_new_finding_posted(fake_gh, artifact_on_disk_v2) -> None:
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True
    )
    assert cli_main(_post_argv(artifact_on_disk_v2)) == 0
    # Task 0 spike: resolveReviewThread is FORBIDDEN for the least-privilege
    # installation token; stale findings are minimized via minimizeComment.
    assert any("minimizeComment" in c.payload.get("query", "")
               for c in fake_gh.calls("POST", "graphql"))
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def test_event_artifact_mismatch_aborts_with_no_side_effects(fake_gh, artifact_on_disk) -> None:
    rc = cli_main(_post_argv(artifact_on_disk, pr=8))  # event says 8
    assert rc == 1
    assert fake_gh.calls("POST") == []  # nothing posted, nothing resolved


def test_malformed_artifact_aborts(fake_gh, tmp_path) -> None:
    bad = tmp_path / "f.json"
    bad.write_text("{not json")
    rc = cli_main(_post_argv(bad))
    assert rc == 1 and fake_gh.calls("POST") == []


def _forged_marker_argv(artifact: Path, *extra: str) -> list[str]:
    """``post-findings`` argv for a single-finding artifact, plus extra flags."""
    return ["post-findings", str(artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r", *extra]


def _write_single_finding_artifact(path: Path, fingerprint: str) -> Path:
    return _write_artifact(
        path / "findings.json",
        [_finding(fingerprint, path="src/app.py", line=10, placement="inline", title="Real finding")],
    )


def test_forged_marker_from_non_bot_commenter_does_not_suppress_finding(
    fake_gh, tmp_path
) -> None:
    # Prior thread carries the SAME fingerprint, but authored by a human -> forged.
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_X"], authors=["evil-attacker"]
    )
    code = cli_main(_forged_marker_argv(artifact, "--bot-login", "daydream"))
    assert code == 0
    # Not suppressed -> review still posted once.
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def test_bot_authored_marker_with_bot_login_suppresses_repost(fake_gh, tmp_path) -> None:
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_X"], authors=["daydream[bot]"]
    )
    code = cli_main(_forged_marker_argv(artifact, "--bot-login", "daydream"))
    assert code == 0
    # Already on the PR -> NO review posted (idempotent).
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 0


def test_bot_login_env_fallback(monkeypatch, fake_gh, tmp_path) -> None:
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_X"], authors=["daydream[bot]"]
    )
    monkeypatch.setenv("DAYDREAM_BOT_HANDLE", "daydream")  # no --bot-login flag
    code = cli_main(_forged_marker_argv(artifact))  # env supplies the login
    assert code == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 0


def test_post_findings_approve_when_clean_and_flag(fake_gh, tmp_path) -> None:
    """low-severity-only artifact + --approve-on-clean -> review event APPROVE."""
    artifact = _write_artifact(tmp_path / "f.json", [
        _finding("a" * 64, path="a.py", line=3, placement="inline",
                 title="Nit", severity="low"),
    ])
    code = cli_main(_post_argv(artifact) + ["--approve-on-clean"])
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "APPROVE"
    assert "no high/medium findings" in posts[0].payload["body"]


def test_post_findings_keeps_comment_when_high_finding(fake_gh, tmp_path) -> None:
    """high-severity finding + --approve-on-clean -> event stays COMMENT."""
    artifact = _write_artifact(tmp_path / "f.json", [
        _finding("a" * 64, path="a.py", line=3, placement="inline",
                 title="Real finding"),  # default severity="high"
    ])
    code = cli_main(_post_argv(artifact) + ["--approve-on-clean"])
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "COMMENT"
    assert "no high/medium findings" not in posts[0].payload["body"]
