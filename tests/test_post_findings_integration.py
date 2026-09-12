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

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream import cli
from daydream.findings import FINDINGS_SCHEMA_VERSION, write_findings_artifact
from daydream.pr_review import parse_finding_markers, validate_diagram_payload
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, init_repo


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


def _console_text(capsys: pytest.CaptureFixture[str]) -> str:
    """Captured stdout with Rich panel borders and line wrapping collapsed.

    ``print_warning`` renders a bordered panel, so a message longer than the
    panel width is broken across lines with box characters between the halves
    — a substring check against the raw capture fails on the wrap.
    """
    out = capsys.readouterr().out
    return " ".join(out.translate({ord(char): " " for char in "│╭╮╰╯─║╔╗╚╝═"}).split())


def _post_argv(
    artifact: Path, *, pr: int = 7, head_sha: str | None = None, target: Path | None = None,
) -> list[str]:
    """The ``post-findings`` argv for *artifact*; override only what a test varies."""
    return [
        "post-findings",
        str(artifact),
        "--pr",
        str(pr),
        "--head-sha",
        head_sha or "h" * 40,
        "--repo",
        "o/r",
        *(["--target", str(target)] if target is not None else []),
    ]


def test_post_findings_uses_two_explicit_checkout_configs_without_chdir(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    ambient = Path.cwd()
    targets = [tmp_path / "checkout A", tmp_path / "checkout B"]
    for target in targets:
        init_repo(target)
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_OLD"], viewer_did_author=True,
    )
    for index, target in enumerate(targets):
        (target / ".daydream.toml").write_text(f"approve_on_clean = {'true' if index == 0 else 'false'}\n")
        artifact = _write_artifact(target / "findings.json", [
            _finding(
                ("b" if index == 0 else "c") * 64, path="a.py", line=1,
                placement="inline", title=f"Checkout {index} finding", severity="low",
            ),
        ])
        before = len(fake_gh.process_calls())
        # The second invocation pins relative paths with spaces against the
        # unchanged ambient cwd, not against the artifact or first checkout.
        argument = target if index == 0 else Path(os.path.relpath(target, ambient))
        assert cli_main(_post_argv(artifact, target=argument)) == 0
        processes = fake_gh.process_calls()[before:]
        assert processes and all(call.cwd == target.resolve() for call in processes)
        assert Path.cwd() == ambient
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert [call.payload["event"] for call in posts] == ["APPROVE", "COMMENT"]
    assert [call.payload["comments"][0]["path"] for call in posts] == ["a.py", "a.py"]
    assert sum("minimizeComment" in call.payload.get("query", "")
               for call in fake_gh.calls("POST", "graphql")) == 2


@pytest.mark.parametrize("target_kind", ["missing", "file"])
def test_post_findings_rejects_invalid_target_before_github(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str], target_kind: str,
) -> None:
    target = tmp_path / target_kind
    if target_kind == "file":
        target.write_text("not a directory")
    artifact = _write_artifact(tmp_path / "findings.json", [])
    assert cli_main(_post_argv(artifact, target=target)) == 2
    assert "--target must be an existing directory" in capsys.readouterr().err
    assert fake_gh.process_calls() == []


def test_post_findings_omitted_target_preserves_invocation_directory(
    fake_gh: FakeGh, artifact_on_disk: Path,
) -> None:
    assert cli_main(_post_argv(artifact_on_disk)) == 0
    assert fake_gh.process_calls()
    assert all(call.cwd == Path.cwd().resolve() for call in fake_gh.process_calls())
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def test_post_findings_reads_valid_diagram_evidence_from_explicit_target(
    fake_gh: FakeGh, git_repo: Path,
) -> None:
    ambient = Path.cwd()
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head = commit(git_repo, "seed diagram evidence")
    artifact = _write_artifact(
        git_repo / "findings.json", [], diagrams=_flowchart_payload(), head_sha=head,
    )
    assert cli_main(_post_argv(artifact, head_sha=head, target=git_repo)) == 0
    assert Path.cwd() == ambient
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "```mermaid" in posts[0].payload["body"]
    assert posts[0].payload["commit_id"] == head
    assert fake_gh.process_calls()
    assert all(call.cwd == git_repo.resolve() for call in fake_gh.process_calls())
    # A checkout that already holds the head commit is read locally: the
    # contents-API fallback (issue #1167) is for targets that do not.
    assert _contents_calls(fake_gh) == []


def _finding(
    fingerprint: str,
    *,
    path: str,
    line: int | None,
    placement: str,
    title: str,
    severity: str = "high",
) -> dict[str, Any]:
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


def _write_artifact(
    path: Path,
    findings: list[dict[str, Any]],
    *,
    run_info: str = "test run info",
    diagrams: dict[str, Any] | None = None,
    head_sha: str = "h" * 40,
) -> Path:
    """Build a valid artifact via write_findings_artifact."""
    write_findings_artifact(
        path,
        {
            "schema_version": FINDINGS_SCHEMA_VERSION,
            "repo": "o/r",
            "pr_number": 7,
            "head_sha": head_sha,
            "run_info": run_info,
            "diagrams": diagrams,
            "findings": findings,
        },
    )
    return path


def _flowchart_payload() -> dict[str, Any]:
    def grounded(element: str, ref: str, final_index: int) -> dict[str, Any]:
        return {
            "element": element,
            "ref": ref,
            "grounded": True,
            "reason": None,
            "strength": "definition",
            "snapped_line": None,
            "in_changed_hunk": True,
            "defined_at": "a.py:1",
            "final_index": final_index,
        }

    return {
        "results": {
            "flowchart": {
                "status": "rendered",
                "reason": None,
                "omit_reasons": [],
                "spec_final": {
                    "root": {"file": "a.py", "name": "run", "line": 1},
                    "nodes": [
                        {
                            "id": "start",
                            "kind": "start",
                            "label": "run",
                            "evidence": {
                                "file": "a.py",
                                "line": 1,
                                "symbol": "run",
                            },
                        },
                        {
                            "id": "end",
                            "kind": "end",
                            "label": "return",
                            "evidence": {
                                "file": "a.py",
                                "line": 2,
                                "symbol": None,
                            },
                        },
                    ],
                    "edges": [{"from": "start", "to": "end", "label": None}],
                },
                "grounding": {
                    "elements": [
                        grounded("root", "run", 0),
                        grounded("node", "start", 0),
                        grounded("node", "end", 1),
                        grounded("edge", "start->end", 0),
                    ],
                    "summary": {
                        "proposed": 4,
                        "grounded_first_pass": 4,
                        "repaired": 0,
                        "pruned": 0,
                    },
                    "capped": {},
                    "root_range": [1, 2],
                },
            }
        }
    }


def _sequence_payload() -> dict[str, Any]:
    def grounded(element: str, ref: str, final_index: int) -> dict[str, Any]:
        return {
            "element": element,
            "ref": ref,
            "grounded": True,
            "reason": None,
            "strength": "definition",
            "snapped_line": None,
            "in_changed_hunk": True,
            "defined_at": "a.py:1",
            "final_index": final_index,
        }

    return {
        "results": {
            "sequence": {
                "status": "rendered",
                "reason": None,
                "omit_reasons": [],
                "spec_final": {
                    "participants": [
                        {
                            "name": "api",
                            "kind": "internal",
                            "files": ["api.py"],
                            "service": None,
                        },
                        {
                            "name": "worker",
                            "kind": "internal",
                            "files": ["worker.py"],
                            "service": None,
                        },
                    ],
                    "messages": [
                        {
                            "from": "api",
                            "to": "worker",
                            "label": "call",
                            "kind": "call",
                            "changed": True,
                            "evidence": {
                                "file": "api.py",
                                "line": 3,
                                "symbol": "worker",
                            },
                        },
                        {
                            "from": "worker",
                            "to": "api",
                            "label": "reply",
                            "kind": "reply",
                            "changed": True,
                            "evidence": {
                                "file": "worker.py",
                                "line": 2,
                                "symbol": "worker",
                            },
                        },
                        {
                            "from": "api",
                            "to": "worker",
                            "label": "call again",
                            "kind": "call",
                            "changed": True,
                            "evidence": {
                                "file": "api.py",
                                "line": 5,
                                "symbol": "worker",
                            },
                        },
                    ],
                    "blocks": [
                        {
                            "kind": "opt",
                            "branches": [
                                {
                                    "condition": "enabled",
                                    "evidence": {"file": "api.py", "line": 4},
                                    "messages": [2],
                                }
                            ],
                        }
                    ],
                },
                "grounding": {
                    "elements": [
                        grounded("participant", "api", 0),
                        grounded("participant", "worker", 1),
                        grounded("message", "0", 0),
                        grounded("message", "1", 1),
                        grounded("message", "2", 2),
                        grounded("block", "b0", 0),
                        grounded("branch", "b0.0", 0),
                    ],
                    "summary": {
                        "proposed": 7,
                        "grounded_first_pass": 7,
                        "repaired": 0,
                        "pruned": 0,
                    },
                    "capped": {},
                },
            }
        }
    }


@pytest.fixture
def artifact_on_disk(tmp_path: Path) -> Path:
    """One inline + one body-only finding (both marker paths exercised)."""
    return _write_artifact(
        tmp_path / "findings.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Inline finding",
            ),
            _finding(
                "b" * 64, path="b.py", line=None, placement="body", title="Body finding"
            ),
        ],
    )


@pytest.fixture
def artifact_on_disk_second(tmp_path: Path) -> Path:
    """A later run: the prior ``a``-finding is gone, one new finding appears."""
    return _write_artifact(
        tmp_path / "findings_second.json",
        [
            _finding(
                "c" * 64, path="c.py", line=5, placement="inline", title="New finding"
            ),
        ],
    )


def test_post_findings_body_names_cli_head_sha(
    fake_gh: FakeGh, artifact_on_disk: Path
) -> None:
    """M3 CI path: the posted body's reviewed-commit line names the
    --head-sha given on the CLI (validated event data), never the
    artifact's untrusted run_info string."""
    assert cli_main(_post_argv(artifact_on_disk)) == 0
    body = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")[0].payload["body"]
    assert (
        "- **Reviewed commit:** [`hhhhhhh`]"
        "(https://github.com/o/r/commit/" + "h" * 40 + ")"
    ) in body


def test_post_findings_ignores_artifact_run_info_sha(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    """The CLI --head-sha wins: a different 40-char SHA embedded in the
    artifact's run_info string must never appear in the reviewed-commit
    line — not even as a fully formatted forged line (issue 2)."""
    artifact = _write_artifact(
        tmp_path / "findings.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Inline finding",
            )
        ],
        run_info=(
            "run from commit "
            + "a" * 40
            + "\n- **Reviewed commit:** [`deadbee`](https://github.com/evil/widgets/commit/"
            + "e" * 40
            + ")"
        ),
    )
    assert cli_main(_post_argv(artifact)) == 0
    body = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")[0].payload["body"]
    commit_lines = [
        line for line in body.splitlines() if line.startswith("- **Reviewed commit:**")
    ]
    assert len(commit_lines) == 1
    assert "a" * 40 not in commit_lines[0]
    # The forged formatted line is stripped in full — its sha and slug must
    # not survive anywhere in the posted body.
    assert "e" * 40 not in body
    assert "evil/widgets" not in body
    assert (
        "- **Reviewed commit:** [`hhhhhhh`]"
        "(https://github.com/o/r/commit/" + "h" * 40 + ")"
    ) in body


def test_fresh_post_then_idempotent_repost(
    fake_gh: FakeGh, artifact_on_disk: Path
) -> None:
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
    assert (
        len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1
    )  # no dup review


def test_stale_finding_resolved_new_finding_posted(
    fake_gh: FakeGh, artifact_on_disk_second: Path
) -> None:
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True
    )
    assert cli_main(_post_argv(artifact_on_disk_second)) == 0
    # Task 0 spike: resolveReviewThread is FORBIDDEN for the least-privilege
    # installation token; stale findings are minimized via minimizeComment.
    assert any(
        "minimizeComment" in c.payload.get("query", "")
        for c in fake_gh.calls("POST", "graphql")
    )
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def test_event_artifact_mismatch_aborts_with_no_side_effects(
    fake_gh: FakeGh, artifact_on_disk: Path
) -> None:
    rc = cli_main(_post_argv(artifact_on_disk, pr=8))  # event says 8
    assert rc == 1
    assert fake_gh.calls("POST") == []  # nothing posted, nothing resolved


def test_malformed_artifact_aborts(fake_gh: FakeGh, tmp_path: Path) -> None:
    bad = tmp_path / "f.json"
    bad.write_text("{not json")
    rc = cli_main(_post_argv(bad))
    assert rc == 1 and fake_gh.calls("POST") == []


def test_malformed_repo_config_warns_and_still_posts(
    fake_gh: FakeGh,
    tmp_path: Path,
) -> None:
    """A malformed .daydream.toml in the checkout must not abort the unattended post.

    post-findings never consulted the repo config before issue #343; the new
    approve-on-clean lookup is best-effort, so a malformed TOML degrades to a
    warning plus the CLI flag instead of a Fatal Error (exit 1).
    """
    (tmp_path / ".daydream.toml").write_text("this is [not valid toml ==")
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    code = cli_main(_forged_marker_argv(artifact, "--target", str(tmp_path)))
    assert code == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def _forged_marker_argv(artifact: Path, *extra: str) -> list[str]:
    """``post-findings`` argv for a single-finding artifact, plus extra flags."""
    return [
        "post-findings",
        str(artifact),
        "--pr",
        "7",
        "--head-sha",
        "h" * 40,
        "--repo",
        "o/r",
        *extra,
    ]


def _write_single_finding_artifact(path: Path, fingerprint: str) -> Path:
    return _write_artifact(
        path / "findings.json",
        [
            _finding(
                fingerprint,
                path="src/app.py",
                line=10,
                placement="inline",
                title="Real finding",
            )
        ],
    )


def test_forged_marker_from_non_bot_commenter_does_not_suppress_finding(
    fake_gh: FakeGh, tmp_path: Path
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


def test_bot_authored_marker_with_bot_login_suppresses_repost(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_X"], authors=["daydream[bot]"]
    )
    code = cli_main(_forged_marker_argv(artifact, "--bot-login", "daydream"))
    assert code == 0
    # Already on the PR -> NO review posted (idempotent).
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 0


def test_bot_login_env_fallback(
    monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, tmp_path: Path
) -> None:
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_X"], authors=["daydream[bot]"]
    )
    monkeypatch.setenv("DAYDREAM_BOT_HANDLE", "daydream")  # no --bot-login flag
    code = cli_main(_forged_marker_argv(artifact))  # env supplies the login
    assert code == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 0


def test_post_findings_approve_when_clean_and_flag(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    """low-severity-only artifact + --approve-on-clean -> review event APPROVE."""
    artifact = _write_artifact(
        tmp_path / "f.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Nit",
                severity="low",
            ),
        ],
    )
    code = cli_main(_post_argv(artifact) + ["--approve-on-clean"])
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "APPROVE"
    assert "no high/medium findings" in posts[0].payload["body"]


def test_post_findings_keeps_comment_when_high_finding(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    """high-severity finding + --approve-on-clean -> event stays COMMENT."""
    artifact = _write_artifact(
        tmp_path / "f.json",
        [
            _finding(
                "a" * 64, path="a.py", line=3, placement="inline", title="Real finding"
            ),  # default severity="high"
        ],
    )
    code = cli_main(_post_argv(artifact) + ["--approve-on-clean"])
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "COMMENT"
    assert "no high/medium findings" not in posts[0].payload["body"]


def test_post_findings_approve_when_all_matched_and_clean_flag(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    """F2: an all-matched clean artifact + --approve-on-clean still posts APPROVE.

    The post-findings spine previously returned 0 on its unconditional empty
    guard, so a re-run with nothing new to comment on never posted the
    approval and ``required_approving_review_count`` stayed unsatisfied — the
    headline two-phase CI use case.
    """
    artifact = _write_artifact(
        tmp_path / "f.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Nit",
                severity="low",
            ),
        ],
    )
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True
    )
    code = cli_main(
        _post_argv(artifact) + ["--approve-on-clean", "--bot-login", "daydream"]
    )
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "APPROVE"
    assert "no high/medium findings" in posts[0].payload["body"]


def test_post_findings_all_matched_no_approve_without_flag(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    """F2: without --approve-on-clean the same all-matched artifact posts nothing."""
    artifact = _write_artifact(
        tmp_path / "f.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Nit",
                severity="low",
            ),
        ],
    )
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True
    )
    code = cli_main(_post_argv(artifact) + ["--bot-login", "daydream"])
    assert code == 0
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []


def test_post_findings_drops_forged_diagram_grounding_attestation(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head_sha = commit(git_repo, "add flowchart source")
    payload = _flowchart_payload()
    flowchart = payload["results"]["flowchart"]
    flowchart["spec_final"]["nodes"] = [
        {
            "id": "start",
            "kind": "start",
            "label": "run",
            "evidence": {"file": "a.py", "line": 1, "symbol": "run"},
        },
        {
            "id": "decision",
            "kind": "decision",
            "label": "invented branch",
            "evidence": {"file": "a.py", "line": 1, "symbol": None},
        },
        {
            "id": "process",
            "kind": "process",
            "label": "invented process",
            "evidence": {"file": "a.py", "line": 1, "symbol": None},
        },
        {
            "id": "end",
            "kind": "end",
            "label": "return",
            "evidence": {"file": "a.py", "line": 2, "symbol": None},
        },
    ]
    flowchart["spec_final"]["edges"] = [
        {"from": "start", "to": "decision", "label": None},
        {"from": "decision", "to": "process", "label": "yes"},
        {"from": "process", "to": "end", "label": None},
    ]
    flowchart["grounding"]["elements"] = [
        {
            "element": element,
            "ref": ref,
            "grounded": True,
            "reason": None,
            "strength": "definition",
            "snapped_line": None,
            "in_changed_hunk": True,
            "defined_at": "a.py:1",
            "final_index": index,
        }
        for element, ref, index in (
            ("root", "run", 0),
            ("node", "start", 0),
            ("node", "decision", 1),
            ("node", "process", 2),
            ("node", "end", 3),
            ("edge", "start->decision", 0),
            ("edge", "decision->process", 1),
            ("edge", "process->end", 2),
        )
    ]
    flowchart["grounding"]["summary"] = {
        "proposed": 8,
        "grounded_first_pass": 8,
        "repaired": 0,
        "pruned": 0,
    }
    artifact = _write_artifact(
        git_repo / "f.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Already posted",
            )
        ],
        diagrams=payload,
        head_sha=head_sha,
    )
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True
    )

    code = cli_main(
        _post_argv(artifact, head_sha=head_sha, target=git_repo) + ["--bot-login", "daydream"]
    )

    assert code == 0
    assert "Diagram dropped (the findings are still posted)" in _console_text(capsys)
    # No review is posted because the artifact's one finding is already on the
    # PR, so the run reaches the no-new-findings short-circuit -- NOT because a
    # forged diagram fails the run closed, which it no longer does (#1176).
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []


def test_post_findings_drops_forged_sequence_grounding_attestation(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (git_repo / "api.py").write_text(
        "from worker import worker\n"
        "def api():\n"
        "    worker()\n"
        "    if enabled:\n"
        "        worker()\n"
    )
    (git_repo / "worker.py").write_text(
        "def worker():\n"
        "    return 1\n"
    )
    git(git_repo, "add", "api.py", "worker.py")
    head_sha = commit(git_repo, "add sequence source")
    payload = _sequence_payload()
    assert validate_diagram_payload(payload, target_dir=git_repo, head_sha=head_sha) is None
    payload["results"]["sequence"]["spec_final"]["messages"][0]["evidence"] = {
        "file": "api.py",
        "line": 2,
        "symbol": "api",
    }
    artifact = _write_artifact(
        git_repo / "findings.json", [], diagrams=payload, head_sha=head_sha
    )

    code = cli_main(_post_argv(artifact, head_sha=head_sha, target=git_repo))

    assert code == 0
    assert "Diagram dropped (the findings are still posted)" in _console_text(capsys)
    # The artifact carries no findings, so the run reaches the no-new-findings
    # short-circuit -- not the old fail-closed diagram gate (#1176).
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []


def test_post_findings_drops_diagram_evidence_absent_from_immutable_head(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head_sha = commit(git_repo, "add flowchart source")

    payload = _flowchart_payload()
    flowchart = payload["results"]["flowchart"]
    flowchart["spec_final"]["root"]["line"] = 100
    flowchart["spec_final"]["nodes"][0]["evidence"]["line"] = 100
    flowchart["spec_final"]["nodes"][1]["evidence"]["line"] = 100
    flowchart["grounding"]["root_range"] = [100, 100]
    artifact = _write_artifact(
        git_repo / "findings.json",
        [],
        diagrams=payload,
        head_sha=head_sha,
    )

    code = cli_main(
        _post_argv(artifact, head_sha=head_sha, target=git_repo) + ["--bot-login", "daydream"]
    )

    assert code == 0
    # The artifact carries no findings, so the run reaches the no-new-findings
    # short-circuit -- not the old fail-closed diagram gate (#1176).
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []
    # The local `git show` branch has proven the head commit present, so an
    # out-of-range citation is still reported as absent, never as unreadable.
    printed = _console_text(capsys)
    assert "is missing from immutable head" in printed


def test_post_findings_drops_a_citation_absent_from_the_local_checkout(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A local read git itself names as a missing path IS an absent citation.

    Guards the ``git show`` half of ``unreadable()`` — git answers "does not
    exist in <sha>", which proves absence, so the verdict must not degrade to
    "could not be read".
    """
    (git_repo / "b.py").write_text("def other():\n    return 1\n")
    git(git_repo, "add", "b.py")
    head_sha = commit(git_repo, "head without the cited file")
    artifact = _write_artifact(
        git_repo / "findings.json", [], diagrams=_flowchart_payload(), head_sha=head_sha,
    )

    code = cli_main(_post_argv(artifact, head_sha=head_sha, target=git_repo))

    assert code == 0
    printed = _console_text(capsys)
    assert "flowchart diagram evidence is missing from immutable head: a.py" in printed
    assert "could not be read from immutable head" not in printed


def test_post_findings_reports_a_damaged_local_object_as_unreadable_not_missing(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A local read that fails for any other reason is the poster's problem.

    The checkout holds the head commit and the citation, but its blob is gone
    from the object store, so ``git show`` fails without naming a missing
    path. Blaming the artifact for that sends an operator hunting a forgery.
    """
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head_sha = commit(git_repo, "add flowchart source")
    blob = git(git_repo, "rev-parse", f"{head_sha}:a.py").strip()
    (git_repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    artifact = _write_artifact(
        git_repo / "findings.json", [], diagrams=_flowchart_payload(), head_sha=head_sha,
    )

    code = cli_main(_post_argv(artifact, head_sha=head_sha, target=git_repo))

    assert code == 0
    printed = _console_text(capsys)
    assert "flowchart diagram evidence could not be read from immutable head" in printed
    assert "is missing from immutable head" not in printed


# --- Issue #1167: head evidence without a checkout ---------------------------
#
# The shipped post workflow runs `post-findings` with no `actions/checkout` at
# all, by design, so the poster reads immutable head evidence from the contents
# API at the same SHA instead of from `git show`.

_API_HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"


def _serve_contents(fake_gh: FakeGh, path: str, source: str) -> None:
    """Serve *source* as the contents-API body for *path* (the ``?ref=`` is asserted separately)."""
    fake_gh.set_response(
        "GET",
        f"repos/o/r/contents/{path}",
        value={
            "type": "file",
            "path": path,
            "sha": "a" * 40,
            "encoding": "base64",
            "content": base64.b64encode(source.encode()).decode(),
        },
    )


def _contents_calls(fake_gh: FakeGh) -> list[str]:
    """Every contents-API endpoint the run requested, in order."""
    return [
        call.endpoint
        for call in fake_gh.calls("GET")
        if call.endpoint.startswith("repos/o/r/contents/")
    ]


def test_post_findings_posts_flowchart_evidence_read_without_a_checkout(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """The regression: an empty (non-repository) target must still post."""
    _serve_contents(fake_gh, "a.py", "def run():\n    return 1\n")
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert _contents_calls(fake_gh) == [f"repos/o/r/contents/a.py?ref={_API_HEAD_SHA}"]
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "```mermaid" in posts[0].payload["body"]


def test_post_findings_posts_sequence_evidence_read_without_a_checkout(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """The sequence branch re-grounds a snapshot built from the same API reads."""
    _serve_contents(
        fake_gh,
        "api.py",
        "from worker import worker\ndef api():\n    worker()\n    if enabled:\n        worker()\n",
    )
    _serve_contents(fake_gh, "worker.py", "def worker():\n    return 1\n")
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_sequence_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert sorted(_contents_calls(fake_gh)) == [
        f"repos/o/r/contents/api.py?ref={_API_HEAD_SHA}",
        f"repos/o/r/contents/worker.py?ref={_API_HEAD_SHA}",
    ]
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "sequenceDiagram" in posts[0].payload["body"]


def test_post_findings_reads_oversized_evidence_through_the_blob_endpoint(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """Past the contents endpoint's inline ceiling GitHub answers ``encoding: none``."""
    blob_sha = "b" * 40
    fake_gh.set_response(
        "GET",
        "repos/o/r/contents/a.py",
        value={"type": "file", "path": "a.py", "sha": blob_sha, "encoding": "none", "content": ""},
    )
    fake_gh.set_response(
        "GET",
        f"repos/o/r/git/blobs/{blob_sha}",
        value={
            "sha": blob_sha,
            "encoding": "base64",
            "content": base64.b64encode(b"def run():\n    return 1\n").decode(),
        },
    )
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert fake_gh.calls("GET", f"repos/o/r/git/blobs/{blob_sha}")
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def test_post_findings_drops_api_evidence_that_contradicts_the_spec(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #1176: the contradicted diagram is dropped, the findings still post.

    Evidence fetched over the API is adjudicated exactly as a checkout's is,
    but a rejected diagram costs only the diagram — the review that carries
    the artifact's validated findings is posted without a mermaid block.
    """
    _serve_contents(fake_gh, "a.py", "x = 1\n")  # no `run` definition, one line
    artifact = _write_artifact(
        tmp_path / "findings.json",
        [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Real finding")],
        diagrams=_flowchart_payload(),
        head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "Real finding" in json.dumps(posts[0].payload)
    assert "```mermaid" not in posts[0].payload["body"]
    printed = _console_text(capsys)
    assert "Diagram dropped (the findings are still posted)" in printed


def test_post_findings_drops_evidence_absent_from_the_api_head(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A 404 at the head SHA is a missing citation, not an unreadable one."""
    fake_gh.set_response(
        "GET", "repos/o/r/contents/a.py", value={"__error__": "gh: Not Found (HTTP 404)"},
    )
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    # The artifact carries no findings, so the run reaches the no-new-findings
    # short-circuit -- not the old fail-closed diagram gate (#1176).
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []
    # gh's own 404 diagnostic is the only positive proof of absence over the
    # API, and it must survive as ``PathAbsentError`` all the way to the wording.
    printed = _console_text(capsys)
    assert "is missing from immutable head" in printed
    assert "could not be read from immutable head" not in printed


def test_post_findings_drops_a_non_file_api_response_as_a_missing_citation(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A directory (or symlink) at the cited path is the second proof of absence."""
    fake_gh.set_response(
        "GET", "repos/o/r/contents/a.py", value={"type": "dir", "path": "a.py"},
    )
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    printed = _console_text(capsys)
    assert "is missing from immutable head" in printed
    assert "could not be read from immutable head" not in printed


def test_post_findings_reports_a_throttled_read_as_unreadable_not_missing(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A rate-limited read is the poster's problem, not a forged citation."""
    fake_gh.set_response(
        "GET",
        "repos/o/r/contents/a.py",
        value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
    )
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    # The artifact carries no findings, so the run reaches the no-new-findings
    # short-circuit -- not the old fail-closed diagram gate (#1176).
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []
    printed = _console_text(capsys)
    assert "could not be read from immutable head" in printed
    assert "is missing from immutable head" not in printed


def test_post_findings_blames_the_artifact_for_a_malformed_citation_path(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A path no read could be attempted for is the artifact's defect.

    The spec schema applies its ``pattern`` with ``re.search``, whose ``$``
    matches before a trailing newline, so ``a.py\\n`` reaches the fullmatch in
    ``_HeadEvidence.read`` and fails it. No read happened, so the verdict must
    name the bad path rather than report the poster as unable to read.
    """
    payload = _flowchart_payload()
    flowchart = payload["results"]["flowchart"]
    flowchart["spec_final"]["root"]["file"] = "a.py\n"
    for node in flowchart["spec_final"]["nodes"]:
        node["evidence"]["file"] = "a.py\n"
    for element in flowchart["grounding"]["elements"]:
        element["defined_at"] = "a.py\n:1"
    artifact = _write_artifact(
        tmp_path / "findings.json",
        [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Real finding")],
        diagrams=payload,
        head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    printed = _console_text(capsys)
    assert "flowchart diagram evidence cites an invalid repository path: 'a.py\\n'" in printed
    assert "could not be read from immutable head" not in printed
    assert _contents_calls(fake_gh) == []
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "```mermaid" not in posts[0].payload["body"]


@pytest.mark.parametrize(
    ("label", "response"),
    [
        ("forbidden", {"__error__": "gh: HTTP 403: Resource not accessible by integration"}),
        ("unauthorized", {"__error__": "gh: HTTP 401: Bad credentials"}),
        ("undecodable", {"__stdout__": "<html>proxy error</html>"}),
    ],
)
def test_post_findings_reports_an_unreadable_api_head_as_unreadable(
    fake_gh: FakeGh,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    label: str,
    response: dict[str, str],
) -> None:
    """Every contents-API failure that is not proven absence reads as unreadable.

    Two of these never reach ``_gh_error_for`` as a recognizable status (the
    undecodable body is raised by ``_parse_gh_json``), which is why absence is
    claimed only where the read positively proved it.
    """
    fake_gh.set_response("GET", "repos/o/r/contents/a.py", value=response)
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    printed = _console_text(capsys)
    assert "could not be read from immutable head" in printed, label
    assert "is missing from immutable head" not in printed, label


def test_post_findings_posts_findings_when_an_unreadable_diagram_is_dropped(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #1176 for the unreadable half: a throttled read costs only the diagram.

    The poster's own inability to read head evidence must not discard findings
    that passed schema, fingerprint and event-fact validation.
    """
    fake_gh.set_response(
        "GET",
        "repos/o/r/contents/a.py",
        value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
    )
    artifact = _write_artifact(
        tmp_path / "findings.json",
        [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Real finding")],
        diagrams=_flowchart_payload(),
        head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "Real finding" in json.dumps(posts[0].payload)
    assert "```mermaid" not in posts[0].payload["body"]
    printed = _console_text(capsys)
    assert "Diagram dropped (the findings are still posted)" in printed
    assert "could not be read from immutable head" in printed


def test_post_findings_minimizes_stale_threads_on_a_degraded_artifact(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A degraded artifact takes the ordinary no-diagram path, minimization included.

    Documented consequence of the #1176 degrade: the run no longer returns
    above ``fetch_prior_findings``, so an empty findings list reconciles as it
    always has and the bot's own resolved threads are minimized.
    """
    fake_gh.set_response(
        "GET",
        "repos/o/r/contents/a.py",
        value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
    )
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True,
    )
    artifact = _write_artifact(
        tmp_path / "findings.json", [], diagrams=_flowchart_payload(), head_sha=_API_HEAD_SHA,
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert sum(
        "minimizeComment" in call.payload.get("query", "")
        for call in fake_gh.calls("POST", "graphql")
    ) == 1
    assert "Diagram dropped (the findings are still posted)" in _console_text(capsys)


def test_post_findings_approves_on_clean_with_a_degraded_artifact(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """Documented consequence of the #1176 degrade: APPROVE becomes reachable.

    ``can_approve`` reads findings only, never diagrams, so a low-severity-only
    artifact whose diagram was dropped approves exactly as a diagram-less one.
    """
    fake_gh.set_response(
        "GET",
        "repos/o/r/contents/a.py",
        value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
    )
    artifact = _write_artifact(
        tmp_path / "findings.json",
        [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Nit", severity="low")],
        diagrams=_flowchart_payload(),
        head_sha=_API_HEAD_SHA,
    )

    code = cli_main(
        _post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path) + ["--approve-on-clean"]
    )

    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "APPROVE"
    assert "```mermaid" not in posts[0].payload["body"]


def test_post_findings_matched_high_blocks_approval(
    fake_gh: FakeGh, tmp_path: Path
) -> None:
    """F2b: a still-live matched high finding blocks APPROVE.

    The approval decision must count the severities of already-posted
    (matched) findings, not just the new ones: a re-run whose only NEW
    finding is low must not post APPROVE over the bot's own open high finding
    on the PR.
    """
    artifact = _write_artifact(
        tmp_path / "f.json",
        [
            _finding(
                "a" * 64,
                path="a.py",
                line=3,
                placement="inline",
                title="Old high finding",
            ),
            _finding(
                "b" * 64,
                path="b.py",
                line=5,
                placement="inline",
                title="New nit",
                severity="low",
            ),
        ],
    )
    fake_gh.serve_prior_threads(
        fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True
    )
    code = cli_main(
        _post_argv(artifact) + ["--approve-on-clean", "--bot-login", "daydream"]
    )
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "COMMENT"
    assert "no high/medium findings" not in posts[0].payload["body"]


@pytest.mark.parametrize("run_info", ["Artifact-owned run details", None])
def test_artifact_post_never_acquires_live_trajectory_details(
    fake_gh: FakeGh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    run_info: str | None,
) -> None:
    artifact = _write_artifact(
        tmp_path / "findings.json",
        [_finding("a" * 64, path="a.py", line=3, placement="inline", title="Finding")],
    )
    document = json.loads(artifact.read_text())
    document["run_info"] = run_info
    artifact.write_text(json.dumps(document))
    attempts: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        attempts.append("live acquisition")
        raise AssertionError("artifact posting attempted live trajectory acquisition")

    with monkeypatch.context() as patch:
        patch.setattr("daydream.trajectory.get_current_recorder", forbidden)
        patch.setattr("daydream.pr_run_info.render_live_run_info", forbidden)
        patch.setattr("daydream.pr_comment_renderer.render_run_info_block", forbidden)
        patch.setattr("tempfile.TemporaryDirectory", forbidden)
        assert cli_main(_post_argv(artifact)) == 0
    assert attempts == []
    body = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")[0].payload["body"]
    assert (run_info if run_info is not None else "*run details unavailable*") in body
    assert body.count("- **Reviewed commit:**") == 1


def test_post_findings_orders_file_writes_and_folds_failed_thread_once(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _write_artifact(tmp_path / "findings.json", [
        _finding("a" * 64, path="first.py", line=None, placement="file", title="First thread"),
        _finding("b" * 64, path="second.py", line=None, placement="file", title="Folded thread"),
        _finding("c" * 64, path="context.py", line=None, placement="body", title="Original body"),
    ])
    fake_gh.set_response("diff-paths", value=["first.py"])

    assert cli_main(_post_argv(artifact)) == 0

    writes = [call for call in fake_gh.calls("POST") if call.endpoint != "graphql"]
    assert [call.endpoint for call in writes] == [
        "repos/o/r/pulls/7/comments", "repos/o/r/pulls/7/comments", "repos/o/r/pulls/7/reviews",
    ]
    assert [call.payload["path"] for call in writes[:2]] == ["first.py", "second.py"]
    assert all(call.payload["subject_type"] == "file" for call in writes[:2])
    payload = writes[-1].payload
    assert payload["commit_id"] == "h" * 40
    assert payload["event"] == "COMMENT"
    assert payload["comments"] == []
    markers = parse_finding_markers(payload["body"])
    assert markers == ["c" * 64, "b" * 64]
    assert payload["body"].index("Original body") < payload["body"].index("Folded thread")
    assert "1 file-level comment(s) failed to post; folded into the review body." in _console_text(capsys)


@pytest.mark.parametrize("file_posts", [False, True], ids=["zero-writes", "partial-write"])
def test_post_findings_final_failure_reports_writes_and_safe_recovery_path(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str], file_posts: bool,
) -> None:
    artifact = _write_artifact(tmp_path / "findings.json", [
        _finding("a" * 64, path="first.py", line=None, placement="file", title="File finding"),
    ])
    fake_gh.set_response("diff-paths", value=["first.py"] if file_posts else [])
    secret = "opaque-private-installation-credential"
    fake_gh.set_response("POST", "repos/o/r/pulls/7/reviews", {"__error__": f"HTTP 422: {secret}"})

    assert cli_main(_post_argv(artifact)) == 1

    writes = [call for call in fake_gh.calls("POST") if call.endpoint != "graphql"]
    assert [call.endpoint for call in writes] == [
        "repos/o/r/pulls/7/comments", "repos/o/r/pulls/7/reviews",
    ]
    review = writes[-1]
    assert review.argv is not None
    payload_path = Path(review.argv[review.argv.index("--input") + 1])
    try:
        assert json.loads(payload_path.read_text()) == review.payload
        message = _console_text(capsys)
        assert "PR Review Post Failed" in message
        assert "payload preserved at" in message
        # Rich may wrap this machine path; removing whitespace recovers the displayed value.
        assert str(payload_path) in message.replace(" ", "")
        if file_posts:
            assert "1 file-level comment(s) were already posted." in message
            assert "No comments were posted." not in message
        else:
            assert "No comments were posted." in message
        assert secret not in message
    finally:
        payload_path.unlink(missing_ok=True)
