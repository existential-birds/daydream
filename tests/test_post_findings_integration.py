"""Exercise post-findings through cli.main with real Git and JSON processing.

Only the gh subprocess boundary is faked. Assertions inspect exit codes,
posted review payloads, and GraphQL mutations."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from rich.console import Console

from daydream import git_ops
from daydream.findings import build_findings_artifact, write_findings_artifact
from daydream.pr_review import (
    ParsedIssue,
    PRInfo,
    parse_finding_markers,
    post_findings_from_artifact,
    validate_diagram_payload,
)
from tests.harness.console import collapse_panel_text as _console_text
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, init_repo
from tests.harness.review_result import findings_artifact, terminal_result
from tests.harness.scripts import cli_main
from tests.test_findings import immutable_findings_artifact
from tests.test_review_result import large_attributed_terminal


def _post_argv(
    artifact: Path, *, pr: int = 7, head_sha: str | None = None, target: Path | None = None,
    repo: str = 'o/r', base_sha: str | None = None, comparison_sha: str | None = None,
) -> list[str]:
    """The ``post-findings`` argv for *artifact*; override only what a test varies."""
    return ["post-findings", str(artifact), "--pr", str(pr), "--head-sha", head_sha or "h" * 40, "--repo", repo,
        *(["--target", str(target)] if target is not None else []),
        *(['--base-sha', base_sha] if base_sha is not None else []),
        *(['--comparison-sha', comparison_sha] if comparison_sha is not None else []),
    ]

def test_post_findings_uses_two_explicit_checkout_configs_without_chdir(fake_gh: FakeGh, tmp_path: Path,) -> None:
    ambient = Path.cwd()
    targets = [tmp_path / "checkout A", tmp_path / "checkout B"]
    for target in targets:
        init_repo(target)
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_OLD"], viewer_did_author=True,)
    for index, target in enumerate(targets):
        (target / ".daydream.toml").write_text(f"approve_on_clean = {'true' if index == 0 else 'false'}\n")
        artifact = _write_artifact(target / "findings.json", [_finding(
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
    assert sum("minimizeComment" in call.payload.get("query", "") for call in fake_gh.calls("POST", "graphql")) == 2

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


def test_post_findings_reads_valid_diagram_evidence_from_explicit_target(fake_gh: FakeGh, git_repo: Path,) -> None:
    ambient = Path.cwd()
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head = commit(git_repo, "seed diagram evidence")
    artifact = _write_artifact(git_repo / "findings.json", [], diagrams=_flowchart_payload(), head_sha=head,)
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


def _finding(fingerprint: str, *, path: str, line: int | None, placement: str, title: str, severity: str = "high",
) -> dict[str, Any]:
    return {"fingerprint": fingerprint, "path": path, "line": line, "placement": placement, "title": title,
        "body": "Body text", "severity": severity, "confidence": "HIGH", "is_cross_stack": False,
    }


def _inline_finding(title: str, *, severity: str = "high") -> dict[str, Any]:
    """An inline ``a.py:3`` finding; every default but title/severity is fixed."""
    return _finding("a" * 64, path="a.py", line=3, placement="inline", title=title, severity=severity)


def _write_artifact(path: Path, findings: list[dict[str, Any]], *, run_info: str = "test run info",
    diagrams: dict[str, Any] | None = None, head_sha: str = "h" * 40,
) -> Path:
    """Build a valid artifact via write_findings_artifact."""
    write_findings_artifact(path, findings_artifact(findings, head_sha=head_sha, run_info=run_info, diagrams=diagrams))
    return path




def _flowchart_payload() -> dict[str, Any]:
    return {"results": {"flowchart": {"status": "rendered", "reason": None, "omit_reasons": [],
                "spec_final": {"root": {"file": "a.py", "name": "run", "line": 1},
                    "nodes": [{"id": "start", "kind": "start", "label": "run",
                            "evidence": {"file": "a.py", "line": 1, "symbol": "run"},
                        }, {"id": "end", "kind": "end", "label": "return",
                            "evidence": {"file": "a.py", "line": 2, "symbol": None},
                        },
                    ], "edges": [{"from": "start", "to": "end", "label": None}],
                },
            }
        }
    }


def _sequence_payload() -> dict[str, Any]:
    return {"results": {"sequence": {"status": "rendered", "reason": None, "omit_reasons": [],
                "spec_final": {"participants": [
                        {"name": "api", "kind": "internal", "files": ["api.py"], "service": None},
                        {"name": "worker", "kind": "internal", "files": ["worker.py"], "service": None},
                    ], "messages": [{"from": "api", "to": "worker", "label": "call", "kind": "call", "changed": True,
                            "evidence": {"file": "api.py", "line": 3, "symbol": "worker"},
                        }, {"from": "worker", "to": "api", "label": "reply", "kind": "reply", "changed": True,
                            "evidence": {"file": "worker.py", "line": 2, "symbol": "worker"},
                        }, {"from": "api", "to": "worker", "label": "call again", "kind": "call", "changed": True,
                            "evidence": {"file": "api.py", "line": 5, "symbol": "worker"},
                        },
                    ], "blocks": [{"kind": "opt",
                            "branches": [
                                {"condition": "enabled", "evidence": {"file": "api.py", "line": 4}, "messages": [2]}
                            ],
                        }
                    ],
                },
            }
        }
    }


@pytest.fixture
def artifact_on_disk(tmp_path: Path) -> Path:
    """One inline + one body-only finding (both marker paths exercised)."""
    return _write_artifact(tmp_path / "findings.json",
        [_inline_finding("Inline finding"),
            _finding("b" * 64, path="b.py", line=None, placement="body", title="Body finding"),
        ],
        run_info=("run from commit " + "a" * 40 + "\n- **Reviewed commit:** [`deadbee`]"
                  "(https://github.com/evil/widgets/commit/" + "e" * 40 + ")"),
    )


@pytest.fixture
def artifact_on_disk_second(tmp_path: Path) -> Path:
    """A later run: the prior ``a``-finding is gone, one new finding appears."""
    return _write_artifact(tmp_path / "findings_second.json",
        [_finding("c" * 64, path="c.py", line=5, placement="inline", title="New finding")],
    )



def test_fresh_post_then_idempotent_repost(
    fake_gh: FakeGh, artifact_on_disk: Path, artifact_on_disk_second: Path,
) -> None:
    artifact = json.loads(artifact_on_disk.read_text())
    artifact['terminal_result'] = large_attributed_terminal()
    write_findings_artifact(artifact_on_disk, artifact)
    assert 1024 * 1024 < artifact_on_disk.stat().st_size < 9 * 1024 * 1024
    argv = _post_argv(artifact_on_disk) + ["--approve-on-clean", "--bot-login", "daydream"]
    assert cli_main(argv) == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1 and posts[0].payload["event"] == "COMMENT"
    payload = posts[0].payload
    assert [(c["path"], c["line"], c["side"]) for c in payload["comments"]] == [("a.py", 3, "RIGHT")]
    assert parse_finding_markers(payload["comments"][0]["body"]) == ["a" * 64]
    assert parse_finding_markers(payload["body"]) == ["b" * 64]
    assert payload["body"].startswith("Actionable comments posted: 2\n\n")
    assert fake_gh.process_calls() and all(call.cwd == Path.cwd().resolve() for call in fake_gh.process_calls())
    body = posts[0].payload["body"]
    commit_lines = [line for line in body.splitlines() if line.startswith("- **Reviewed commit:**")]
    assert commit_lines == ["- **Reviewed commit:** [`hhhhhhh`](https://github.com/o/r/commit/" + "h" * 40 + ")"]
    assert "a" * 40 not in commit_lines[0] and "e" * 40 not in body and "evil/widgets" not in body
    # Replay the ACTUAL posted review as prior state: inline comments become
    # GraphQL threads, the review body becomes a REST review. Author is set to
    # the bot login so both harvest paths trust it via the [bot]-tolerant
    # comparator. This proves the wire-format round trip (poster emits ->
    # harvester reads) through the real CLI, on the real posted payload —
    # not just unit-level fabricated markers.
    fake_gh.serve_prior_threads_from(posts[0], author="daydream[bot]")
    assert cli_main(argv) == 0
    assert (len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1
    )  # no dup review
    # A later artifact retires the actual prior inline finding and publishes its replacement.
    assert cli_main(_post_argv(artifact_on_disk_second) + ["--bot-login", "daydream"]) == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 2 and "New finding" in posts[-1].payload["comments"][0]["body"]
    queries = [call.payload.get("query", "") for call in fake_gh.calls("POST", "graphql")]
    assert any("minimizeComment" in query for query in queries)
    assert not any("resolveReviewThread" in query for query in queries)


def test_event_artifact_mismatch_aborts_with_no_side_effects(fake_gh: FakeGh, artifact_on_disk: Path) -> None:
    rc = cli_main(_post_argv(artifact_on_disk, pr=8))  # event says 8
    assert rc == 1
    assert fake_gh.calls("POST") == []  # nothing posted, nothing resolved


@pytest.mark.parametrize('target', ['repo', 'pr', 'head_sha', 'base_sha', 'comparison_sha',
                                   'missing-base', 'missing-comparison'])
def test_immutable_publisher_requires_all_independent_target_facts(
    fake_gh: FakeGh, tmp_path: Path, target: str,
) -> None:
    """A schema-2 PR/head check does not protect immutable requested/comparison bases."""
    path = tmp_path / 'immutable.json'
    write_findings_artifact(path, immutable_findings_artifact([_inline_finding('Confirmed concern')]))
    expected: dict[str, Any] = {'repo': 'o/r', 'pr': 7, 'head_sha': 'a' * 40,
                               'base_sha': 'b' * 40, 'comparison_sha': 'c' * 40}
    if target.startswith('missing-'):
        expected.pop('base_sha' if target == 'missing-base' else 'comparison_sha')
    else:
        expected[target] = 8 if target == 'pr' else 'other/repository' if target == 'repo' else 'd' * 40
    assert cli_main(_post_argv(path, **expected)) == 1
    assert fake_gh.process_calls() == [], 'Rejected provenance must not reach even GitHub inventory reads'


def test_immutable_library_publisher_rejects_boolean_pr_authority(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """CLI integer parsing hides Python's True == 1 alias from the library publisher boundary."""
    path = tmp_path / 'immutable.json'
    artifact = immutable_findings_artifact([_inline_finding('Confirmed concern')])
    artifact['pr_number'] = artifact['immutable_input']['provenance']['pr_number'] = 1
    write_findings_artifact(path, artifact)
    assert post_findings_from_artifact(
        tmp_path, path, pr_number=True, head_sha='a' * 40, repo='o/r', console=Console(),
        base_sha='b' * 40, comparison_sha='c' * 40,
    ) == 1
    assert fake_gh.process_calls() == []


@pytest.mark.parametrize('state', ['complete', 'incomplete'])
def test_immutable_publisher_uses_original_head_and_retains_partial_coverage(
    fake_gh: FakeGh, tmp_path: Path, state: str,
) -> None:
    """Synthetic source references must survive export while posting targets captured original A."""
    path = tmp_path / 'immutable.json'
    artifact = immutable_findings_artifact([_inline_finding('Confirmed concern')], state=state)
    write_findings_artifact(path, artifact)
    assert cli_main(_post_argv(path, head_sha='a' * 40, base_sha='b' * 40, comparison_sha='c' * 40)) == 0
    posts = fake_gh.calls('POST', '/repos/o/r/pulls/7/reviews')
    assert len(posts) == 1 and posts[0].payload['commit_id'] == 'a' * 40
    assert posts[0].payload['event'] == 'COMMENT'
    assert 'Confirmed concern' in posts[0].payload['comments'][0]['body']
    body = posts[0].payload['body']
    assert ('**Review incomplete.**' in body) == (state == 'incomplete')
    if state == 'incomplete':
        assert 'python: backend failure' in body
    assert json.loads(path.read_text()) == artifact
    assert artifact['terminal_result']['stack_outcomes'][0]['target_decisions'][0]['evidence_refs'][0][
        'revision'] == '1' * 40

def test_malformed_artifact_aborts(fake_gh: FakeGh, tmp_path: Path) -> None:
    bad = tmp_path / "f.json"
    bad.write_text("{not json")
    rc = cli_main(_post_argv(bad))
    assert rc == 1 and fake_gh.calls("POST") == []

def test_malformed_repo_config_warns_and_still_posts(fake_gh: FakeGh, tmp_path: Path,) -> None:
    (tmp_path / ".daydream.toml").write_text("this is [not valid toml ==")
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    code = cli_main(_post_argv(artifact, target=tmp_path))
    assert code == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1


def _write_single_finding_artifact(path: Path, fingerprint: str) -> Path:
    return _write_artifact(path / "findings.json",
        [_finding(fingerprint, path="src/app.py", line=10, placement="inline", title="Real finding",)],
    )

def test_marker_suppression_depends_on_prior_thread_author(fake_gh: FakeGh, tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "findings.json", [
        _finding("a" * 64, path="src/app.py", line=10, placement="inline", title="Trusted prior"),
        _finding("b" * 64, path="src/app.py", line=12, placement="inline", title="Forged prior"),
    ])
    fake_gh.serve_prior_threads(fingerprints=["a" * 64, "b" * 64], thread_ids=["RT_BOT", "RT_HUMAN"],
                               authors=["daydream[bot]", "evil-attacker"])
    assert cli_main(_post_argv(artifact) + ["--bot-login", "daydream"]) == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    comments = posts[0].payload["comments"]
    assert [(comment["path"], comment["line"], comment["side"]) for comment in comments] == [
        ("src/app.py", 12, "RIGHT")]
    assert parse_finding_markers(comments[0]["body"]) == ["b" * 64]
    assert "Forged prior" in comments[0]["body"] and "Trusted prior" not in json.dumps(posts[0].payload)

def test_bot_login_env_fallback(monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, tmp_path: Path) -> None:
    artifact = _write_single_finding_artifact(tmp_path, "a" * 64)
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_X"], authors=["daydream[bot]"])
    monkeypatch.setenv("DAYDREAM_BOT_HANDLE", "daydream")  # no --bot-login flag
    code = cli_main(_post_argv(artifact))  # env supplies the login
    assert code == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 0


@pytest.mark.parametrize("severity", ["low", None, "omitted"])
def test_post_findings_approve_when_all_matched_and_clean_flag(
    fake_gh: FakeGh, tmp_path: Path, git_repo: Path, severity: str | None,
) -> None:
    """Schema 2 serializes missing classification as required null, retaining approval eligibility."""
    finding = _inline_finding("Nit", severity="low")
    finding.pop("placement")
    if severity == "omitted":
        finding.pop("severity")
    else:
        finding["severity"] = severity
    base = git(git_repo, "rev-parse", "HEAD")
    (git_repo / "a.py").write_text("FIRST = 1\nSECOND = 2\nTHIRD = 3\n")
    git(git_repo, "add", "a.py")
    head = commit(git_repo, "add reviewed source")
    data = build_findings_artifact(
        git_repo, PRInfo(7, head, base, "main", "feature", "o", "r", ""), [ParsedIssue(**finding)],
        run_info=None, terminal_result=terminal_result(head_sha=head), snapshot_diff=git_ops.diff(git_repo, base, head),
    )
    assert data["findings"][0]["severity"] == (None if severity == "omitted" else severity)
    artifact = tmp_path / "f.json"
    write_findings_artifact(artifact, data)
    argv = _post_argv(artifact, head_sha=head, target=git_repo) + ["--bot-login", "daydream"]
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True)
    assert cli_main(argv) == 0
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []
    code = cli_main(argv + ["--approve-on-clean"])
    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "APPROVE"
    assert "Actionable comments posted: 0" in posts[0].payload["body"]


def test_post_findings_drops_flowchart_nodes_absent_from_source(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head_sha = commit(git_repo, "add flowchart source")
    payload = _flowchart_payload()
    flowchart = payload["results"]["flowchart"]
    flowchart["spec_final"]["nodes"][0]["kind"] = "decision"
    artifact = _write_artifact(
        git_repo / "f.json", [_inline_finding("Already posted")], diagrams=payload, head_sha=head_sha,
    )
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True)

    code = cli_main(_post_argv(artifact, head_sha=head_sha, target=git_repo) + ["--bot-login", "daydream"])

    assert code == 0
    assert "Diagram dropped (the findings are still posted)" in _console_text(capsys)
    # No review is posted because the artifact's one finding is already on the
    # PR, so the run reaches the no-new-findings short-circuit -- NOT because a
    # forged diagram fails the run closed, which it no longer does (#1176).
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []

def test_post_findings_drops_sequence_calls_absent_from_source(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (git_repo / "api.py").write_text(
        "from worker import worker\n"
        "def api():\n"
        "    worker()\n"
        "    if enabled:\n"
        "        worker()\n"
    )
    (git_repo / "worker.py").write_text("def worker():\n" "    return 1\n")
    git(git_repo, "add", "api.py", "worker.py")
    head_sha = commit(git_repo, "add sequence source")
    payload = _sequence_payload()
    assert validate_diagram_payload(payload, target_dir=git_repo, head_sha=head_sha) is None
    payload["results"]["sequence"]["spec_final"]["messages"][0]["evidence"] = {
        "file": "api.py", "line": 2, "symbol": "api",
    }
    artifact = _write_artifact(git_repo / "findings.json", [], diagrams=payload, head_sha=head_sha)

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
    artifact = _write_artifact(git_repo / "findings.json", [], diagrams=payload, head_sha=head_sha,)

    code = cli_main(_post_argv(artifact, head_sha=head_sha, target=git_repo) + ["--bot-login", "daydream"])

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
    """An explicit missing-path Git response proves absence, rather than unreadability."""
    (git_repo / "b.py").write_text("def other():\n    return 1\n")
    git(git_repo, "add", "b.py")
    head_sha = commit(git_repo, "head without the cited file")
    artifact = _write_artifact(git_repo / "findings.json", [], diagrams=_flowchart_payload(), head_sha=head_sha,)

    code = cli_main(_post_argv(artifact, head_sha=head_sha, target=git_repo))

    assert code == 0
    printed = _console_text(capsys)
    assert "flowchart diagram evidence is missing from immutable head: a.py" in printed
    assert "could not be read from immutable head" not in printed

def test_post_findings_reports_a_damaged_local_object_as_unreadable_not_missing(
    fake_gh: FakeGh, git_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing local blob is a reader failure, not evidence of a forged citation."""
    (git_repo / "a.py").write_text("def run():\n    return 1\n")
    git(git_repo, "add", "a.py")
    head_sha = commit(git_repo, "add flowchart source")
    blob = git(git_repo, "rev-parse", f"{head_sha}:a.py").strip()
    (git_repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    artifact = _write_artifact(git_repo / "findings.json", [], diagrams=_flowchart_payload(), head_sha=head_sha,)

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


def _api_head_artifact(
    tmp_path: Path, findings: list[dict[str, Any]] | None = None, *, diagrams: dict[str, Any] | None = None,
) -> Path:
    """An artifact pinned to the immutable API head, flowchart diagram by default."""
    return _write_artifact(tmp_path / "findings.json", findings if findings is not None else [],
        diagrams=_flowchart_payload() if diagrams is None else diagrams, head_sha=_API_HEAD_SHA,
    )


def _serve_contents(fake_gh: FakeGh, path: str, source: str) -> None:
    """Serve *source* as the contents-API body for *path* (the ``?ref=`` is asserted separately)."""
    fake_gh.set_response("GET", f"repos/o/r/contents/{quote(path)}",
        value={"type": "file", "path": path, "sha": "a" * 40, "encoding": "base64",
            "content": base64.b64encode(source.encode()).decode(),
        },
    )


def _contents_calls(fake_gh: FakeGh) -> list[str]:
    """Every contents-API endpoint the run requested, in order."""
    return [call.endpoint for call in fake_gh.calls("GET") if call.endpoint.startswith("repos/o/r/contents/")]

def test_post_findings_posts_flowchart_evidence_read_without_a_checkout(fake_gh: FakeGh, tmp_path: Path,) -> None:
    _serve_contents(fake_gh, "src/deep/a b.py", "def run():\n    return 1\n")
    diagrams = _flowchart_payload()
    spec = diagrams["results"]["flowchart"]["spec_final"]
    spec["root"]["file"] = "src/deep/a b.py"
    spec["nodes"][0]["evidence"]["file"] = "src/deep/a b.py"
    spec["nodes"][1]["evidence"]["file"] = "src/deep/a b.py"
    artifact = _api_head_artifact(tmp_path, diagrams=diagrams)

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert _contents_calls(fake_gh) == [f"repos/o/r/contents/src/deep/a%20b.py?ref={_API_HEAD_SHA}"]
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "```mermaid" in posts[0].payload["body"]

def test_post_findings_posts_sequence_evidence_read_without_a_checkout(fake_gh: FakeGh, tmp_path: Path,) -> None:
    _serve_contents(fake_gh, "api.py",
        "from worker import worker\ndef api():\n    worker()\n    if enabled:\n        worker()\n",
    )
    _serve_contents(fake_gh, "worker.py", "def worker():\n    return 1\n")
    artifact = _api_head_artifact(tmp_path, diagrams=_sequence_payload())

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert sorted(_contents_calls(fake_gh)) == [
        f"repos/o/r/contents/api.py?ref={_API_HEAD_SHA}", f"repos/o/r/contents/worker.py?ref={_API_HEAD_SHA}",
    ]
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "sequenceDiagram" in posts[0].payload["body"]

def test_post_findings_reads_oversized_evidence_through_the_blob_endpoint(fake_gh: FakeGh, tmp_path: Path,) -> None:
    """Past the contents endpoint's inline ceiling GitHub answers ``encoding: none``."""
    blob_sha = "b" * 40
    fake_gh.set_response("GET", "repos/o/r/contents/a.py",
        value={"type": "file", "path": "a.py", "sha": blob_sha, "encoding": "none", "content": ""},
    )
    fake_gh.set_response("GET", f"repos/o/r/git/blobs/{blob_sha}",
        value={"sha": blob_sha, "encoding": "base64",
            "content": base64.b64encode(b"def run():\n    return 1\n").decode(),
        },
    )
    artifact = _api_head_artifact(tmp_path)

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert fake_gh.calls("GET", f"repos/o/r/git/blobs/{blob_sha}")
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1

@pytest.mark.parametrize("unreadable", [False, True], ids=["contradiction", "unreadable"])
def test_post_findings_posts_findings_when_invalid_diagram_is_dropped(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str], unreadable: bool,
) -> None:
    if unreadable:
        fake_gh.set_response("GET", "repos/o/r/contents/a.py",
            value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
        )
    else:
        _serve_contents(fake_gh, "a.py", "x = 1\n")  # no `run` definition, one line
    artifact = _api_head_artifact(
        tmp_path, [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Real finding")],
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert "Real finding" in json.dumps(posts[0].payload)
    assert "```mermaid" not in posts[0].payload["body"]
    printed = _console_text(capsys)
    assert "Diagram dropped (the findings are still posted)" in printed
    if unreadable:
        assert "could not be read from immutable head" in printed

@pytest.mark.parametrize(("label", "response", "present_message", "absent_message"),
    [("not-found", {"__error__": "gh: Not Found (HTTP 404)"}, "is missing from immutable head",
            "could not be read from immutable head",
        ), ("directory", {"type": "dir", "path": "a.py"}, "is missing from immutable head",
            "could not be read from immutable head",
        ), ("rate-limited", {"__error__": "gh: HTTP 403: API rate limit exceeded"},
            "could not be read from immutable head", "is missing from immutable head",
        ), ("forbidden", {"__error__": "gh: HTTP 403: Resource not accessible by integration"},
            "could not be read from immutable head", "is missing from immutable head",
        ), ("unauthorized", {"__error__": "gh: HTTP 401: Bad credentials"},
            "could not be read from immutable head", "is missing from immutable head",
        ), ("undecodable", {"__stdout__": "<html>proxy error</html>"},
            "could not be read from immutable head", "is missing from immutable head",
        ),
    ], ids=["not-found", "directory", "rate-limited", "forbidden", "unauthorized", "undecodable"],
)
def test_post_findings_classifies_api_head_reads(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str], label: str, response: dict[str, str],
    present_message: str, absent_message: str,
) -> None:
    """Only positive absence evidence (404 or non-file) means missing.

    Rate limits, unrecognized statuses and undecodable bodies mean unreadable.
    With no findings, all cases reach the ordinary no-new-findings short-circuit
    and post no review."""
    fake_gh.set_response("GET", "repos/o/r/contents/a.py", value=response)
    artifact = _api_head_artifact(tmp_path)

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews") == []
    printed = _console_text(capsys)
    assert present_message in printed, label
    assert absent_message not in printed, label

def test_post_findings_blames_the_artifact_for_a_malformed_citation_path(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A trailing newline can pass schema re.search but must fail path fullmatch.

    No head read occurred; attribute the bad path to the artifact."""
    payload = _flowchart_payload()
    flowchart = payload["results"]["flowchart"]
    flowchart["spec_final"]["root"]["file"] = "a.py\n"
    for node in flowchart["spec_final"]["nodes"]:
        node["evidence"]["file"] = "a.py\n"
    artifact = _api_head_artifact(
        tmp_path, [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Real finding")], diagrams=payload,
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


def test_post_findings_minimizes_stale_threads_on_a_degraded_artifact(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    fake_gh.set_response("GET", "repos/o/r/contents/a.py", value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
    )
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True,)
    artifact = _api_head_artifact(tmp_path)

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path))

    assert code == 0
    assert sum("minimizeComment" in call.payload.get("query", "") for call in fake_gh.calls("POST", "graphql")) == 1
    assert "Diagram dropped (the findings are still posted)" in _console_text(capsys)

def test_post_findings_approves_on_clean_with_a_degraded_artifact(fake_gh: FakeGh, tmp_path: Path,) -> None:
    fake_gh.set_response("GET", "repos/o/r/contents/a.py", value={"__error__": "gh: HTTP 403: API rate limit exceeded"},
    )
    artifact = _api_head_artifact(
        tmp_path, [_finding("a" * 64, path="a.py", line=1, placement="inline", title="Nit", severity="low")],
    )

    code = cli_main(_post_argv(artifact, head_sha=_API_HEAD_SHA, target=tmp_path) + ["--approve-on-clean"])

    assert code == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "APPROVE"
    assert len(posts[0].payload["comments"]) == 1
    assert posts[0].payload["body"].startswith("Actionable comments posted: 1\n\n")
    assert "```mermaid" not in posts[0].payload["body"]

@pytest.mark.parametrize("schema_version", [2, 3])
def test_post_findings_matched_blocking_finding_prevents_approval(
    fake_gh: FakeGh, tmp_path: Path, schema_version: int,
) -> None:
    """Deduplication must retain blockers without reposting the matched findings."""
    retained = _inline_finding("Retained concern")
    if schema_version == 3:
        retained["severity"] = None
    findings = [retained,
        _finding("b" * 64, path="b.py", line=5, placement="inline", title="New nit", severity="low"),
    ]
    artifact = tmp_path / "f.json"
    data = immutable_findings_artifact(findings) if schema_version == 3 else findings_artifact(findings)
    write_findings_artifact(artifact, data)
    argv = _post_argv(artifact, head_sha="a" * 40 if schema_version == 3 else None,
                      base_sha="b" * 40 if schema_version == 3 else None,
                      comparison_sha="c" * 40 if schema_version == 3 else None)
    argv += ["--approve-on-clean", "--bot-login", "daydream"]
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_1"], viewer_did_author=True)
    assert cli_main(argv) == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1
    assert posts[0].payload["event"] == "COMMENT"
    assert [parse_finding_markers(comment["body"]) for comment in posts[0].payload["comments"]] == [["b" * 64]]
    assert "Retained concern" not in json.dumps(posts[0].payload)
    assert "no high/medium findings" not in posts[0].payload["body"]
    fake_gh.serve_prior_threads(fingerprints=["a" * 64, "b" * 64], thread_ids=["RT_1", "RT_2"], viewer_did_author=True)
    assert cli_main(argv) == 0
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")) == 1

@pytest.mark.parametrize("run_info", ["Artifact-owned run details", None])
def test_artifact_post_never_acquires_live_trajectory_details(
    fake_gh: FakeGh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_info: str | None,
) -> None:
    artifact = _write_artifact(
        tmp_path / "findings.json", [_finding("a" * 64, path="a.py", line=3, placement="inline", title="Finding")],
    )
    document = json.loads(artifact.read_text())
    document["run_info"] = run_info
    document["terminal_result"]["run_id"] = 'run```\n<script>'
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
    assert 'Run:\n````text\nrun``` <script>\n````' in body

def test_post_findings_orders_file_writes_and_folds_failed_thread_once(
    fake_gh: FakeGh, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _write_artifact(tmp_path / "findings.json", [
        _finding("a" * 64, path="first.py", line=None, placement="file", title="First thread"),
        _finding("b" * 64, path="second.py", line=None, placement="file", title="Folded thread"),
        _finding("c" * 64, path="context.py", line=None, placement="body", title="Original body"),
        _finding("d" * 64, path="inline.py", line=3, placement="inline", title="Inline finding"),
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
    assert len(payload["comments"]) == 1
    assert payload["body"].startswith("Actionable comments posted: 4\n\n")
    assert '#### `context.py` (1)' in payload["body"] and '<summary>context.py' not in payload["body"]
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
    assert [call.endpoint for call in writes] == ["repos/o/r/pulls/7/comments", "repos/o/r/pulls/7/reviews"]
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

@pytest.mark.parametrize(('state', 'has_finding'), [('warning', False), ('incomplete', True), ('failed', True)])
def test_unfinished_review_comments_without_approval_or_stale_resolution(
    fake_gh: FakeGh, tmp_path: Path, state: str, has_finding: bool,
) -> None:
    fake_gh.serve_prior_threads(fingerprints=["a" * 64], thread_ids=["RT_OLD"], viewer_did_author=True)
    findings = [_finding("b" * 64, path="a.py", line=1, placement="inline", title="Survivor", severity="low")]
    path = tmp_path / "unfinished.json"
    artifact = findings_artifact(findings if has_finding else [],
        terminal_result=terminal_result("complete" if state == "warning" else state),
        review_warnings=["Alternatives: wall_budget_exceeded"] if state == "warning" else [],
        review_context={'scopes': [], 'scopes_omitted': 0,
            'verification': {'status': 'unproduced', 'value': None},
            'author_reported': {'source': 'pull_request_body', 'head_sha': None,
                              'body': 'Author reports CI passes.', 'truncated': False}})
    write_findings_artifact(path, artifact)
    assert cli_main(_post_argv(path) + ["--approve-on-clean"]) == 0
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(posts) == 1 and posts[0].payload["event"] == "COMMENT"
    body = posts[0].payload['body']
    assert 'Run:\n```text\nrun-1\n```' in body
    assert 'Review outcome' not in body and 'Author reports CI passes.' not in body
    assert body.index('⚠️ **Review') < body.index('Reviewed commit')
    if not has_finding:
        assert 'Additional findings' not in body and 'Fix with an agent' not in body
    if state == 'warning':
        assert 'Alternatives: wall_budget_exceeded' in body and 'Review incomplete' in body
    else:
        assert f'Review {state}' in body and 'python: backend failure' in body
    assert len(posts[0].payload["comments"]) == int(has_finding)
    assert not any("minimizeComment" in c.payload.get("query", "") for c in fake_gh.calls("POST", "graphql"))


def test_unsupported_version_is_rejected_before_github_writes(fake_gh: FakeGh, tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "unknown.json", [_inline_finding("Legacy nit", severity="low")])
    data = json.loads(path.read_text())
    data["schema_version"] = 99
    path.write_text(json.dumps(data))
    assert cli_main(_post_argv(path) + ["--approve-on-clean"]) == 1
    assert fake_gh.calls("POST") == []
