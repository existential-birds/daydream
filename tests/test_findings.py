"""Tests for the findings artifact (build/write/load) in `daydream/findings.py`."""
import json
import re
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.findings import (
    FINDINGS_SCHEMA_VERSION,
    MAX_ARTIFACT_BYTES,
    FindingsValidationError,
    build_findings_artifact,
    load_findings_artifact,
    write_findings_artifact,
)
from daydream.pr_review import ParsedIssue, PRInfo
from daydream.runner import RunConfig, run
from tests.harness.phase_backend import PhaseDispatchBackend


def test_build_artifact_declares_target_envelope(tmp_path: Path) -> None:
    """build_findings_artifact stamps the PR identity envelope and passes each
    issue's fingerprint through. A cross-stack issue routes to body placement
    without touching the diff, so no git/collaborator mocking is needed; inline
    snapping is exercised by the real-path test below.
    """
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main",
                owner="o", repo="r", url="u")
    issues = [ParsedIssue(path="a.py", line=None, title="T", body="B", severity="high",
                          confidence="HIGH", fingerprint="f" * 64, is_cross_stack=True)]
    artifact = build_findings_artifact(tmp_path, pr, issues, run_info=None)
    assert (artifact["repo"], artifact["pr_number"], artifact["head_sha"]) == ("o/r", 7, "h" * 40)
    f = artifact["findings"][0]
    assert (f["fingerprint"], f["placement"], f["line"]) == ("f" * 64, "body", None)


def test_write_artifact_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    write_findings_artifact(path, {"schema_version": FINDINGS_SCHEMA_VERSION, "repo": "o/r",
                                   "pr_number": 7, "head_sha": "h" * 40, "findings": []})
    assert json.loads(path.read_text())["schema_version"] == FINDINGS_SCHEMA_VERSION


# --- Load + validation (confused-deputy gate) ---------------------------------


@pytest.fixture
def valid_artifact() -> dict[str, Any]:
    """Artifact dict with one inline finding."""
    return {
        "schema_version": FINDINGS_SCHEMA_VERSION,
        "repo": "o/r",
        "pr_number": 7,
        "head_sha": "h" * 40,
        "run_info": None,
        "findings": [
            {
                "fingerprint": "f" * 64,
                "path": "a.py",
                "line": 12,
                "placement": "inline",
                "title": "T",
                "body": "B",
                "severity": "high",
                "confidence": "HIGH",
                "is_cross_stack": False,
            }
        ],
    }


@pytest.mark.parametrize("mutate, match", [
    (lambda a: a.pop("head_sha"), "schema"),
    (lambda a: a.update(head_sha="e" * 40), "does not match"),
    (lambda a: a.update(pr_number=8), "does not match"),
    (lambda a: a.update(unexpected=1), "schema"),
    (lambda a: a["findings"][0].update(fingerprint="nope"), "schema"),
])
def test_load_rejects_invalid_artifacts(
    tmp_path: Path,
    valid_artifact: dict[str, Any],
    mutate: Any,
    match: Any,
) -> None:
    mutate(valid_artifact)
    p = tmp_path / "f.json"
    p.write_text(json.dumps(valid_artifact))
    with pytest.raises(FindingsValidationError, match=match):
        load_findings_artifact(p, expected_repo="o/r", expected_pr_number=7,
                               expected_head_sha="h" * 40)


def test_load_rejects_oversized_artifact(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text("[" + " " * MAX_ARTIFACT_BYTES)
    with pytest.raises(FindingsValidationError, match="size"):
        load_findings_artifact(p, expected_repo="o/r", expected_pr_number=7,
                               expected_head_sha="h" * 40)


# --- Real-path Phase A emission (--findings-out via runner.run) --------------


@contextmanager
def _review_run_env(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    out: Any,
    backend: Any,
    pr: Any,
) -> Iterator[Any]:
    """Shared `--review --findings-out` real-path setup: env, config, patch stack.

    Clears the GitHub App env, builds the review-mode RunConfig, and patches the
    backend seam plus the GitHub lookups. Each test supplies only its backend,
    PRInfo, and assertions.
    """
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    config = RunConfig(target=str(feature_branch_repo), output_mode="review",
                       pr_number=7, findings_out=str(out), non_interactive=True)
    with patch("daydream.runner.create_backend", return_value=backend), \
         patch("daydream.github_app.resolve_user_identity", return_value="tester"), \
         patch("daydream.pr_review.find_pr_by_number", return_value=pr):
        yield config


async def test_review_mode_writes_findings_artifact(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`--review --findings-out` writes a fingerprinted artifact pinned to the PR.

    Enters from ``runner.run`` with a real temp git repo and a scripted backend
    injected through the ``create_backend`` seam (the existing phase-dispatch
    harness in events mode). Only the backend and the GitHub lookups
    (``find_pr_by_number`` / identity) are mocked; classification, fingerprints,
    and the artifact write all run for real.
    """
    out = tmp_path / "findings.json"
    issue = {
        "id": 1,
        "title": "Greeting changed without tests",
        "description": "`hello` now returns a different greeting with no test coverage",
        "recommendation": "Add a regression test for the new greeting",
        "severity": "medium",
        "confidence": "HIGH",
        "files": ["main.py"],
        "file": "main.py",
        "line": 1,
        "rationale": "",
        "evidence": "main.py:1",
    }
    # Scripted backend: every phase replays the same event stream; the
    # alternative-review phase consumes the structured issues, the intent and
    # plan phases tolerate the same payload (intent stringifies it; the plan
    # phase renders an empty change list from it).
    backend = PhaseDispatchBackend(events=[
        TextEvent(text="Review complete."),
        # Issue #742: the deep per-stack parse schema requires a ``verdicts``
        # property (Codex strict-mode output).
        ResultEvent(
            structured_output={"issues": [issue], "verdicts": []},
            continuation=None,
        ),
    ])
    head = git_ops.head_sha(feature_branch_repo)
    base = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", "rev-parse", "main"],  # noqa: S607 - git is a trusted command
        cwd=feature_branch_repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    pr = PRInfo(number=7, head_sha=head, base_sha=base, base_ref="main",
                owner="o", repo="r", url="https://example.invalid/pr/7")

    with _review_run_env(feature_branch_repo, monkeypatch, out, backend, pr) as config:
        assert await run(config) == 0

    data = json.loads(out.read_text())
    assert data["pr_number"] == 7
    assert data["head_sha"] == git_ops.head_sha(feature_branch_repo)
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["fingerprint"]) for f in data["findings"])
    assert data["findings"], "scripted issue must survive to the artifact"


async def test_review_mode_errored_agent_never_writes_clean_artifact(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A backend error must abort the review run, not produce an empty artifact.

    Regression guard for the sandbox acceptance failure: with an invalid
    ANTHROPIC_API_KEY the agent errored on every invocation, yet the run
    exited 0, printed "no issues found", and uploaded an empty findings
    artifact that Phase B happily validated. Enters from ``runner.run`` with
    a real temp git repo; only the backend (raising ``ClaudeAgentError`` the
    way the fixed ClaudeBackend does on ``ResultMessage.is_error``) and the
    GitHub lookups are mocked.
    """
    from daydream.backends.claude import ClaudeAgentError

    out = tmp_path / "findings.json"

    class ErroringBackend:
        model = None

        async def execute(
            self,
            cwd: Any,
            prompt: Any,
            output_schema: Any=None,
            continuation: Any=None,
            agents: Any=None,
            max_turns: Any=None,
            read_only: Any=False,
        ) -> AsyncIterator[AgentEvent]:
            yield TextEvent(text="Invalid API key · Fix external API key")
            raise ClaudeAgentError(
                "Claude agent run failed: Invalid API key · Fix external API key"
            )

        async def cancel(self) -> None:
            pass

    head = git_ops.head_sha(feature_branch_repo)
    pr = PRInfo(number=7, head_sha=head, base_sha=head, base_ref="main",
                owner="o", repo="r", url="https://example.invalid/pr/7")

    with _review_run_env(feature_branch_repo, monkeypatch, out, ErroringBackend(), pr) as config:
        with pytest.raises(ClaudeAgentError, match="Invalid API key"):
            await run(config)

    assert not out.exists(), "an errored run must never write a findings artifact"


# --- Grounded diagrams (issue #1113) ----------------------------------------


def _diagram_payload() -> dict[str, Any]:
    """A minimal but shape-real diagram payload."""
    return {
        "eligibility": {"sequence": {"eligible": True, "rule": "forced", "reason": "r"}},
        "results": {
            "sequence": {
                "status": "rendered",
                "reason": None,
                "spec_proposed": {"participants": [], "messages": [], "blocks": []},
                "spec_final": {"participants": [], "messages": [], "blocks": []},
                "grounding": {
                    "elements": [],
                    "summary": {
                        "proposed": 0,
                        "grounded_first_pass": 0,
                        "repaired": 0,
                        "pruned": 0,
                    },
                    "capped": {},
                    "root_range": None,
                },
                "omit_reasons": [],
            },
            "flowchart": None,
        },
    }


def test_diagram_artifact_round_trips_kind_and_payload(tmp_path: Path) -> None:
    """``kind``/``diagrams`` survive build -> write -> validate -> load."""
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main",
                owner="o", repo="r", url="u")
    payload = _diagram_payload()
    artifact = build_findings_artifact(
        tmp_path, pr, [], run_info=None, kind="diagram", diagrams=payload
    )
    assert artifact["kind"] == "diagram"
    assert artifact["diagrams"] == payload

    path = tmp_path / "diagram-findings.json"
    write_findings_artifact(path, artifact)
    loaded = load_findings_artifact(
        path, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40
    )
    assert loaded.kind == "diagram"
    assert loaded.diagrams == payload
    assert loaded.findings == []


def test_review_artifact_defaults_kind_and_diagrams(tmp_path: Path) -> None:
    """A pre-#1113 artifact with neither key loads as a review with no diagrams."""
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": FINDINGS_SCHEMA_VERSION,
                "repo": "o/r",
                "pr_number": 7,
                "head_sha": "h" * 40,
                "findings": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_findings_artifact(
        path, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40
    )
    assert loaded.kind == "review"
    assert loaded.diagrams is None


def test_unknown_artifact_kind_is_rejected_by_the_schema(tmp_path: Path) -> None:
    """``kind`` is an enum: an unknown value fails validation before any use."""
    path = tmp_path / "bad-kind.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": FINDINGS_SCHEMA_VERSION,
                "repo": "o/r",
                "pr_number": 7,
                "head_sha": "h" * 40,
                "kind": "sabotage",
                "findings": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FindingsValidationError, match="schema validation"):
        load_findings_artifact(
            path, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40
        )


def test_write_rejects_an_oversized_artifact(tmp_path: Path) -> None:
    """The size cap fails in the job that produced the artifact, not one job later."""
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main",
                owner="o", repo="r", url="u")
    payload = _diagram_payload()
    payload["results"]["sequence"]["spec_final"]["messages"] = [
        {"label": "x" * 512} for _ in range(4000)
    ]
    artifact = build_findings_artifact(
        tmp_path, pr, [], run_info=None, kind="diagram", diagrams=payload
    )
    path = tmp_path / "huge.json"
    with pytest.raises(FindingsValidationError, match="size check failed"):
        write_findings_artifact(path, artifact)
    assert not path.exists(), "an over-cap artifact must not be left on disk"
