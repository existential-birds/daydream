"""Compatibility helpers for deep-orchestrator tests and sibling modules."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent
from daydream.deep.artifacts import (
    DeepArtifact,
    diff_key,
)
from daydream.phases import TestAndHealResult, TestAttemptEvidence
from daydream.pr_review import PRInfo
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_RULE, PR_DESCRIPTION_UNTRUSTED_FRAMING
from daydream.review_profile import ResolvedProfile, build_default_profile, parse_profile
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.workspace import _resolve_base
from tests.harness.git_helpers import (
    commit as _commit,
    git as _git,
    seed_feature_branch,
)
from tests.harness.stub_backend import StubBackend, force_interactive, install_stub_backend, silence

if TYPE_CHECKING:
    pass

_StubBackend = StubBackend

_silence = silence

_force_interactive = force_interactive

_install_stub_backend = install_stub_backend

MakeConfig = Callable[..., "RunConfig"]

Mute = Callable[..., None]

def _add_bare_remote(repo: Path) -> Path:
    """Create a sibling bare origin for real push and remote-HEAD verification."""
    bare = repo.parent / (repo.name + "-remote.git")
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(bare)], check=True, capture_output=True,)
    return bare

def _profile_with_pipeline(**overrides: object) -> "ResolvedProfile":
    """Build a test ResolvedProfile with the default strategies + pipeline overrides."""
    pipeline = "\n".join(f"{key} = {json.dumps(value)}" for key, value in overrides.items())
    parsed = parse_profile(f"[pipeline]\n{pipeline}")
    return ResolvedProfile(profile=replace(build_default_profile(), pipeline=parsed.pipeline), source_kind="test",)

async def _run_deep(
    target: Path, *, start_at: str = "review", precision_mode: bool = False, approve_on_clean: bool = False,
    review_profile: "ResolvedProfile | None" = None, review_cache_enabled: bool = True, test_command: str | None = None,
) -> int:
    # cleanup=False suppresses the interactive cleanup prompt; deep is the default.
    config = RunConfig(target=str(target), start_at=start_at, cleanup=False, precision_mode=precision_mode,
        approve_on_clean=approve_on_clean, review_profile=review_profile, review_cache_enabled=review_cache_enabled,
        test_command=test_command or f"{sys.executable} -c pass",
    )
    return await run(config)

def _merge_item(item_id: int, file: str, severity: str, *, desc: str | None = None) -> dict[str, Any]:
    """Build a validated merged item (shape copied from the stub default)."""
    return {"id": item_id, "lens": "per-stack", "file": file, "line": 1, "severity": severity,
        "description": desc if desc is not None else f"{severity} issue in {file}", "confidence": "MEDIUM",
        "rationale": "rationale", "evidence": f"{file}:1", "related_files": None, "source_uids": None,
    }

def _add_to_reviewed_diff(target: Path, files: list[str]) -> None:
    """Commit *files* to the branch for tests that need reviewed-diff fixtures."""
    for name in files:
        (target / name).touch()
    _git(target, "add", *files)
    _commit(target, f"test: add {' '.join(files)} to the reviewed diff")

def _migration_project(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Build a feature-branch fixture with one historical migration."""
    migration_rel = "migrations/0001_init.sql"
    project = _feature_branch_repo(tmp_path, name,
        initial={
            "api.py": "def hello():\n    return 'world'\n",
            "App.tsx": 'import React from "react"; export const App = () => <div>hello</div>;\n',
            "README.md": "# Project\n",
            migration_rel: "SELECT 1;\n",
        },
        changed={
            "api.py": "def hello():\n    return 'universe'\n",
            "App.tsx": 'import React from "react"; export const App = () => <div>universe</div>;\n',
            "README.md": "# Project\n\nUpdated.\n",
            migration_rel: "SELECT 1;\nSELECT 2;\n",
        },
    )
    return project, project / migration_rel

def _record(**overrides: Any) -> dict[str, Any]:
    """Build one on-disk per-stack record (the shape a merge resume reads back)."""
    record: dict[str, Any] = {"id": 1, "description": "issue", "file": "api.py", "line": 1,
                              "severity": "medium", "confidence": "MEDIUM",
                              "rationale": "fixture defect", "evidence": "api.py:1"}
    record.update(overrides)
    return record

def _write_matching_diff_key(target: Path, deep: Path) -> None:
    """Key primed resume artifacts to the current diff, matching the deep preamble."""
    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base)
    DeepArtifact.DIFF_KEY.at(deep).write_text(diff_key(diff or ""), encoding="utf-8")

def _prime_merge_resume(
    target: Path, *, python: list[dict[str, Any]] | None = None, react: list[dict[str, Any]] | None = None,
    generic: list[dict[str, Any]] | None = None, structure: list[dict[str, Any]] | None = None,
) -> Path:
    """Write required intent/alternative artifacts and optional stack records.

    None deliberately omits a stack file to exercise the missing-record guard.
    Return the deep artifact directory.
    """
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    # Mirror a fresh run: the key marks the beginning of artifact production, so
    # every primed prerequisite must be newer than it.
    _write_matching_diff_key(target, deep)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    from daydream.deep.artifacts import persist_review_coverage
    from daydream.deep.diff import _diff_changed_files
    from daydream.deep.orchestrator import _prepare_review_stacks
    from daydream.deep.records import stamp_record_uids
    from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage
    from tests.harness.review_result import frozen_target_decisions

    config = RunConfig(target=str(target), start_at="merge", cleanup=False)
    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base) or ""
    head = git_ops.head_sha(target)
    merge_base = git_ops.resolve_diff_merge_base(target, base, head)
    stacks, _, _ = _prepare_review_stacks(config, _diff_changed_files(diff), diff, target, "deep")
    coverage = ReviewCoverage("primed-review-fixture", AnalyzedRevision(head, merge_base, diff_key(diff)),
                              [PlannedScope(stack.stack_name, stack.stack_name.split("#", 1)[0],
                                            files=tuple(sorted(stack.files))) for stack in stacks],
                              ("intent", "alternatives", "merge"))
    coverage.record_phase("intent", "complete")
    coverage.record_phase("alternatives", "complete", noop=True)
    for stack, records in (("python", python), ("react", react), ("generic", generic), ("structure", structure)):
        if records is not None:
            stamp_record_uids(records, stack)
            envelope = {"issues": records, "scope_id": stack, "analyzed_revision": coverage.revision.to_dict(),
                        "originating_run_id": coverage.run_id}
            (deep / f"stack-{stack}-records.json").write_text(json.dumps(envelope))
            if stack in coverage.scopes:
                assignment = next(assignment for assignment in stacks if assignment.stack_name == stack)
                coverage.record_scope(stack, "complete", target_decisions=frozen_target_decisions(
                    target, assignment, coverage.revision, diff))
    persist_review_coverage(deep, coverage)
    return deep

async def _ok(*_a: Any, **kwargs: Any) -> Any:
    """Always-passing async test phase shared with archive data-capture tests."""
    key = kwargs["capture_tree_key"]()
    return TestAndHealResult(passed=True, retries=0, proceed=True, ignored=False,
        attempts=(TestAttemptEvidence(
                session_id=kwargs["session_id"], kind="agent", command=None, passed=True, input_tree_key=key,
                output_tree_key=key,
            ),
        ),
    )

async def _noop_commit(*_a: Any, **_k: Any) -> None:
    """Async no-op stand-in for phase_commit_push (see ``_ok`` on why it stays)."""
    return None

def _pin_findings_pr(monkeypatch: pytest.MonkeyPatch, target: Path) -> "PRInfo":
    """Provide the PR metadata required by the findings-out artifact."""
    head = git_ops.head_sha(target)
    base = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", "rev-parse", "main"],  # noqa: S607 - git is a trusted command
        cwd=target, capture_output=True, text=True, check=True,
    ).stdout.strip()
    pr = PRInfo(number=7, head_sha=head, base_sha=base, base_ref="main", head_ref="feature", owner="o", repo="r",
        url="https://example.invalid/pr/7", pr_base_sha=base,
    )
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", lambda target_dir, n, **_kwargs: pr)
    return pr

def _feature_branch_repo(tmp_path: Path, name: str, *, initial: dict[str, str], changed: dict[str, str],) -> Path:
    """Commit ``initial`` on the default branch, branch, then commit ``changed``."""
    project = tmp_path / name
    seed_feature_branch(project, base=initial, feature=changed, base_message="init", feature_message="change",)
    return project

def _build_scope_creep_target(tmp_path: Path, name: str) -> Path:
    """Change only api.py while leaving unrelated.py tracked outside the reviewed diff."""
    return _feature_branch_repo(tmp_path, name,
        initial={
            "api.py": "def hello():\n    return 'universe'\n",
            "unrelated.py": "def util():\n    return 'untouched'\n",
        },
        changed={"api.py": "def hello():\n    return 'galaxy'\n"},
    )

def _sanctioned_inputs(prompt: str) -> dict[str, Path]:
    """The label -> path map the rendered sanctioned-inputs block enumerates."""
    lines = prompt.splitlines()
    start = lines.index("Sanctioned phase inputs (read only these exact files):") + 1
    sanctioned: dict[str, Path] = {}
    for line in lines[start:]:
        if not line.startswith("- "):
            break
        label, path = line.removeprefix("- ").split(": ", 1)
        sanctioned[label] = Path(path)
    return sanctioned

class _ExtraEditBackend(_StubBackend):
    """Make a fix turn also edit a second file, optionally creating it."""

    def __init__(self, target: Path, extra: Path, text: str, *, append: bool = True) -> None:
        super().__init__(target)
        self._extra = extra
        self._text = text
        self._append = append

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,) -> AsyncIterator[AgentEvent]:
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event
        if prompt.lower().startswith(("fix this issue", "fix these")):
            extra = cwd / self._extra.relative_to(self._target)
            before = extra.read_text() if self._append else ""
            extra.write_text(before + self._text)

def _fix_prompts(stub: _StubBackend) -> list[str]:
    # Same-file findings are batched into one "Fix these N issues" turn; a lone
    # finding still uses the single-finding "Fix this issue" prompt. Match both.
    return [c["prompt"] for c in stub.calls if c["prompt"].startswith(("Fix this issue", "Fix these"))]

PR_SENTINEL = "DELIBERATE_RATIO_PASS_THROUGH_IS_INTENTIONAL"

def _intent_calls(stub: _StubBackend) -> list[dict[str, Any]]:
    """Recover the intent-phase calls by their stable instruction text."""
    return [c for c in stub.calls if "understand the intent of these changes" in c["prompt"].lower()]

def _intent_prompt(stub: _StubBackend) -> str:
    """Recover the intent-phase prompt by its stable instruction text."""
    return cast(str, next(c["prompt"] for c in _intent_calls(stub)))

def _review_prompts_by_kind(stub: _StubBackend) -> dict[str, list[str]]:
    """Classify finding-producing prompts by their stable role-opening phrases."""
    by_kind: dict[str, list[str]] = {
        "per-stack": [], "generic-fallback": [], "structural": [], "arbiter": [], "merge": [],
    }
    for c in stub.calls:
        pl = c["prompt"].lower()
        if "you are reviewing the generic-fallback stack" in pl:
            by_kind["generic-fallback"].append(c["prompt"])
        elif "you are reviewing the " in pl:
            by_kind["per-stack"].append(c["prompt"])
        elif "you are the structural reviewer" in pl:
            by_kind["structural"].append(c["prompt"])
        elif "you are the arbiter" in pl:
            by_kind["arbiter"].append(c["prompt"])
        elif "cross-stack merge agent" in pl:
            by_kind["merge"].append(c["prompt"])
    return by_kind

def _assert_authoritative_rule_gated(stub: _StubBackend, *, expect_present: bool) -> None:
    """Assert the precedence rule AND the #579 untrusted framing are present/absent
    in every finding-producing prompt."""
    by_kind = _review_prompts_by_kind(stub)
    missing = [k for k, prompts in by_kind.items() if not prompts]
    assert not missing, f"expected prompts for all five kinds, missing: {missing}"
    gated_constants = {"AUTHORITATIVE_INTENT_RULE": AUTHORITATIVE_INTENT_RULE,
        "PR_DESCRIPTION_UNTRUSTED_FRAMING": PR_DESCRIPTION_UNTRUSTED_FRAMING,
    }
    for kind, prompts in by_kind.items():
        for name, constant in gated_constants.items():
            assert all((constant in p) == expect_present for p in prompts), (
                f"{kind}: expected {name} {'in' if expect_present else 'absent from'} every prompt"
            )

_TWIN_DESCRIPTION = "The staging cache URL does not match the documented shared instance"

