"""Compatibility helpers for deep-orchestrator tests and sibling modules."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent
from daydream.phases import TestAndHealResult, TestAttemptEvidence
from daydream.pr_review import PRInfo
from daydream.review_profile import ResolvedProfile, build_default_profile, parse_profile
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.git_helpers import (
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
    # cleanup=False suppresses the interactive cleanup prompt; callers force a terminal to answer the fix gate.
    config = RunConfig(target=str(target), output_mode="loop", start_at=start_at, cleanup=False,
        precision_mode=precision_mode, approve_on_clean=approve_on_clean, review_profile=review_profile,
        review_cache_enabled=review_cache_enabled, test_command=test_command or f"{sys.executable} -c pass",
    )
    return await run(config)

def _merge_item(item_id: int, file: str, severity: str, *, desc: str | None = None) -> dict[str, Any]:
    """Build a validated merged item (shape copied from the stub default)."""
    return {"id": item_id, "lens": "per-stack", "file": file, "line": 1, "severity": severity,
        "description": desc if desc is not None else f"{severity} issue in {file}", "confidence": "MEDIUM",
        "rationale": "rationale", "evidence": f"{file}:1", "related_files": None, "source_uids": None,
    }


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
