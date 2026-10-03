"""Compatibility helpers for deep-orchestrator tests and sibling modules."""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator, Callable, Sequence
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
from daydream.deep.records import record_issues
from daydream.phases import TestAttemptEvidence
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
from tests.harness.stub_backend import PARTIAL_FIX_MARKER, StubBackend, force_interactive, install_stub_backend, silence

if TYPE_CHECKING:
    from daydream.config_file import DaydreamFileConfig

_PARTIAL_FIX_MARKER = PARTIAL_FIX_MARKER

_StubBackend = StubBackend

_silence = silence

_force_interactive = force_interactive

_install_stub_backend = install_stub_backend

MakeConfig = Callable[..., "RunConfig"]

Mute = Callable[..., None]

def _accept_intent_decline_other(_console: Any, message: str, _default: str = "") -> str:
    """Accept intent confirmation while declining later optional gates."""
    return "y" if "understanding correct" in message.lower() else "n"

def _recording_prompter(asked: list[str]) -> Callable[..., str]:
    """Accept intent confirmation, recording then declining later optional gates."""
    def _prompt(_console: Any, message: str, _default: str = "") -> str:
        if "understanding correct" in message.lower():
            return "y"
        asked.append(message)
        return "n"
    return _prompt

def _add_bare_remote(repo: Path) -> Path:
    """Create a sibling bare origin for real push and remote-HEAD verification."""
    bare = repo.parent / (repo.name + "-remote.git")
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(bare)], check=True, capture_output=True,)
    return bare

def _install_model_capturing_stubs(monkeypatch: pytest.MonkeyPatch, target: Path, *, parse_severity: str | None = None,
    merge_echo_records: bool = False, arbiter_omit_verdicts: bool = False,
    parse_by_stack: dict[str, dict[str, Any]] | None = None, suppression_keep: bool = True,
) -> list[dict[str, Any]]:
    """Install cached per-(name, model) stubs sharing one model-tagged call log."""
    shared_calls: list[dict[str, Any]] = []

    def factory(name: str, model: str | None = None, **kwargs: object) -> _StubBackend:
        stub = _StubBackend(target, model=model or "mock-model", shared_calls=shared_calls)
        stub.parse_severity = parse_severity
        stub.merge_echo_records = merge_echo_records
        stub.arbiter_omit_verdicts = arbiter_omit_verdicts
        stub.parse_by_stack = parse_by_stack
        stub.suppression_keep = suppression_keep
        return stub

    monkeypatch.setattr("daydream.runner.create_backend", factory)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    return shared_calls

def _profile_with_pipeline(**overrides: object) -> "ResolvedProfile":
    """Build a test ResolvedProfile with the default strategies + pipeline overrides."""
    pipeline = "\n".join(f"{key} = {json.dumps(value)}" for key, value in overrides.items())
    parsed = parse_profile(f"[pipeline]\n{pipeline}")
    return ResolvedProfile(profile=replace(build_default_profile(), pipeline=parsed.pipeline), source_kind="test",)

async def _run_deep(
    target: Path, *, start_at: str = "review", precision_mode: bool = False, approve_on_clean: bool = False,
    review_profile: "ResolvedProfile | None" = None, review_cache_enabled: bool = True,
) -> int:
    # cleanup=False suppresses the interactive cleanup prompt; deep is the default.
    config = RunConfig(target=str(target), start_at=start_at, cleanup=False, precision_mode=precision_mode,
        approve_on_clean=approve_on_clean, review_profile=review_profile, review_cache_enabled=review_cache_enabled,
    )
    return await run(config)

_SEVERITY_SORT_RANK = {"high": 0, "medium": 1, "low": 2}

def _severity_sort_key(s: str) -> int:
    """Sort rank for canonical severities; errors naming unknown/absent values."""
    try:
        return _SEVERITY_SORT_RANK[s]
    except KeyError:
        raise ValueError(f"unexpected severity in fixture: {s!r}") from None

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
            "App.tsx": "export const App = () => <div>hello</div>;\n",
            "README.md": "# Project\n",
            migration_rel: "SELECT 1;\n",
        },
        changed={
            "api.py": "def hello():\n    return 'universe'\n",
            "App.tsx": "export const App = () => <div>universe</div>;\n",
            "README.md": "# Project\n\nUpdated.\n",
            migration_rel: "SELECT 1;\nSELECT 2;\n",
        },
    )
    return project, project / migration_rel

def _go_quote_project(tmp_path: Path) -> Path:
    """Build a feature-branch fixture whose reviewed diff is a single Go file."""
    # Only main.go changes in the feature-branch commit: it is the sole
    # reviewed-diff file, so a fix to it is the only edit the run commits.
    return _feature_branch_repo(tmp_path, "go_quote_repo",
        initial={"main.go": "package main\n\n// doc\n", "notes.md": "# Notes\n"},
        changed={"main.go": "package main\n\n// doc updated\n"},
    )

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

def _record_issues(loaded: Any) -> list[dict[str, Any]]:
    """Use canonical record normalization; malformed non-list fixtures yield []."""
    issues = record_issues(loaded)
    return issues if issues is not None else []

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
                coverage.record_scope(stack, "complete")
    persist_review_coverage(deep, coverage)
    return deep

async def _ok(_backend: Any, session: Any, **_kwargs: Any) -> bool:
    """Always-passing async test phase shared with archive data-capture tests."""
    key = session.capture_key()
    session.test_attempts.append(TestAttemptEvidence(
        session_id=session.session_id, kind="agent", command=None, passed=True,
        input_tree_key=key, output_tree_key=key,
    ))
    return True

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

_FIX_EDIT_VERBOSE = (
    "\ndef choose(x):\n"
    "    if x > 0:\n"
    "        return 1\n"
    "    else:\n"
    "        return 2\n"
    "    if x > 0:\n"
    "        return 1\n"
    "    else:\n"
    "        return 2\n"
)

_FIX_EDIT_ERODED = (
    "\ndef route(request):\n"
    "    if request == 'a':\n"
    "        return 1\n"
    "    elif request == 'b':\n"
    "        return 2\n"
    "    elif request == 'c':\n"
    "        return 3\n"
    "    elif request == 'd':\n"
    "        return 4\n"
    "    elif request == 'e':\n"
    "        return 5\n"
    "    elif request == 'f':\n"
    "        return 6\n"
    "    elif request == 'g':\n"
    "        return 7\n"
    "    elif request == 'h':\n"
    "        return 8\n"
    "    elif request == 'i':\n"
    "        return 9\n"
    "    elif request == 'j':\n"
    "        return 10\n"
    "    else:\n"
    "        return 0\n"
)

def _feature_branch_repo(tmp_path: Path, name: str, *, initial: dict[str, str], changed: dict[str, str],) -> Path:
    """Commit ``initial`` on the default branch, branch, then commit ``changed``."""
    project = tmp_path / name
    seed_feature_branch(project, base=initial, feature=changed, base_message="init", feature_message="change",)
    return project

def _build_gate_target(tmp_path: Path, name: str) -> Path:
    """Build a python-only fixture repo (the shape the quality gate measures)."""
    return _feature_branch_repo(tmp_path, name,
        initial={"api.py": "def hello():\n    return 'universe'\n"},
        changed={"api.py": "def hello():\n    return 'galaxy'\n"},
    )

def _build_gate_target_no_functions(tmp_path: Path, name: str) -> Path:
    """A python-only fixture repo whose api.py has NO functions (erosion None pre-fix)."""
    initial = (
        "import os\n"
        "import sys\n"
        "\n"
        "NAME = 'gate_five'\n"
        "VERSION = '1.0.0'\n"
        "\n"
        "CONFIG = os.environ.get('APP_CONFIG', 'default')\n"
    )
    return _feature_branch_repo(
        tmp_path, name, initial={"api.py": initial}, changed={"api.py": initial.replace("1.0.0", "1.0.1")},
    )

def _build_gate_target_with_helper(tmp_path: Path, name: str) -> Path:
    """Add tracked helper.py as a parseable quality-gate target outside the fix group."""
    project = _build_gate_target(tmp_path, name)
    (project / "helper.py").write_text("def helper():\n    return 'util'\n")
    _git(project, "add", "helper.py")
    _commit(project, "add helper")
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

class _PromptHookStub(_StubBackend):
    """Let intercept replace a turn's events, or return None to use the base stub."""

    def intercept(self, cwd: Path, prompt: str) -> Sequence[AgentEvent] | None:
        return None

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,) -> AsyncIterator[AgentEvent]:
        own = self.intercept(cwd, prompt)
        if own is not None:
            for event in own:
                yield event
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

def _prompt_ref(prompt: str, label: str) -> str:
    """The value named by the prompt's single ``- <label>: <value>`` pointer line."""
    return next(line.removeprefix(f"- {label}: ") for line in prompt.splitlines() if line.startswith(f"- {label}: "))

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

def _read_quality_gate(target: Path) -> dict[str, Any]:
    gate_p = target / ".daydream" / "deep" / "fix-quality-gate.json"
    assert gate_p.is_file(), "fix-quality-gate.json must be written"
    return cast(dict[str, Any], json.loads(gate_p.read_text(encoding="utf-8")))

async def _run_quality_gate_fixture(
    target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute, *,
    fix_edit_line: str | None = _FIX_EDIT_VERBOSE, file_config: DaydreamFileConfig | None = None,
) -> int:
    """Run one high-severity api.py fix, append fix_edit_line, and return the exit code."""
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_edit_line = fix_edit_line
    return await run(make_config(
            target, assume="yes", output_mode="loop", non_interactive=False, archive=True, file_config=file_config,
        )
    )

INTENT_SENTINEL = "SKIP_IF_NO_QUERY_IS_A_DELIBERATE_GUARD"

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

def _registry_text(plugin_names: list[str]) -> str:
    return '{"version": 2, "plugins": {' + ", ".join(f'"{name}@marketplace": []' for name in plugin_names) + "}}"

def _write_plugin_registry(config_dir: Path, plugin_names: list[str]) -> None:
    registry = config_dir / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(_registry_text(plugin_names))

_TWIN_DESCRIPTION = "The staging cache URL does not match the documented shared instance"

def _twin_parse_by_stack(structural_line: int) -> dict[str, dict[str, Any]]:
    """Create a structural/language twin differing only in severity or whole-file scope.

    Move other findings to distinct files so arbitration selects exactly the twin.
    """
    return {"python": {
            "severity": "medium", "confidence": "MEDIUM", "file": "api.py", "line": 1, "description": _TWIN_DESCRIPTION,
        },
        "structure": {"severity": "high", "confidence": "HIGH", "file": "api.py", "line": structural_line,
            "description": _TWIN_DESCRIPTION,
        },
        "react": {"severity": "medium", "confidence": "MEDIUM", "file": "App.tsx", "line": 1,
            "description": "Unrelated React concern",
        },
        "generic": {"severity": "medium", "confidence": "MEDIUM", "file": "README.md", "line": 1,
            "description": "Unrelated docs concern",
        },
    }

_PRECISION_STACKS: dict[str, dict[str, Any]] = {
    "python": {"severity": "high", "confidence": "HIGH", "file": "api.py", "line": 1},
    "react": {"severity": "low", "confidence": "MEDIUM", "file": "App.tsx", "line": 1},
}

_CONFIDENCE_KNOB_STACKS: dict[str, dict[str, Any]] = {
    "python": {"severity": "high", "confidence": "HIGH", "file": "api.py", "line": 1},
    "react": {"severity": "medium", "confidence": "MEDIUM", "file": "App.tsx", "line": 1},
}

_SUPPRESSION_COLLISION_STACKS: dict[str, dict[str, Any]] = {"python": {
        "severity": "high", "confidence": "HIGH", "file": "py_module.py", "line": 7, "description": "the HIGH finding",
        "extra": {"severity": "low", "confidence": "MEDIUM", "file": "py_module.py", "line": 7,
            "description": "borderline sibling sharing the HIGH location",
        },
    },
}
