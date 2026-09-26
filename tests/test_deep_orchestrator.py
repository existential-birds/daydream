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
from daydream.deep.artifacts import diff_key, diff_key_path
from daydream.eval.analyzer import _records_issues
from daydream.phases import TestAndHealResult, TestAttemptEvidence
from daydream.pr_review import PRInfo
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_RULE, PR_DESCRIPTION_UNTRUSTED_FRAMING
from daydream.review_profile import ResolvedProfile, build_default_profile, parse_profile
from daydream.runner import RunConfig, run
from daydream.workspace import _resolve_base
from tests.harness.git_helpers import commit as _commit, git as _git, init_repo as _init_repo
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


def _add_bare_remote(repo: Path) -> Path:
    """Give *repo* a real, pushable ``origin``: a sibling bare clone.

    Host-native commit/push (issue #726) really runs ``git push`` and verifies
    the remote holds the pushed HEAD, so a flow test that leaves the commit
    step unmuted needs an actual remote to succeed against.
    """
    bare = repo.parent / (repo.name + "-remote.git")
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", str(bare)],
        check=True,
        capture_output=True,
    )
    return bare


def _install_model_capturing_stubs(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    parse_severity: str | None = None,
    merge_echo_records: bool = False,
    arbiter_omit_verdicts: bool = False,
    parse_by_stack: dict[str, dict[str, Any]] | None = None,
    suppression_keep: bool = True,
) -> list[dict[str, Any]]:
    """Patch create_backend with a per-(name, model) stub factory (#168).

    Each phase resolves its own model, so the orchestrator's (name, model)
    backend cache produces a distinct stub instance per model. Every instance
    shares one model-tagged call list, letting a test assert which model ran
    each phase — the observable proof that the per-stack fan-out runs on Sonnet,
    the merge on Opus, and the arbiter on Opus exactly when it should.

    Returns the shared, model-tagged call list (one dict per execute()).
    """
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
    return ResolvedProfile(
        profile=replace(build_default_profile(), pipeline=parsed.pipeline),
        source_kind="test",
    )


async def _run_deep(
    target: Path,
    *,
    start_at: str = "review",
    precision_mode: bool = False,
    approve_on_clean: bool = False,
    review_profile: "ResolvedProfile | None" = None,
) -> int:
    # cleanup=False suppresses the interactive cleanup prompt; deep is the default.
    config = RunConfig(
        target=str(target),
        start_at=start_at,
        cleanup=False,
        precision_mode=precision_mode,
        approve_on_clean=approve_on_clean,
        review_profile=review_profile,
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
    return {
        "id": item_id,
        "lens": "per-stack",
        "file": file,
        "line": 1,
        "severity": severity,
        "description": desc if desc is not None else f"{severity} issue in {file}",
        "confidence": "MEDIUM",
        "rationale": "rationale",
        "evidence": f"{file}:1",
    }


def _add_to_reviewed_diff(target: Path, files: list[str]) -> None:
    """Commit *files* to the branch for tests that need reviewed-diff fixtures."""
    for name in files:
        (target / name).touch()
    _git(target, "add", *files)
    _commit(target, f"test: add {' '.join(files)} to the reviewed diff")


def _migration_project(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Build a feature-branch fixture with one historical migration."""
    project = tmp_path / name
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'world'\n")
    (project / "App.tsx").write_text("export const App = () => <div>hello</div>;\n")
    (project / "README.md").write_text("# Project\n")
    (project / "migrations").mkdir()
    migration = project / "migrations" / "0001_init.sql"
    migration.write_text("SELECT 1;\n")
    _init_repo(project)
    _git(project, "add", "api.py", "App.tsx", "README.md", "migrations/0001_init.sql")
    _commit(project, "test: initialize migration fixture")
    _git(project, "checkout", "-b", "feature")
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    (project / "App.tsx").write_text("export const App = () => <div>universe</div>;\n")
    (project / "README.md").write_text("# Project\n\nUpdated.\n")
    migration.write_text("SELECT 1;\nSELECT 2;\n")
    _git(project, "add", "api.py", "App.tsx", "README.md", "migrations/0001_init.sql")
    _commit(project, "test: prepare migration change")
    return project, migration


def _go_quote_project(tmp_path: Path) -> Path:
    """Build a feature-branch fixture whose reviewed diff is a single Go file."""
    project = tmp_path / "go_quote_repo"
    project.mkdir()
    (project / "main.go").write_text("package main\n\n// doc\n")
    (project / "notes.md").write_text("# Notes\n")
    _init_repo(project)
    _git(project, "add", "main.go", "notes.md")
    _commit(project, "test: initialize go fixture")
    _git(project, "checkout", "-b", "feature")
    # Only main.go changes in the feature-branch commit: it is the sole
    # reviewed-diff file, so a fix to it is the only edit the run commits.
    (project / "main.go").write_text("package main\n\n// doc updated\n")
    _git(project, "add", "main.go")
    _commit(project, "test: change the go source")
    return project


def _record(**overrides: Any) -> dict[str, Any]:
    """Build one on-disk per-stack record (the shape a merge resume reads back)."""
    record: dict[str, Any] = {"id": 1, "description": "issue", "file": "api.py", "line": 1}
    record.update(overrides)
    return record


def _write_matching_diff_key(target: Path, deep: Path) -> None:
    """Write ``diff-key`` for *target*'s current diff into *deep*.

    These primed artifacts stand in for a prior run over the same diff, so the
    key must match what ``run_deep``'s preamble computes.
    """
    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base)
    diff_key_path(deep).write_text(diff_key(diff or ""), encoding="utf-8")


def _record_issues(loaded: Any) -> list[dict[str, Any]]:
    """Normalize a per-stack records file to its bare issues list.

    Delegates to ``analyzer._records_issues`` (the canonical records-shape
    normalization); non-list shapes (malformed fixtures) degrade to ``[]``.
    """
    issues = _records_issues(loaded)
    return issues if issues is not None else []


def _prime_merge_resume(
    target: Path,
    *,
    python: list[dict[str, Any]] | None = None,
    react: list[dict[str, Any]] | None = None,
    generic: list[dict[str, Any]] | None = None,
    structure: list[dict[str, Any]] | None = None,
) -> Path:
    """Prime the deep artifacts a ``--start-at`` resume reads, returning the deep dir.

    ``intent.md`` + ``alternatives.json`` are the TTT artifacts every resume gate
    requires. Any stack passed a record list also gets its
    ``stack-<name>-records.json``; a stack left as ``None`` is deliberately
    absent (the shape that drives the missing-records guard).
    """
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    # Mirror a fresh run: the key marks the beginning of artifact production, so
    # every primed prerequisite must be newer than it.
    _write_matching_diff_key(target, deep)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    for stack, records in (
        ("python", python),
        ("react", react),
        ("generic", generic),
        ("structure", structure),
    ):
        if records is not None:
            (deep / f"stack-{stack}-records.json").write_text(json.dumps(records))
    return deep


async def _ok(*_a: Any, **kwargs: Any) -> Any:
    """Async stand-in for phase_test_and_heal that always passes.

    Kept for the sibling modules that import it (tests/test_archive_data_capture.py);
    tests in this module use the ``mute_side_effects`` fixture instead.
    """
    key = kwargs["capture_tree_key"]()
    return TestAndHealResult(
        passed=True,
        retries=0,
        proceed=True,
        ignored=False,
        attempts=(
            TestAttemptEvidence(
                session_id=kwargs["session_id"],
                kind="agent",
                command=None,
                passed=True,
                input_tree_key=key,
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
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    pr = PRInfo(
        number=7,
        head_sha=head,
        base_sha=base,
        base_ref="main",
        head_ref="feature",
        owner="o",
        repo="r",
        url="https://example.invalid/pr/7",
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


def _build_gate_target(tmp_path: Path, name: str) -> Path:
    """Build a python-only fixture repo (the shape the quality gate measures)."""
    project = tmp_path / name
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    _init_repo(project)
    _git(project, "add", "api.py")
    _commit(project, "init")
    _git(project, "checkout", "-b", "feature")
    (project / "api.py").write_text("def hello():\n    return 'galaxy'\n")
    _git(project, "add", "api.py")
    _commit(project, "change")
    return project


def _build_gate_target_no_functions(tmp_path: Path, name: str) -> Path:
    """A python-only fixture repo whose api.py has NO functions (erosion None pre-fix)."""
    project = tmp_path / name
    project.mkdir()
    (project / "api.py").write_text(
        "import os\n"
        "import sys\n"
        "\n"
        "NAME = 'gate_five'\n"
        "VERSION = '1.0.0'\n"
        "\n"
        "CONFIG = os.environ.get('APP_CONFIG', 'default')\n"
    )
    _init_repo(project)
    _git(project, "add", "api.py")
    _commit(project, "init")
    _git(project, "checkout", "-b", "feature")
    (project / "api.py").write_text(
        "import os\n"
        "import sys\n"
        "\n"
        "NAME = 'gate_five'\n"
        "VERSION = '1.0.1'\n"
        "\n"
        "CONFIG = os.environ.get('APP_CONFIG', 'default')\n"
    )
    _git(project, "add", "api.py")
    _commit(project, "change")
    return project


def _build_gate_target_with_helper(tmp_path: Path, name: str) -> Path:
    """``_build_gate_target`` plus a second tracked python file (helper.py).

    The quality gate measures python files; this fixture gives the fix agent a
    SECOND parseable python file to touch outside its finding group (#329 /
    Finding 6).
    """
    project = _build_gate_target(tmp_path, name)
    (project / "helper.py").write_text("def helper():\n    return 'util'\n")
    _git(project, "add", "helper.py")
    _commit(project, "add helper")
    return project


def _build_scope_creep_target(tmp_path: Path, name: str) -> Path:
    """A python-only fixture repo whose diff does NOT include a tracked module.

    ``api.py`` changes on the feature branch (so ``changed_files`` names only
    it), while ``unrelated.py`` is a tracked file committed on ``main`` and left
    untouched by the diff — the exact shape issue #336's post-fix residual check
    must protect: a fix agent editing ``unrelated.py`` is editing outside the
    reviewed diff.
    """
    project = tmp_path / name
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    (project / "unrelated.py").write_text("def util():\n    return 'untouched'\n")
    _init_repo(project)
    _git(project, "add", "api.py", "unrelated.py")
    _commit(project, "init")
    _git(project, "checkout", "-b", "feature")
    (project / "api.py").write_text("def hello():\n    return 'galaxy'\n")
    _git(project, "add", "api.py")
    _commit(project, "change")
    return project


class _PromptHookStub(_StubBackend):
    """``_StubBackend`` whose ``intercept`` hook may answer one prompt itself.

    Returning ``None`` (the default) falls through to the base stub's stream;
    returning a sequence of events replaces that turn's stream entirely.
    """

    def intercept(self, cwd: Path, prompt: str) -> Sequence[AgentEvent] | None:
        return None

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
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

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event
        if prompt.lower().startswith(("fix this issue", "fix these")):
            before = self._extra.read_text() if self._append else ""
            self._extra.write_text(before + self._text)


def _read_quality_gate(target: Path) -> dict[str, Any]:
    gate_p = target / ".daydream" / "deep" / "fix-quality-gate.json"
    assert gate_p.is_file(), "fix-quality-gate.json must be written"
    return cast(dict[str, Any], json.loads(gate_p.read_text(encoding="utf-8")))


async def _run_quality_gate_fixture(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    *,
    fix_edit_line: str | None = _FIX_EDIT_VERBOSE,
    file_config: DaydreamFileConfig | None = None,
) -> int:
    """Drive a deep run to the fix phase over *target*, editing api.py verbosely.

    The stub merge emits one high-severity api.py item; the fix agent appends
    ``fix_edit_line`` to the tracked file. Returns the run exit code.
    """
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_edit_line = fix_edit_line
    return await run(
        make_config(
            target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=True,
            file_config=file_config,
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
    """Classify captured prompts for the finding-producing builders (#279).

    Keys are ``per-stack``, ``generic-fallback``, ``structural``, ``arbiter``,
    and ``merge``; each is identified by its stable opening phrase.
    """
    by_kind: dict[str, list[str]] = {
        "per-stack": [],
        "generic-fallback": [],
        "structural": [],
        "arbiter": [],
        "merge": [],
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
    gated_constants = {
        "AUTHORITATIVE_INTENT_RULE": AUTHORITATIVE_INTENT_RULE,
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
    """Stub per-stack overrides staging one structural/language twin on api.py.

    The python stack and the structural meta-stack report the same defect in the
    same words at the same file, diverging only on severity (and, for case B, on
    whether the structural side is anchored whole-file). The other two stacks are
    moved onto their own files so nothing else collides at that location and the
    arbiter's target set is exactly the twin.
    """
    return {
        "python": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "api.py",
            "line": 1,
            "description": _TWIN_DESCRIPTION,
        },
        "structure": {
            "severity": "high",
            "confidence": "HIGH",
            "file": "api.py",
            "line": structural_line,
            "description": _TWIN_DESCRIPTION,
        },
        "react": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "App.tsx",
            "line": 1,
            "description": "Unrelated React concern",
        },
        "generic": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "README.md",
            "line": 1,
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

_SUPPRESSION_COLLISION_STACKS: dict[str, dict[str, Any]] = {
    "python": {
        "severity": "high",
        "confidence": "HIGH",
        "file": "py_module.py",
        "line": 7,
        "description": "the HIGH finding",
        "extra": {
            "severity": "low",
            "confidence": "MEDIUM",
            "file": "py_module.py",
            "line": 7,
            "description": "borderline sibling sharing the HIGH location",
        },
    },
}
