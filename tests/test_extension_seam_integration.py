"""Real-path tests: fork flow mutations reach the registered flow definitions.

Drives the production entrypoints (``runner.run_feedback`` / ``runner.run``)
against a real temp git repo, mocking ONLY the backend seam
(``daydream.runner.create_backend``) per the testing standard. A
``daydream_ext`` package written by the ``ext_dir`` fixture mutates the flow
definitions (remove/insert steps); assertions are on the prompts the backend
actually received and the exit code. Grows across Tasks 9-15 of the
extension-seam plan, one flow migration at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.backends import ResultEvent, TextEvent, ToolStartEvent
from daydream.deep.orchestrator import _step_post_review
from daydream.extensions.registry import Registry
from daydream.flows.engine import FlowContext
from daydream.runner import RunConfig
from daydream.workspace import WorkContext
from tests.conftest import ExtDir
from tests.harness.fake_gh import FakeGh
from tests.harness.phase_backend import PhaseDispatchBackend

KEEP_ME = "KEEP_ME"
DROP_ME = "DROP_ME"


FILTER_ITEMS_EXT = """
import json

from daydream.extensions import FlowStep

DAYDREAM_EXT_API = 3


async def _filter_items(ctx):
    items_file = ctx.data["items_file"]
    payload = json.loads(items_file.read_text())
    payload["items"] = [
        item for item in payload["items"] if item["description"] != "DROP_ME"
    ]
    items_file.write_text(json.dumps(payload))


def register(r):
    r.register_phase(FlowStep(name="filter-items", run=_filter_items))
    r.insert_after("deep", anchor="load-items", step="filter-items")
"""


def _post_context(*, dd: Path, items_file: Path) -> FlowContext:
    """Build the smallest context needed by the post-review step."""
    return FlowContext(
        config=RunConfig(),
        work=WorkContext(
            repo=dd.parent,
            source=dd.parent,
            base_branch="main",
            base_sha="base",
            head_branch="feature",
            head_sha="head",
            is_ephemeral=False,
            run_id="test-run",
        ),
        registry=Registry(),
        data={"dd": dd, "items_file": items_file},
    )


async def _record_path(paths: list[Path], path: Path) -> None:
    paths.append(path)


def _filtered_items() -> list[dict[str, Any]]:
    """Return valid canonical items with distinctive post-filter markers."""
    return [
        {
            "id": 1,
            "lens": "per-stack",
            "file": "api.py",
            "line": 1,
            "severity": "high",
            "description": KEEP_ME,
            "confidence": "HIGH",
            "rationale": "keep this finding",
            "evidence": "api.py:1",
        },
        {
            "id": 2,
            "lens": "per-stack",
            "file": "api.py",
            "line": 1,
            "severity": "medium",
            "description": DROP_ME,
            "confidence": "MEDIUM",
            "rationale": "drop this finding",
            "evidence": "api.py:1",
        },
    ]


def _install_filtered_surface(
    ext_dir: ExtDir,
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Install the one extension fork and a real-path deep backend."""
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    ext_dir.write_module(FILTER_ITEMS_EXT)
    backend = _install_stub_backend(monkeypatch, target)
    backend.merge_items = _filtered_items()
    _silence(monkeypatch)
    return backend


def _serve_pr_view(fake_gh: FakeGh, target: Path) -> None:
    """Configure a PR whose SHAs match the real fixture repository."""
    fake_gh.serve_pr_view(
        {
            "number": 7,
            "state": "OPEN",
            "headRefName": "feature",
            "baseRefName": "main",
            "headRefOid": git_ops.head_sha(target),
            "baseRefOid": git_ops.merge_base(target, "main"),
            "url": "https://github.com/acme/widgets/pull/7",
            "body": "",
        }
    )


async def test_post_review_uses_published_items_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post step passes through the canonical path published by load-items."""
    private_dir = tmp_path / "private-deep"
    stable_items = tmp_path / "stable-merged-items.json"
    ctx = _post_context(dd=private_dir, items_file=stable_items)
    posted_paths: list[Path] = []
    monkeypatch.setattr(
        "daydream.pr_review.post_review_to_pr_from_report",
        lambda repo, path, *, console: _record_path(posted_paths, path),
    )

    await _step_post_review(ctx)

    assert posted_paths == [stable_items]


async def test_fork_filter_controls_findings_artifact(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_gh: Any,
    tmp_path: Path,
) -> None:
    """A load-items fork controls the canonical findings export surface."""
    _install_filtered_surface(ext_dir, multi_stack_target, monkeypatch)
    _serve_pr_view(fake_gh, multi_stack_target)
    findings_out = tmp_path / "findings.json"
    merged_items = multi_stack_target / ".daydream" / "deep" / "merged-items.json"

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            pr_number=7,
            findings_out=str(findings_out),
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )
    artifact = findings_out.read_text()
    canonical = merged_items.read_text()

    assert rc == 0
    assert KEEP_ME in artifact and KEEP_ME in canonical
    assert DROP_ME not in artifact and DROP_ME not in canonical


async def test_fork_filter_controls_pr_post_payload(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_gh: Any,
) -> None:
    """A load-items fork controls the canonical PR review payload."""
    _install_filtered_surface(ext_dir, multi_stack_target, monkeypatch)
    _serve_pr_view(fake_gh, multi_stack_target)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            pr_number=7,
            assume="yes",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )
    # A finding reaches the PR either in the review payload or as its own
    # file-level comment; the fork's filter must govern both surfaces.
    posted = json.dumps(
        [
            call.payload
            for call in (
                *fake_gh.calls("POST", "repos/acme/widgets/pulls/7/reviews"),
                *fake_gh.calls("POST", "repos/acme/widgets/pulls/7/comments"),
            )
        ]
    )

    assert rc == 0
    assert fake_gh.pr_view_calls()
    assert KEEP_ME in posted
    assert DROP_ME not in posted


async def test_fork_filter_controls_fix_prompts(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_gh: Any,
) -> None:
    """A load-items fork controls the findings that reach the fix phase."""
    from tests.test_deep_orchestrator import _fix_prompts

    backend = _install_filtered_surface(ext_dir, multi_stack_target, monkeypatch)
    _serve_pr_view(fake_gh, multi_stack_target)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            pr_number=7,
            assume="yes",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )
    fix_prompts = "\n".join(_fix_prompts(backend))

    assert rc == 0
    assert KEEP_ME in fix_prompts
    assert DROP_ME not in fix_prompts


class RecordingBackend:
    """Prompt-recording stub modelled on ``_PRFeedbackStubBackend``.

    Dispatches on prompt content just enough to drive the pr-feedback flow
    past every gate: writes the review-output file for the fetch prompt,
    yields ONE parseable feedback item for the parse prompt, and no-ops
    everything else (fix, commit, respond).
    """

    model = "mock-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.prompts.append(prompt)
        pl = prompt.lower()

        if "fetch-pr-feedback" in pl:
            (cwd / ".review-output.md").write_text(
                "# PR Feedback\n\n"
                "## x[bot]\n\n"
                "1. [api.py:1] `hello()` returns 'universe' but the docstring "
                "says 'world' — align the return value.\n"
            )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        if "extract only actionable issues" in pl:
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "issues": [
                        {
                            "id": 1,
                            "description": "Align hello() return value with docstring",
                            "file": "api.py",
                            "line": 1,
                            "confidence": "HIGH",
                            "rationale": "return value diverges from docstring",
                            "evidence": "api.py:1",
                        }
                    ]
                },
                continuation=None,
            )
            return

        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        # Mirror ClaudeBackend: append args so the test can read the slot from the prompt.
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result


class DeferredWriteBackend:
    """Yield a write start before performing the write on generator resumption."""

    model = "mock-model"
    fanout_concurrency = 4
    retry_attempts = 1
    retry_base_delay_s = 0.0

    def __init__(self, target: Path) -> None:
        self.target = target
        self.execute_calls = 0

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.execute_calls += 1
        yield ToolStartEvent(
            id="write-1",
            name="Write",
            input={"path": str(self.target), "content": "backend resumed"},
        )
        self.target.write_text("backend resumed")
        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_fork_disables_respond_step(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext removal of ``respond-feedback`` skips only that step.

    Observable outcomes: exit 0, the flow still ran (fetch prompt reached the
    backend), and the removed respond step never invoked its skill.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 3\n"
        "def register(r):\n"
        "    r.remove('pr-feedback', 'respond-feedback')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    rc = await runner.run_feedback(
        RunConfig(target=str(multi_stack_target), bot="x[bot]", non_interactive=True), pr=1
    )

    assert rc == 0
    assert any("fetch" in p.lower() for p in backend.prompts)  # flow still ran
    assert not any("respond-pr-feedback" in p for p in backend.prompts)  # removed step never invoked


async def test_fork_inserts_custom_phase_into_review_flow(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext phase inserted after ``review-alternatives`` runs in ``--review``.

    Observable outcomes: exit 0 and the custom phase's prompt reached the
    backend through the registered ``review`` flow.
    """
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "DAYDREAM_EXT_API = 3\n"
        "async def _ro(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    await run_agent(ctx.backend_for('ro_audit'), ctx.work.repo, 'RO-AUDIT-PROMPT',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='ro_audit', run=_ro))\n"
        "    r.insert_after('review', anchor='review-alternatives', step='ro_audit')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            output_mode="review",
            non_interactive=True,
            archive=False,
        )
    )

    idx = [i for i, p in enumerate(backend.prompts) if p == "RO-AUDIT-PROMPT"]
    assert idx, "custom phase never reached the backend"
    assert rc == 0


class ShallowRecordingBackend(PhaseDispatchBackend):
    """The shared shallow phase-dispatch fake, plus full-prompt recording.

    ``PhaseDispatchBackend`` drives the shallow review-parse-fix-test flow
    past every gate; ``prompts`` records the exact prompt each ``execute``
    call received so the test can assert the fork phase's prompt arrived.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompts: list[str] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.prompts.append(prompt)
        async for event in super().execute(
            cwd, prompt, output_schema, continuation, agents, max_turns, read_only
        ):
            yield event


async def test_fork_inserts_phase_before_summary_in_shallow(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext phase inserted before ``summary`` runs in ``--shallow``.

    Observable outcomes: exit 0 and the custom phase's prompt reached the
    backend through the registered ``shallow`` flow (after the iterate loop
    group, before the summary step).
    """
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "DAYDREAM_EXT_API = 3\n"
        "async def _ro(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    await run_agent(ctx.backend_for('ro_shallow'), ctx.work.repo, 'RO-SHALLOW-PROMPT',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='ro_shallow', run=_ro))\n"
        "    r.insert_before('shallow', anchor='summary', step='ro_shallow')\n"
    )
    backend = ShallowRecordingBackend(
        parse_results=[[{"id": 1, "description": "Align hello() return value", "file": "api.py", "line": 1}]]
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            shallow=True,
            skill="python",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    assert "RO-SHALLOW-PROMPT" in backend.prompts, "fork phase never reached the backend"
    assert rc == 0


# Distinctive literal from ``build_alternative_review_prompt``'s body: present in
# every alternatives prompt, absent from every other prompt builder.
ALTERNATIVES_MARKER = "Given this intent, explore the codebase and evaluate the implementation"


async def test_fork_disables_alternatives_in_deep(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext removal of ``alternatives`` skips only that step in deep.

    Observable outcomes: exit 0, the deep pipeline still ran (the intent prompt
    reached the backend), and the removed alternatives step never sent its
    prompt. The presence + exit-code assertions make the absence assertion
    discriminating: a run that no-ops entirely fails the presence check.
    """
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    ext_dir.write_module(
        "DAYDREAM_EXT_API = 3\n"
        "def register(r):\n"
        "    r.remove('deep', 'alternatives')\n"
    )
    backend = _install_stub_backend(monkeypatch, multi_stack_target)
    _silence(monkeypatch)

    # The PR post runs before the fix gate; stub the non-idempotent GitHub write.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    prompts = [call["prompt"] for call in backend.calls]
    assert rc == 0
    assert any("intent" in p.lower() for p in prompts)  # pipeline still ran
    assert not any(ALTERNATIVES_MARKER in p for p in prompts)  # removed step never sent its prompt


# Task 17 fixture source: a fork registers phase ``ro_gate`` whose run() resolves
# its OWN backend (``ctx.backend_for('ro_gate')`` -> per-phase config), its OWN
# registered prompt (``prompt('ro_gate')``), and its phase-bound skill slot
# (``skill('phase:ro_gate')``), then inserts it into the deep flow after ``intent``.
FULL_RO_EXT = (
    "from daydream.extensions import FlowStep, get_registry\n"
    "DAYDREAM_EXT_API = 3\n"
    "def _ro_prompt(skill):\n"
    "    return f'RO-GATE {skill}'\n"
    "async def _ro(ctx):\n"
    "    from daydream.agent import run_agent\n"
    "    from daydream.trajectory import DaydreamPhase\n"
    "    r = get_registry()\n"
    "    prompt = r.prompt('ro_gate')(skill=r.skill('phase:ro_gate'))\n"
    "    await run_agent(ctx.backend_for('ro_gate'), ctx.work.repo, prompt,\n"
    "                    phase=DaydreamPhase.REVIEW)\n"
    "def register(r):\n"
    "    r.register_phase(FlowStep(name='ro_gate', run=_ro))\n"
    "    r.override_prompt('ro_gate', _ro_prompt)\n"
    "    r.override_skill('phase:ro_gate', 'ro-core:gate-skill')\n"
    "    r.insert_after('deep', anchor='intent', step='ro_gate')\n"
)


# A minimal fork flow: one custom step that sends a marker prompt, registered
# as a brand-new flow name (NOT one of the four built-ins).
CUSTOM_FLOW_EXT = (
    "from daydream.extensions import FlowStep\n"
    "DAYDREAM_EXT_API = 3\n"
    "async def _audit(ctx):\n"
    "    from daydream.agent import run_agent\n"
    "    from daydream.trajectory import DaydreamPhase\n"
    "    await run_agent(ctx.backend_for('ro_audit'), ctx.work.repo, 'CUSTOM-FLOW-PROMPT',\n"
    "                    phase=DaydreamPhase.REVIEW)\n"
    "def register(r):\n"
    "    r.register_phase(FlowStep(name='ro_audit', run=_audit))\n"
    "    r.set_flow('ro-audit', ['ro_audit'])\n"
)


def _step_for_tool(trajectory: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Return the recorded agent step containing a call to *tool_name*."""
    for step in trajectory["steps"]:
        if any(
            call.get("function_name") == tool_name
            for call in step.get("tool_calls", [])
        ):
            return step
    raise AssertionError(f"no trajectory step recorded tool {tool_name!r}")


async def _run_tool_case(
    ext_dir: ExtDir,
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    register_supervisor: bool,
    supervisor_raises: bool = False,
    backend_capture: list[DeferredWriteBackend] | None = None,
) -> tuple[Path, Path, int]:
    """Run the extension-defined custom flow against the deferred-write backend."""
    supervisor_registration = ""
    if register_supervisor:
        if supervisor_raises:
            supervisor_registration = (
                "    class RetryableSupervisorError(RuntimeError):\n"
                "        retryable = True\n"
                "    def _supervise(name, tool_input, *, phase):\n"
                "        raise RetryableSupervisorError('supervisor failed')\n"
                "    r.register_tool_supervisor(_supervise)\n"
            )
        else:
            supervisor_registration = (
                "    def _supervise(name, tool_input, *, phase):\n"
                "        return ToolDecision(veto=name == 'Write', reason='protected path')\n"
                "    r.register_tool_supervisor(_supervise)\n"
            )

    ext_dir.write_module(
        "from daydream.extensions import FlowStep, ToolDecision\n"
        "DAYDREAM_EXT_API = 3\n"
        "async def _audit(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    await run_agent(ctx.backend_for('ro_audit'), ctx.work.repo, 'CUSTOM-TOOL-PROMPT',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='ro_audit', run=_audit))\n"
        "    r.set_flow('ro-audit', ['ro_audit'])\n"
        + supervisor_registration
    )

    written = target / "deferred-write.txt"
    trajectory = target / ".daydream" / "tool-supervisor-trajectory.json"
    backend = DeferredWriteBackend(written)
    if backend_capture is not None:
        backend_capture.append(backend)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(target),
            flow_name="ro-audit",
            non_interactive=True,
            cleanup=False,
            archive=False,
            trajectory_path=trajectory,
        )
    )
    return written, trajectory, rc


async def test_builtin_and_fork_tool_supervisor_conflict_fails_loud(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Config-enabled built-in and fork supervisors cannot silently compose."""
    from daydream.config_file import load_file_config

    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "DAYDREAM_EXT_API = 3\n"
        "def _fork_supervisor(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r):\n"
        "    r.register_tool_supervisor(_fork_supervisor)\n"
    )
    (multi_stack_target / ".daydream.toml").write_text('tool_supervisor = "rules"\n')
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        runner.RunConfig(
            target=str(multi_stack_target),
            non_interactive=True,
            cleanup=False,
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert rc == 1
    output = capsys.readouterr().out.lower()
    assert "tool supervisor conflict" in output
    assert "config-enabled built-in" in output
    assert "extension-registered" in output


async def test_fork_tool_supervisor_vetoes_write(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered supervisor veto closes the deferred backend before its write."""
    denied, traj, rc = await _run_tool_case(
        ext_dir, multi_stack_target, monkeypatch, register_supervisor=True
    )

    assert rc == 0
    assert not denied.exists()
    step = _step_for_tool(json.loads(traj.read_text()), "Write")
    assert step["tool_calls"][0]["function_name"] == "Write"
    assert step["extra"]["stop_reason"] == "tool_vetoed:Write"


async def test_no_tool_supervisor_allows_write(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without registration, the same deferred backend resumes and writes."""
    written, traj, rc = await _run_tool_case(
        ext_dir, multi_stack_target, monkeypatch, register_supervisor=False
    )

    assert rc == 0
    assert written.read_text() == "backend resumed"
    assert "tool_vetoed:Write" not in traj.read_text()


async def test_retryable_tool_supervisor_failure_propagates_without_retry(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retryable supervisor error propagates without entering backend retry."""
    backends: list[DeferredWriteBackend] = []

    with pytest.raises(RuntimeError, match="supervisor failed") as exc_info:
        await _run_tool_case(
            ext_dir,
            multi_stack_target,
            monkeypatch,
            register_supervisor=True,
            supervisor_raises=True,
            backend_capture=backends,
        )

    assert getattr(exc_info.value, "retryable", False) is True
    assert len(backends) == 1
    assert backends[0].execute_calls == 1


async def test_custom_flow_dispatches_and_dumps_artifacts(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path, tmp_path: Path,
) -> None:
    """A fork-registered custom flow selected via flow_name runs end-to-end and
    --dump-artifacts writes the bundle (must-haves 1 + 2)."""
    ext_dir.write_module(CUSTOM_FLOW_EXT)
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    dump_dir = tmp_path / "uploaded-artifacts"
    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            flow_name="ro-audit",
            non_interactive=True,
            cleanup=False,
            archive=False,
            dump_artifacts=str(dump_dir),
        )
    )

    assert rc == 0
    assert "CUSTOM-FLOW-PROMPT" in backend.prompts  # custom flow actually ran
    assert (dump_dir / "manifest.json").is_file()    # dump fired on the custom path
    assert (dump_dir / "trajectory.json").is_file()


async def test_unknown_flow_name_errors(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered flow name fails with exit 1 (Extension Error panel; must-have 3)."""
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            flow_name="does-not-exist",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    assert rc == 1
    assert not any("CUSTOM-FLOW-PROMPT" in p for p in backend.prompts)


async def test_pr_feedback_not_selectable_via_flow(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--flow pr-feedback errors (needs PR number + bot; must-have 5)."""
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            flow_name="pr-feedback",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )
    assert rc == 1


async def test_custom_phase_full_stack(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam acceptance (Task 17): custom phase end-to-end through ``runner.run``.

    Proves the four must-haves are wired together: a fork-registered phase runs
    inside the deep flow, builds its prompt from its OWN registered prompt
    builder, resolves its OWN phase-bound skill slot, and gets its backend
    through ``[tool.daydream.phases.ro_gate]`` per-phase config (Assumption 7:
    ``_coerce_phases`` / ``_resolved_model`` accept arbitrary phase strings).

    Observable outcomes: exit 0, the ``RO-GATE`` prompt containing the bound
    skill reached the backend, and ``create_backend`` was called with the
    per-phase model from ``.daydream.toml``.
    """
    from daydream.config_file import load_file_config
    from tests.test_deep_orchestrator import _silence, _StubBackend

    ext_dir.write_module(FULL_RO_EXT)
    (multi_stack_target / ".daydream.toml").write_text('[phases.ro_gate]\nmodel = "test-model-x"\n')

    backend = _StubBackend(multi_stack_target)
    created: list[tuple[str, str | None]] = []

    def fake_create(name: str, model: str | None = None, **kwargs: object) -> _StubBackend:
        created.append((name, model))
        return backend

    monkeypatch.setattr("daydream.runner.create_backend", fake_create)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    _silence(monkeypatch)

    # The PR post runs before the fix gate; stub the non-idempotent GitHub write.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            non_interactive=True,
            cleanup=False,
            archive=False,
            file_config=load_file_config(multi_stack_target),
        )
    )

    prompts = [call["prompt"] for call in backend.calls]
    ro_prompts = [p for p in prompts if p.startswith("RO-GATE")]
    assert rc == 0
    assert ro_prompts and "ro-core:gate-skill" in ro_prompts[0]  # own prompt + bound skill
    assert ("claude", "test-model-x") in created  # [tool.daydream.phases.ro_gate] honored


async def test_flow_deep_routes_to_deep_helper(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--flow deep runs the real deep pipeline (must-have 4): the intent prompt
    reaches the backend via the deep flow, exit 0."""
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    backend = _install_stub_backend(monkeypatch, multi_stack_target)
    _silence(monkeypatch)

    async def _no_post(target_dir: Path, report_path: Path, *, console) -> None:
        return None
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            flow_name="deep",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    prompts = [call["prompt"] for call in backend.calls]
    assert rc == 0
    assert any("intent" in p.lower() for p in prompts)  # deep pipeline ran


async def test_flow_review_routes_to_review_helper(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--flow review runs the real review pipeline: the alternatives prompt
    reaches the backend via the review flow, exit 0."""
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            flow_name="review",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    assert rc == 0
    assert any(ALTERNATIVES_MARKER in p for p in backend.prompts)  # review pipeline ran


async def test_flow_shallow_routes_to_shallow_helper(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--flow shallow runs the real shallow pipeline: the parse phase fires,
    exit 0."""
    backend = ShallowRecordingBackend(
        parse_results=[[{"id": 1, "description": "Align return", "file": "api.py", "line": 1}]]
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            flow_name="shallow",
            skill="python",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    assert rc == 0
    assert backend.parse_calls >= 1  # shallow pipeline ran
