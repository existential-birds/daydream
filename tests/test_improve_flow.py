import copy
import hashlib
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream import git_ops, runner as runner_module
from daydream.backends import AgentEvent, Backend, ResultEvent, TextEvent
from daydream.backends.claude import ClaudeBackend
from daydream.config import AUDIT_CATEGORIES, EFFORT_TIERS, PHASE_DEFAULT_MODELS, VET_BATCH_MAX_FINDINGS
from daydream.config_file import DaydreamFileConfig, load_file_config
from daydream.deep.detection import StackAssignment
from daydream.exploration import EXPLORATION_SECTION_PREFIX
from daydream.exploration_runner import repo_scan
from daydream.extensions.loader import build_registry
from daydream.flows.engine import FlowContext
from daydream.git_ops import GitError, head_sha
from daydream.improve import authoring
from daydream.improve.audit import (
    _apply_vet_verdicts,
)
from daydream.improve.audit_scope import (
    _restrict_diff_to_services,
    _stacks_for_services,
    _stamp_finding,
)
from daydream.improve.context import _audit_repo
from daydream.improve.partition import Partition
from daydream.improve.planning import (
    _step_write_plans,
)
from daydream.improve.prompts import (
    PLAN_AUTHOR_SCHEMA,
    build_audit_prompt,
)
from daydream.improve.recon import _build_recon_prompt, _host_enumerated_commands
from daydream.prompts.grounding import (
    REPOSITORY_SEARCH_GUIDANCE,
    UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY,
)
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.services import Service, enumerate_services
from daydream.workspace import AuditWorkspace, WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import (
    commit,
    git,
    init_repo,
)
from tests.harness.improve_backend import (
    ImproveStubBackend,
    ManifestReadingBackend,
    ProductionPathBackend,
    RendezvousImproveBackend,
    improve_artifact,
    install_capable_improve_backend,
    install_improve_stub,
    stub_recon_commands,
)
from tests.harness.review_profile import default_strategy as _default_strategy
from tests.harness.stub_backend import force_interactive as _force_interactive
from tests.harness.trajectory import (
    dispatch_encloses_children,
    root_trajectory as _root_run_trajectory,
)

MakeConfig = Callable[..., RunConfig]

@pytest.fixture
def stub(improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch) -> ImproveStubBackend:
    return install_improve_stub(monkeypatch, improve_monorepo_target)


def _record_phase_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each resolved phase backend its own (model, reasoning_effort) while sharing the call log."""
    shared_factory = cast(Callable[..., ImproveStubBackend], getattr(runner_module, "create_backend"))

    def _per_phase(*args: Any, **kwargs: Any) -> ImproveStubBackend:
        backend = copy.copy(shared_factory(*args, **kwargs))
        backend.model, backend.reasoning_effort = kwargs["model"], kwargs["reasoning_effort"]
        return backend

    monkeypatch.setattr(runner_module, "create_backend", _per_phase)


def _tiers_by_marker(calls: list[dict[str, Any]]) -> dict[str, set[tuple[str, str | None]]]:
    tiers: dict[str, set[tuple[str, str | None]]] = {}
    for call in calls:
        tiers.setdefault(call["marker"], set()).add((call["model"], call["reasoning_effort"]))
    return tiers


def _load_improve_json(repo: Path, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(improve_artifact(repo, name).read_text(encoding="utf-8")))

def _plan_write_diagnostics(repo: Path) -> dict[str, Any]:
    return _load_improve_json(repo, "plan-write-diagnostics.json")

def _flow_context_with_audit_root(root: Path, *, audit_repo: Path,) -> FlowContext:
    work = WorkContext(
        repo=root, source=root, base_branch="main", base_sha="1" * 40, head_branch="feature", head_sha="2" * 40,
        is_ephemeral=False, run_id="run-1",
    )
    boundary = AuditWorkspace(
        repo=audit_repo, source=root, repo_git_common_dir=audit_repo / ".git", source_git_common_dir=root / ".git",
        outward_symlinks=frozenset(),
    )
    return FlowContext(config=RunConfig(target=str(root), flow_name="improve"), work=work, registry=build_registry(),
        audit_workspace=boundary,
    )

@pytest.mark.anyio
async def test_unsupported_improve_backend_fails_atomic_preflight(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    execute_calls: list[str] = []

    async def _execute_canary(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        del args, kwargs
        execute_calls.append("claude")
        yield TextEvent(text="UNEXPECTED_CLAUDE_EXECUTION")

    monkeypatch.setattr(ClaudeBackend, "execute", _execute_canary)
    file_config = DaydreamFileConfig(phases={"recon": {"backend": "codex"}})
    before_head = head_sha(improve_monorepo_target)
    before_status = _git_status_porcelain(improve_monorepo_target)
    code = await _run_improve(make_config, improve_monorepo_target, file_config=file_config)
    output = capsys.readouterr().out
    assert code == 1
    assert execute_calls == []
    assert "Phase 'recon'" in output
    assert "backend 'codex'" in output
    assert "Use backend 'claude'" in output
    assert len(output) < 5_000
    assert head_sha(improve_monorepo_target) == before_head
    assert _git_status_porcelain(improve_monorepo_target) == before_status


def test_improve_cli_discovers_nested_workspace_and_preserves_downstream_facts(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, cli_runner: Any,
) -> None:
    """One CLI discovery retains tracked guidance and nested command evidence downstream.

    Ignored and untracked source-only manifests never reach a model input or recon.json.
    """
    repo = improve_monorepo_target
    nested = repo / "platform" / "payments" / "ui"
    nested.mkdir(parents=True)
    (repo / "package.json").write_text(json.dumps({"workspaces": ["platform/*/ui"]}) + "\n")
    (nested / "package.json").write_text(json.dumps({"scripts": {"test": "touch candidate-command-ran"}}, indent=2))
    (nested / "App.tsx").write_text("export const App = () => <div/>;\n")
    (nested / "AGENTS.md").write_text("Preserve payment accessibility conventions.\n")
    git(repo, "add", "package.json", "platform")
    commit(repo, "add nested workspace")
    canaries = ("PRIVATE_IGNORED_MAKE_CANARY", "PRIVATE_UNTRACKED_MAKE_CANARY", "private-ignored-service-canary")
    (repo / ".git/info/exclude").write_text("/Makefile\n/services/\n", encoding="utf-8")
    (repo / "Makefile").write_text(f"test-private: ; @echo {canaries[0]}\n")
    (nested / "Makefile").write_text(f"test-private: ; @echo {canaries[1]}\n")
    (repo / "services" / canaries[2]).mkdir(parents=True)
    (repo / "services" / canaries[2] / "pyproject.toml").write_text(f'[project]\nname = "{canaries[2]}"\n')
    before_head = head_sha(repo)

    class NestedDiscoveryBackend(ImproveStubBackend):
        async def _pre_execute(self, cwd: Path, prompt: str, **kwargs: Any) -> list[AgentEvent] | None:
            if "IMPROVE_RECON" in prompt:
                guidance_path = "platform/payments/ui/AGENTS.md"
                guidance = (cwd / guidance_path).read_text(encoding="utf-8").strip()
                self.recon_output_override = {
                    "languages": ["python", "typescript", "FastAPI"], "commands": stub_recon_commands(),
                    "conventions": ["OpenAPI First: openapi.yaml is the HTTP contract (source: AGENTS.md)"],
                    "guidelines": ["Keep API examples synchronized with openapi.yaml.",
                        f"{guidance} (source: {guidance_path})"],
                    "intent_docs": ["README.md"],
                }
            return None

    backend = install_capable_improve_backend(monkeypatch, NestedDiscoveryBackend(repo, n_findings=1))
    _record_phase_tiers(monkeypatch)
    result = cli_runner.invoke(["improve", str(repo), "--non-interactive", "--no-archive", "--no-eval"])

    assert result.exit_code == 0
    # Each Improve phase resolves its own built-in model and reasoning tier.
    models = PHASE_DEFAULT_MODELS["claude"]
    assert _tiers_by_marker(backend.calls) == {
        "recon": {(models["recon"], "low")}, "audit": {(models["audit"], "high")},
        "vet": {(models["vet"], "xhigh")}, "plan-writer": {(models["plan_write"], "max")},
    }
    discovery = [call for call in backend.calls if call["marker"] in {"repo-scan", "recon"}]
    assert len(discovery) == 1
    assert discovery[0]["marker"] == "recon"
    assert "Combined repository discovery" in discovery[0]["prompt"]
    assert discovery[0]["prompt"].count(REPOSITORY_SEARCH_GUIDANCE) == 1
    assert "only Read/Glob/Grep inspection tools; Bash and\ncommand execution are unavailable" in discovery[0]["prompt"]
    assert "do not execute helper, Git, or discovered commands" in discovery[0]["prompt"]
    assert "platform/payments/ui/AGENTS.md" in discovery[0]["prompt"]
    assert "platform/payments/ui" in discovery[0]["prompt"]
    recon = _load_improve_json(repo, "recon.json")
    command = next(item for item in recon["commands"] if item["command"] == "npm run test")
    assert command["working_directory"] == "platform/payments/ui"
    assert command["applicability"]["scope"] == {"kind": "in-scope-paths", "paths": ["platform/payments/ui"]}
    assert command["evidence"]["source_path"] == "platform/payments/ui/package.json"
    assert command["evidence"]["kind"] == "host-derived"
    assert recon["discovery_mode"] == "combined"
    nested_guidance = "Preserve payment accessibility conventions. (source: platform/payments/ui/AGENTS.md)"
    assert nested_guidance in recon["guidelines"]
    for marker in ("audit", "plan-writer"):
        calls = [call for call in backend.calls if call["marker"] == marker]
        assert calls
        for call in calls:
            assert "openapi.yaml is the HTTP contract (source: AGENTS.md)" in call["prompt"]
            assert "Keep API examples synchronized with openapi.yaml." in call["prompt"]
            assert nested_guidance in call["prompt"]
            assert '"FastAPI"' in call["prompt"]
            assert '"README.md"' in call["prompt"]
            assert '"npm run test"' in call["prompt"]
            assert '"platform/payments/ui"' in call["prompt"]
    assert improve_artifact(repo, "report.md").is_file()
    assert list((repo / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert all(call["read_only"] for call in backend.calls)
    assert not (nested / "candidate-command-ran").exists()
    model_inputs = [call["prompt"] for call in backend.calls] + [json.dumps(recon)]
    assert all(canary not in text for canary in canaries for text in model_inputs)
    assert head_sha(repo) == before_head
    assert _git_status_porcelain(repo) == ""


@pytest.mark.anyio
async def test_custom_improve_survey_uses_compatible_two_pass_fallback(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_config: MakeConfig,
) -> None:
    """A custom survey still runs, and its guidance survives a recon that omits it."""
    repo = improve_monorepo_target
    profile = tmp_path / "custom-survey.toml"
    profile.write_text(
        'schema_version = 1\nname = "custom-survey"\n'
        '[strategies."exploration.repository_survey"]\n'
        'content = "You are the **repo-survey** specialist. CUSTOM_SURVEY_INSTRUCTION"\nsource = "test"\n'
    )

    class CustomSurveyBackend(ImproveStubBackend):
        async def _pre_execute(self, cwd: Path, prompt: str, **kwargs: Any) -> list[AgentEvent] | None:
            if "CUSTOM_SURVEY_INSTRUCTION" in prompt:
                self._record_call(cwd, prompt, "repo-scan", **kwargs)
                raise RuntimeError("survey unavailable")
            return None

    backend = install_capable_improve_backend(monkeypatch, CustomSurveyBackend(repo, n_findings=1))
    backend.recon_output_override = {
        "languages": ["python"], "commands": stub_recon_commands(), "conventions": [], "intent_docs": [],
    }
    assert await _run_improve(make_config, repo, review_profile_path=profile) == 0
    discovery = [call for call in backend.calls if call["marker"] in {"repo-scan", "recon"}]
    assert [call["marker"] for call in discovery] == ["repo-scan", "recon"]
    assert "CUSTOM_SURVEY_INSTRUCTION" in discovery[0]["prompt"]
    assert "Combined repository discovery" not in discovery[1]["prompt"]
    recon = _load_improve_json(repo, "recon.json")
    assert recon["discovery_mode"] == "survey-recon"
    assert recon["conventions"] == []
    assert recon["guidelines"] == []
    events = _root_run_trajectory(repo)["extra"]["phase_events"]
    assert any(event["phase"] == "exploration" and event["metadata"] == {"stage": "repo-survey"} for event in events)
    assert all(not Path(call["cwd"]).exists() for call in backend.calls)


@pytest.mark.anyio
async def test_improve_git_environment_refusal_names_key_without_value(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig, tmp_path: Path,
) -> None:
    repo = improve_monorepo_target
    private_config = tmp_path / "PRIVATE_ENV_VALUE_SENTINEL"
    private_config.write_text("", encoding="utf-8")
    private_value = str(private_config)
    backend = install_improve_stub(monkeypatch, repo)
    before_head = head_sha(repo)
    before_status = _git_status_porcelain(repo)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", private_value)

    code = await _run_improve(make_config, repo)
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert code == 1
    assert backend.calls == []
    assert "GIT_CONFIG_GLOBAL" in output
    assert private_value not in output
    assert len(output) < 5_000
    assert head_sha(repo) == before_head
    assert _git_status_porcelain(repo) == before_status

class _UnbornAuditBackend(ImproveStubBackend):
    """Capability fake that observes and commits inside its unborn snapshot."""

    def __init__(self, target: Path) -> None:
        super().__init__(target, n_findings=0)
        self.audit_paths: list[Path] = []
        self.first_symbolic_head: str | None = None
        self.first_was_unborn: bool | None = None
        self.ignored_visible: bool | None = None

    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if not self.audit_paths:
            self.first_was_unborn = git_ops.is_unborn_head(cwd)
            self.first_symbolic_head = git(cwd, "symbolic-ref", "--short", "HEAD")
            self.ignored_visible = (cwd / "ignored-secret.txt").exists()
            self.audit_paths.append(cwd)
            (cwd / "model-scratch.txt").write_text("snapshot only\n", encoding="utf-8")
            git(cwd, "add", "model-scratch.txt")
            git(cwd, "-c", "user.name=Audit Test", "-c", "user.email=audit@example.invalid", "commit", "-m",
                "snapshot initial commit",
            )
        elif cwd not in self.audit_paths:
            self.audit_paths.append(cwd)
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

class _SymlinkGuardBackend(ImproveStubBackend):
    """Drive the production Claude guard inside a real audit snapshot."""

    def __init__(self, target: Path) -> None:
        super().__init__(target, n_findings=0)
        self.audit_paths: list[Path] = []
        self.glob_denied: bool | None = None
        self.external_enumeration_attempted = False

    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if cwd not in self.audit_paths:
            self.audit_paths.append(cwd)
        if self.glob_denied is None:
            backend = ClaudeBackend(model="opus", audit_root=cwd, audit_outward_symlinks=self.audit_outward_symlinks,)
            guard = backend._audit_root_guard  # noqa: SLF001 - security boundary
            assert guard is not None
            decision = cast(dict[str, Any], await guard(
                cast(Any, {"tool_name": "Glob", "tool_input": {"path": "sub", "pattern": "loop/escape/*"},}), None,
                {"signal": None},
            ))
            self.glob_denied = (decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny")
            if not self.glob_denied:
                self.external_enumeration_attempted = True
                list((cwd / "sub" / "loop" / "escape").iterdir())
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

def _make_unborn_improve_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "unborn-improve"
    init_repo(repo)
    git(repo, "symbolic-ref", "HEAD", "refs/heads/seedless")
    (repo / ".gitignore").write_text("ignored-secret.txt\n", encoding="utf-8")
    (repo / "tracked.bin").write_bytes(b"\x00staged\xff")
    (repo / "ignored-secret.txt").write_text("do-not-copy\n", encoding="utf-8")
    git(repo, "add", ".gitignore", "tracked.bin")
    return repo

@pytest.mark.anyio
async def test_full_improve_denies_glob_through_inward_then_outward_symlink(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, tmp_path: Path,
) -> None:
    repo = improve_monorepo_target
    outside = tmp_path / "private outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("never enumerate\n", encoding="utf-8")
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    sub = repo / "sub"
    sub.mkdir()
    (sub / "loop").symlink_to("..", target_is_directory=True)
    git(repo, "add", "escape", "sub/loop")
    commit(repo, "add audit symlink topology")
    before_head = head_sha(repo)
    before_status = _git_status_porcelain(repo)
    backend = install_capable_improve_backend(monkeypatch, _SymlinkGuardBackend(repo))
    code = await _run_improve(make_config, repo)

    assert code == 0
    assert backend.glob_denied is True
    assert backend.external_enumeration_attempted is False
    assert backend.audit_paths
    assert all(not path.exists() for path in backend.audit_paths)
    assert head_sha(repo) == before_head
    assert _git_status_porcelain(repo) == before_status
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "never enumerate\n"

@pytest.mark.anyio
async def test_full_improve_uses_independent_unborn_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    repo = _make_unborn_improve_repo(tmp_path)
    before_patch = git_ops.staged_patch(repo)
    before_status = _git_status_porcelain(repo)
    backend = install_capable_improve_backend(monkeypatch, _UnbornAuditBackend(repo))
    code = await _run_improve(make_config, repo)

    assert code == 0
    assert backend.first_was_unborn is True
    assert backend.first_symbolic_head == "seedless"
    assert backend.ignored_visible is False
    assert backend.calls
    assert git_ops.is_unborn_head(repo)
    assert git_ops.staged_patch(repo) == before_patch
    assert _git_status_porcelain(repo) == before_status
    assert all(not path.exists() for path in backend.audit_paths)

@pytest.mark.anyio
async def test_unborn_improve_cancellation_cleans_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    repo = _make_unborn_improve_repo(tmp_path)
    started = anyio.Event()
    audit_paths: list[Path] = []

    async def _blocking_stream() -> AsyncIterator[AgentEvent]:
        yield TextEvent(text="Audit cancellation fixture ready")
        started.set()
        await anyio.sleep_forever()

    def responder(cwd: Any, *_args: Any, **_kwargs: Any) -> AsyncIterator[AgentEvent]:
        audit_paths.append(cwd)
        return _blocking_stream()

    backend: Any = install_capable_improve_backend(monkeypatch,
        cast(ImproveStubBackend,
            ScriptedBackend(responder=responder, audit_paths=audit_paths, retry_attempts=20, retry_base_delay_s=0.0,
                retry_max_delay_s=0.0,
            ),
        ),
    )

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_run_improve, make_config, repo)
        await started.wait()
        tasks.cancel_scope.cancel()

    assert backend.audit_paths
    assert git_ops.is_unborn_head(repo)
    assert all(not path.exists() for path in backend.audit_paths)

@pytest.mark.anyio
async def test_unborn_commit_anchored_improve_modes_fail_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], make_config: MakeConfig,
) -> None:
    repo = _make_unborn_improve_repo(tmp_path)
    factory_calls: list[object] = []
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: factory_calls.append((args, kwargs)),)
    for config_fields, message in (({"improve_focus": "branch"}, "--focus branch requires a commit anchor"),
            ({"improve_plan_description": "add checks"}, "requires a planned-at commit")):
        assert await _run_improve(make_config, repo, **config_fields) == 1
        assert message in capsys.readouterr().out
    assert factory_calls == []
    assert git_ops.is_unborn_head(repo)

def test_audit_repo_requires_flow_context_boundary_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    context = _flow_context_with_audit_root(source, audit_repo=audit)
    context.data["audit_repo"] = audit
    assert _audit_repo(context) == audit
    context.data["audit_repo"] = tmp_path / "forged"
    with pytest.raises(RuntimeError, match="audit workspace"):
        _audit_repo(context)

@pytest.mark.anyio
async def test_unborn_plan_write_never_constructs_a_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected_count = 2
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    work = WorkContext(
        repo=source, source=source, base_branch="trunk", base_sha=None, head_branch="trunk", head_sha=None,
        is_ephemeral=False, run_id="run-unborn",
    )
    boundary = AuditWorkspace(
        repo=audit, source=source, repo_git_common_dir=audit / ".git", source_git_common_dir=source / ".git",
        outward_symlinks=frozenset(),
    )
    context = FlowContext(
        config=RunConfig(target=str(source), flow_name="improve"), work=work, registry=build_registry(),
        audit_workspace=boundary,
    )
    improve_dir = source / ".daydream" / "improve"
    improve_dir.mkdir(parents=True)
    context.data.update({"improve_dir": improve_dir, "selection_mode": "non-interactive-default",
            "selected_findings": [{"title": f"Finding {index}", "fingerprint": f"fingerprint-{index}",}
                for index in range(selected_count)
            ],
        }
    )

    def fail_backend_for(self: FlowContext, phase: str) -> Backend:
        raise AssertionError(f"backend_for unexpectedly called for {phase}")

    monkeypatch.setattr(FlowContext, "backend_for", fail_backend_for)

    await _step_write_plans(context)

    result = context.data["plan_write"]
    assert result["written"] == []
    assert result["skipped"] == []
    assert len(result["failed"]) == selected_count
    assert len(result["diagnostics"]) == selected_count
    assert context.data["plan_exit_code"] == 1
    for entry in [*result["failed"], *result["diagnostics"]]:
        assert "UNBORN_PLAN_ANCHOR_UNAVAILABLE" in json.dumps(entry)
    assert not (source / "daydream_plans").exists()

_GROUP = {"name": "group-01", "stack": "python", "file_count": 24,
    "partitions": [{"name": "billing", "root": "apps/billing", "file_count": 12, "service": "billing"},
        {"name": "web", "root": "frontend/web", "file_count": 12, "service": None},
    ],
}

def _audit_prompt(category: str, *, tier: str = "standard") -> str:
    """Build one improve-audit prompt against the shared ``_GROUP`` envelope."""
    return build_audit_prompt(
        category=category, strategy=_default_strategy(f"improve.audit.{category}"), group=_GROUP, scope_note="",
        recon_summary="{}", cwd=Path("/repo"), tier=EFFORT_TIERS[tier],
    )

def test_audit_prompt_carries_group_roots_and_no_file_list() -> None:
    prompt = _audit_prompt("correctness")
    assert "apps/billing" in prompt and "group-01" in prompt
    assert "Relevant tracked files:" not in prompt
    assert "<primary_manifest>\n{}\n</primary_manifest>" in prompt
    assert "frontend/web" in prompt and "service billing" in prompt
    assert "Deep audit: search every relevant package very thoroughly." in _audit_prompt("correctness", tier="deep")

@pytest.fixture
def tmp_git_repo(improve_monorepo_target: Path) -> Path:
    return improve_monorepo_target

def _git_status_porcelain(repo: Path) -> str:
    return git(repo, "status", "--porcelain", "--untracked-files=no")

def _dispatch_for_phase(trajectory: dict[str, Any], phase: str) -> dict[str, Any]:
    steps = [step
        for step in trajectory["steps"]
        if step.get("extra", {}).get("daydream_phase") == phase
        and "dispatch_id" in step.get("extra", {})
    ]
    assert len(steps) == 1
    return cast(dict[str, Any], steps[0])

def _phase_end(trajectory: dict[str, Any], phase: str) -> dict[str, Any]:
    ends = [event
        for event in trajectory["extra"]["phase_events"]
        if event["event"] == "phase_end" and event["phase"] == phase
    ]
    assert len(ends) == 1
    return cast(dict[str, Any], ends[0])

def _improve_observable_texts(repo: Path) -> list[str]:
    return [path.read_text(encoding="utf-8")
        for root in (repo / ".daydream" / "improve", repo / ".daydream" / "runs", repo / "daydream_plans",)
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]

def _raise_enumeration_failure(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    raise RuntimeError("unparseable repository manifest")

def _no_open_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.git_ops.gh_issue_list_strict", lambda *args, **kwargs: [])

@pytest.mark.anyio
async def test_repo_scan_sample_spans_the_tree_not_the_alphabetical_head(tmp_git_repo: Path,) -> None:
    """A capped sample must still reach real source, not just dotfile dirs."""
    dotdir = tmp_git_repo / ".agents" / "skills"
    dotdir.mkdir(parents=True)
    for i in range(40):
        (dotdir / f"skill-{i:02d}.md").write_text(f"# skill {i}\n")
    git(tmp_git_repo, "add", ".")
    commit(tmp_git_repo, "add skills")
    stub = ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(cast(Backend, stub), tmp_git_repo, max_files=10)
    prompt = stub.calls[0]["prompt"]
    sample = prompt.split("<tracked_file_sample>")[1].split("</tracked_file_sample>")[0]
    sampled = [line[2:] for line in sample.strip().splitlines()]
    # Head-truncation would return .agents/ entries exclusively -- the source
    # tree sorts after them and never made it into the prompt.
    assert any(path.startswith("apps/") for path in sampled)
    assert "10 of 47 tracked files" in prompt
    assert len(stub.calls) == 1 and stub.calls[0]["marker"] == "repo-scan"
    assert stub.calls[0]["read_only"] is True
    assert any(convention.name == "OpenAPI First" for convention in ctx.conventions)
    assert all(text not in prompt
        for text in ("pattern-scanner", "git diff", "affected_files", "relevant to the changes"))
    assert "no change set here" in prompt.lower()
    assert ctx.affected_files == []
    section = ctx.to_prompt_section()
    assert "Affected Files" not in section
    assert section.startswith(EXPLORATION_SECTION_PREFIX)
    recon_prompt = _build_recon_prompt(tmp_git_repo, [], [], section)
    assert recon_prompt.count(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY) == 1
    assert "OpenAPI First" in recon_prompt

def _pin_stack_availability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use an absent registry so ambient Beagle plugins cannot alter stack availability or group count."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config-absent"))

def _append_improve_config(target: Path, body: str) -> DaydreamFileConfig:
    """Commit repo config and load it through the same parser used before CLI RunConfig construction."""
    pyproject = target / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + body)
    git(target, "add", "pyproject.toml")
    commit(target, "configure improve partition bounds")
    return load_file_config(target)


@pytest.mark.anyio
async def test_audit_dispatch_interval_preserves_isolation_when_all_failed(
    improve_monorepo_target: Path, stub: ImproveStubBackend, make_config: MakeConfig,
) -> None:
    stub.fail_categories = set(AUDIT_CATEGORIES)
    stub.recon_output_override = ["not a recon object"]
    code = await _run_improve(make_config, improve_monorepo_target)

    assert code == 1
    # A malformed recon container is recorded and never stops the audit dispatch.
    recon = _load_improve_json(improve_monorepo_target, "recon.json")
    assert {"code": "RECON_CONTAINER_INVALID", "pointer": "/"} in recon["command_rejections"]
    assert recon["languages"] == []
    assert not improve_artifact(improve_monorepo_target, "report.md").exists()
    coverage = _load_improve_json(improve_monorepo_target, "coverage.json")
    assert coverage["not_audited"]
    assert coverage["partially_audited"] == []
    assert all(group["status"] == "failed" for group in coverage["groups"])
    assert set(coverage["failed_assignments"]) == {
        f"{category}:{group['name']}" for category in AUDIT_CATEGORIES for group in coverage["groups"]
    }
    audit = _load_improve_json(improve_monorepo_target, "audit-findings.json")
    assert audit["findings"] == [] and audit["failed"] == coverage["failed_assignments"]

    audit_cwds = {call["cwd"] for call in stub.calls if call["marker"] == "audit"}
    assert len(audit_cwds) == 1
    assert improve_monorepo_target not in audit_cwds
    assert all(cwd != improve_monorepo_target for cwd in audit_cwds)
    assert all(not cwd.exists() for cwd in audit_cwds)
    trajectory = _root_run_trajectory(improve_monorepo_target)
    dispatch = _dispatch_for_phase(trajectory, "audit")
    assert dispatch["extra"]["dispatch_status"] == "failed"
    assert dispatch["extra"]["reason_code"] == "all_children_failed"
    end = _phase_end(trajectory, "audit")
    assert end["status"] == "failed"
    assert end["reason_code"] == "all_children_failed"

def _not_audited_section(report: str) -> str:
    return report.split("## What was not audited")[1].split("## ")[0]

@pytest.mark.anyio
async def test_report_names_unaudited_partitions_and_failed_groups(
    improve_scaled_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = install_improve_stub(monkeypatch, improve_scaled_monorepo_target, n_findings=0)
    stub.fail_categories = {"docs"}
    code = await _run_improve(make_config, improve_scaled_monorepo_target, file_config=file_config)

    assert code == 0
    report = improve_artifact(improve_scaled_monorepo_target, "report.md").read_text()
    section = _not_audited_section(report)
    assert "**frontend**" in section and "group-ceiling" in section
    assert "`frontend/`" in section
    # The retained Python group was reached by seven categories, while docs
    # failed; the generic slice remains omitted by the ceiling for every lens.
    assert "**residue**" in section
    assert "not audited" in section
    assert "omitted: generic, python" in section
    assert "partially audited" in section
    failed = report.split("### Failed audit assignments")[1].split("## ")[0]
    assert "**docs / group-01**" in failed and "apps/svc00/" in failed
    coverage = _load_improve_json(improve_scaled_monorepo_target, "coverage.json")
    assert {entry["partition"] for entry in coverage["not_audited"]} == {"frontend"}
    assert coverage["not_audited"][0]["reason"] == "group-ceiling"
    partial = {entry["partition"]: entry for entry in coverage["partially_audited"]}
    assert set(partial) == {*(f"svc{index:02d}" for index in range(12)), "residue"}
    assert partial["residue"]["reason"] == "group-ceiling-and-failed"
    assert partial["residue"]["audited_file_count"] == 0
    assert partial["residue"]["reached_file_count"] == 1
    assert partial["residue"]["missing_by_category"] == {
        category: ["README.md", "pyproject.toml"] if category == "docs" else ["README.md"]
        for category in AUDIT_CATEGORIES
    }
    for index in range(12):
        entry = partial[f"svc{index:02d}"]
        paths = [f"apps/svc{index:02d}/api.py", f"apps/svc{index:02d}/pyproject.toml"]
        assert entry["missing_by_category"] == {
            category: paths if category == "docs" else [] for category in AUDIT_CATEGORIES
        }
        assert entry["audited_file_count"] == 0 and entry["reached_file_count"] == 2
    assert coverage["groups"] and coverage["partitions"]
    assert [entry["status"] for entry in coverage["groups"]] == ["partial"]
    assert coverage["groups"][0]["categories"] == {
        category: "failed" if category == "docs" else "audited" for category in AUDIT_CATEGORIES
    }
    assert "docs:group-01" in coverage["failed_assignments"]


@pytest.mark.anyio
async def test_vet_dispatch_interval_batch_failure_fails_closed_per_batch(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(improve_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.findings_per_category = 45
    stub.vet_reject_titles = {"Security finding 01"}
    stub.fail_vet_titles = {"Security finding 41"}  # the third batch's agent raises
    code = await _run_improve(make_config, improve_monorepo_target, improve_focus="security", file_config=file_config)

    assert code == 0
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert "Findings dropped without a usable vet verdict: 5" in report
    assert _load_improve_json(improve_monorepo_target, "audit-findings.json")["categories_run"] == ["security"]
    assert all(call["read_only"] for call in stub.calls if call["marker"] == "audit")
    vet_calls = [call for call in stub.calls if call["marker"] == "vet"]
    sizes = [len(json.loads(call["prompt"].split("```json\n")[1].split("```")[0])) for call in vet_calls]
    assert sorted(sizes) == [5, 20, 20] and all(size <= VET_BATCH_MAX_FINDINGS for size in sizes)
    vetted = _load_improve_json(improve_monorepo_target, "vetted-findings.json")
    members = [member for package in vetted["findings"] for member in package.get("members", [package])]
    assert len(members) == 39 and len(vetted["findings"]) == 8
    assert {finding["title"] for finding in members} == {f"Security finding {number:02d}" for number in range(2, 41)}
    rejected = json.loads((improve_monorepo_target / "daydream_plans/rejected.json").read_text())
    assert [entry["title"] for entry in rejected["rejected"]] == ["Security finding 01"]
    trajectory = _root_run_trajectory(improve_monorepo_target)
    dispatch = _dispatch_for_phase(trajectory, "vet")
    assert dispatch["extra"]["dispatch_status"] == "partial"
    assert dispatch["extra"]["reason_code"] == "some_children_failed"
    end = _phase_end(trajectory, "vet")
    assert end["status"] == "partial"
    assert end["reason_code"] == "some_children_failed"
    stub.fail_vet_titles = {f"Security finding {number:02d}" for number in range(1, 46)}
    code = await _run_improve(make_config, improve_monorepo_target, improve_focus="security", file_config=file_config)
    assert code == 1

def test_host_enumeration_failure_is_visible_and_keeps_model_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("daydream.improve.recon.enumerate_repository_commands", _raise_enumeration_failure,)
    model_commands = [{"id": "test-suite", "command": "uv run pytest", "working_directory": "."}]
    assert _host_enumerated_commands(tmp_path, [], [], model_commands=model_commands) == (
        0, [], ["HOST_COMMAND_ENUMERATION_FAILED@/host_commands"])
    assert "Host command enumeration failed" in capsys.readouterr().out
    assert model_commands == [{"id": "test-suite", "command": "uv run pytest", "working_directory": "."}]

_AUDIT_FINDING_SECRET = "sk-audit-title-secret123456"


class _AuditFindingCredentialBackend(ProductionPathBackend):
    """Append a credential to one audit finding body."""

    async def _pre_execute(self, cwd: Path, prompt: str, **turn_kwargs: Any) -> list[AgentEvent] | None:
        events = await super()._pre_execute(cwd, prompt, **turn_kwargs)
        if events and "read-only improve audit specialist" in prompt:
            output = cast(ResultEvent, events[0]).structured_output
            for finding in cast(dict[str, Any], output)["findings"]:
                if finding["title"] == "Production finding 05":
                    finding["body"] += f" OPENAI_API_KEY={_AUDIT_FINDING_SECRET}"
        return events


@pytest.mark.anyio
async def test_capable_improve_partial_plan_failure_is_safe_and_publishes_the_rest(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    """One blocked plan keeps planning successful, stays secret-free, and publishes only the landed plan."""
    _force_interactive(monkeypatch)
    answers: list[str] = []

    def _answer(*_args: Any, **_kwargs: Any) -> str:
        answers.append("2,1-2" if answers else "two")
        return answers[-1]

    monkeypatch.setattr("daydream.run_context._prompt_user", _answer)
    backend = _AuditFindingCredentialBackend(improve_monorepo_target, failed_title="Production finding 03",)
    install_capable_improve_backend(monkeypatch, backend)
    _no_open_issues(monkeypatch)
    created: list[str] = []

    def _create_issue(*args: Any, **kwargs: Any) -> str:
        created.append(str(kwargs["title"]))
        return "https://github.com/acme/widgets/issues/1"

    monkeypatch.setattr("daydream.git_ops.gh_issue_create", _create_issue)

    code = await _run_publish(make_config, improve_monorepo_target, non_interactive=False)
    selected = _load_improve_json(improve_monorepo_target, "selected.json")
    publication = _load_improve_json(improve_monorepo_target, "published-issues.json",)
    vetted = _load_improve_json(improve_monorepo_target, "vetted-findings.json")["findings"]
    assert answers == ["two", "2,1-2"]
    assert selected["mode"] == "interactive"
    assert selected["selected"] == [vetted[1]["fingerprint"], vetted[0]["fingerprint"]]
    assert code == 2
    assert publication["status"] == "partial"
    assert len(publication["published"]) == len(created) == 1
    assert [(entry["stage"], entry["plan_path"]) for entry in publication["failed"]] == [("plan-write", None)]
    assert len(publication["published"]) + len(publication["failed"]) == len(selected["selected"])
    recon = _load_improve_json(improve_monorepo_target, "recon.json")
    plan_files = sorted((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    diagnostics_text = improve_artifact(improve_monorepo_target, "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    diagnostics = json.loads(diagnostics_text)
    failed = [attempt for attempt in diagnostics["attempts"] if attempt["disposition"] == "blocked"]
    console_output = capsys.readouterr().out
    observables = [console_output, *_improve_observable_texts(improve_monorepo_target),]

    assert recon["commands"]
    assert recon["command_rejections"] == [{"code": "RECON_MALFORMED_COMMAND", "pointer": "/commands/42/command",}]
    assert len(plan_files) == 1
    assert len(failed) == 1
    assert failed[0]["finding"]["title"] == "Production finding 01"
    assert failed[0]["errors"] == [{"code": "PROCESS_EXIT", "pointer": "/"}]
    assert "Plan blocked for Production finding 01: PROCESS_EXIT at /." in console_output
    report = improve_artifact(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    assert "Plans written: 1" in report
    assert "Plans blocked by plan-writing failure: 1" in report
    assert all(backend.recon_secret not in text for text in observables)
    assert all(backend.planner_secret not in text for text in observables)
    # An audit finding carrying a credential is redacted before any artifact, plan or report.
    assert all(_AUDIT_FINDING_SECRET not in text for text in observables)
    assert "coverage 05. OPENAI_API_KEY=[REDACTED" in "\n".join(observables)

@pytest.mark.anyio
async def test_branch_focus_manifest_retains_deleted_paths_without_reading_missing_files(
    improve_branch_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_config: MakeConfig,
) -> None:
    repo = improve_branch_target
    deleted, binary = "web/App.tsx", "web/logo.png"
    regular, linked, gitlink = ("apps/billing/link-to-regular.py", "apps/billing/regular-to-link.py",
        "apps/billing/regular-to-gitlink.py")
    target = "apps/billing/replacement-target.py"
    (repo / target).write_text("NEW_TARGET_CONTENT = True\n")
    (repo / regular).symlink_to("replacement-target.py")
    (repo / linked).write_text("OLD_LINK_CONTENT = True\n")
    (repo / gitlink).write_text("OLD_GITLINK_CONTENT = True\n")
    (repo / binary).write_bytes(b"\x89PNG\x00binary payload")
    git(repo, "add", binary, regular, linked, gitlink, target)
    commit(repo, "seed three tracked type replacements")
    git(repo, "branch", "-f", "main", "HEAD")
    (repo / regular).unlink()
    (repo / regular).write_text("SURVIVING_REGULAR_CONTENT = 12345\n")
    (repo / linked).unlink()
    (repo / linked).symlink_to("replacement-target.py")
    (repo / gitlink).unlink()
    nested = tmp_path / "replacement-origin"
    init_repo(nested)
    (nested / "nested.py").write_text("NESTED_CONTENT_NOT_CAPTURED = True\n")
    git(nested, "add", "nested.py")
    nested_commit = commit(nested, "seed replacement submodule")
    git(repo, "rm", "--cached", "--", gitlink)
    git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(nested), gitlink)
    reference = {"path": gitlink, "commit": nested_commit}
    (repo / "apps/billing/api.py").write_text('def service_name():\n    return "billing-v3"\n')
    git(repo, "add", "apps/billing/api.py", regular, linked, gitlink)
    git(repo, "rm", deleted, binary)
    commit(repo, "replace three tracked types and delete frontend files")
    source_bytes = {path: (repo / path).read_bytes() for path in (regular, linked, f"{gitlink}/nested.py")}
    before = (head_sha(repo), _git_status_porcelain(repo), git(repo, "ls-files", "--stage"))
    backend = install_capable_improve_backend(monkeypatch, ManifestReadingBackend(repo))

    assert await _run_improve(make_config, repo, improve_focus="branch") == 0

    assert len(backend.assignment_reads) == len(AUDIT_CATEGORIES)
    ordinary = [".gitmodules", "apps/billing/api.py", regular, linked]
    for assignment in backend.assignment_reads:
        assert assignment["primary_files"] == ordinary
        assert assignment["deleted_files"] == [deleted, binary]
        assert assignment["submodule_references"] == [reference]
        assert assignment["primary_sha256"][regular] == hashlib.sha256(source_bytes[regular]).hexdigest()
        assert assignment["primary_sha256"][linked] == hashlib.sha256(source_bytes[linked]).hexdigest()
    coverage = _load_improve_json(repo, "coverage.json")
    assert coverage["groups"][0]["primary_files"] == sorted(ordinary + [deleted, binary, gitlink])
    assert coverage["groups"][0]["file_count"] == 7
    assert coverage["groups"][0]["submodule_reference_count"] == 1
    assert coverage["groups"][0]["status"] == "audited"
    assert coverage["not_audited"] == coverage["partially_audited"] == []
    assert (head_sha(repo), _git_status_porcelain(repo), git(repo, "ls-files", "--stage")) == before
    assert {path: (repo / path).read_bytes() for path in source_bytes} == source_bytes
    assert not (repo / regular).is_symlink() and (repo / linked).is_symlink()


@pytest.mark.anyio
async def test_branch_fanout_preserves_serial_limits_and_quick_filtering(
    improve_branch_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """One category group still obeys each independently selected serial ceiling."""
    backend_capacity = 3
    expected = min(EFFORT_TIERS["quick"].max_concurrency, backend_capacity)
    backend = install_capable_improve_backend(monkeypatch,
        RendezvousImproveBackend(improve_branch_target, capacity=backend_capacity, expected_concurrency=expected,
            n_findings=None),
    )
    backend.findings_per_category = 3
    backend.low_confidence_tail = True
    with anyio.fail_after(15):
        code = await _run_improve(make_config, improve_branch_target, improve_focus="branch", improve_effort="quick")
        assert code == 0
    assert backend.max_active == {"audit": expected, "vet": expected}
    audited = _load_improve_json(improve_branch_target, "audit-findings.json")
    assert audited["categories_run"] == list(AUDIT_CATEGORIES)
    assert len(audited["findings"]) == 6
    assert {finding["confidence"] for finding in audited["findings"]} == {"HIGH"}
    report = improve_artifact(improve_branch_target, "report.md").read_text()
    assert "Non-HIGH-confidence findings dropped by tier: 8" in report
    assert "Lowest-leverage findings dropped by tier cap: 10" in report


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["audit", "vet"])
async def test_branch_cancellation_joins_provider_streams_before_snapshot_cleanup(
    improve_branch_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, phase: str,
) -> None:
    """Cancellation with running and queued siblings must close the real run evidence."""
    repo = improve_branch_target
    backend = install_capable_improve_backend(monkeypatch,
        RendezvousImproveBackend(repo, capacity=2, expected_concurrency=2, block_phase=phase,
            n_findings=3 if phase == "vet" else 2),
    )
    before = (head_sha(repo), git(repo, "show-ref"), git(repo, "ls-files", "--stage"), _git_status_porcelain(repo))
    with anyio.fail_after(15):
        async with anyio.create_task_group() as tasks:
            async def run_branch() -> None:
                await _run_improve(make_config, repo, improve_focus="branch")
            tasks.start_soon(run_branch)
            await backend.blocked_ready.wait()
            tasks.cancel_scope.cancel()
    assert backend.max_active[phase] == 2
    assert backend.closed[phase] == 2
    assert backend.closed_before_cleanup[phase] == [True, True]
    assert backend.active == {"audit": 0, "vet": 0}
    assert backend.audit_root is not None and not backend.audit_root.exists()
    assert (head_sha(repo), git(repo, "show-ref"), git(repo, "ls-files", "--stage"),
        _git_status_porcelain(repo)) == before
    assert not improve_artifact(repo, "report.md").exists()
    trajectory = _root_run_trajectory(repo)
    dispatch = _dispatch_for_phase(trajectory, phase)
    assert dispatch["extra"]["dispatch_status"] == "cancelled"
    assert dispatch["extra"]["attempted_count"] == dispatch["extra"]["completed_count"] == 2
    assert dispatch["extra"]["planned_count"] > 2
    assert _phase_end(trajectory, phase)["status"] == "cancelled"
    # Exact closed child intervals and refs, including cancellation, are durable.
    descriptors = ([f"audit-{category}-group-01" for category in AUDIT_CATEGORIES[:2]]
        if phase == "audit" else ["vet-security-00", "vet-correctness-01"])
    assert [child["content"] for child in dispatch["observation"]["results"]] == [
        f"Dispatched to {descriptor}" for descriptor in descriptors]
    assert dispatch_encloses_children(dispatch, repo)


@pytest.mark.anyio
async def test_branch_focus_on_base_branch_reports_and_exits_cleanly(
    improve_monorepo_target: Path, stub: ImproveStubBackend, make_config: MakeConfig,
) -> None:
    code = await _run_improve(make_config, improve_monorepo_target, improve_focus="branch")
    assert code == 1


@pytest.mark.anyio
async def test_group_scope_slices_search_but_report_names_the_unaudited_rest(
    improve_monorepo_target: Path, stub: ImproveStubBackend, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named group audits every member, names the rest unaudited, and a phase table retunes only its phase."""
    ledger = improve_monorepo_target / "apps/ledger"
    ledger.mkdir()
    (ledger / "pyproject.toml").write_text('[project]\nname = "ledger"\n')
    (ledger / "api.py").write_text('def service_name():\n    return "ledger"\n')
    git(improve_monorepo_target, "add", "apps/ledger")
    commit(improve_monorepo_target, "add ledger service")
    _record_phase_tiers(monkeypatch)
    code = await _run_improve(make_config, improve_monorepo_target, improve_scope="core",
        file_config=DaydreamFileConfig(improve_service_groups={"core": ["apps/billing", "apps/ledger"]},
            phases={"plan_write": {"model": "file-plan-model", "reasoning_effort": "medium"}}),
    )
    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert audit_calls and all("apps/billing" in call["prompt"] and "apps/ledger" in call["prompt"]
        for call in audit_calls)
    assert not any("apps/catalog" in call["prompt"] for call in audit_calls)
    tiers = _tiers_by_marker(stub.calls)
    assert tiers["plan-writer"] == {("file-plan-model", "medium")}
    assert tiers["recon"] == {(PHASE_DEFAULT_MODELS["claude"]["recon"], "low")}
    # A slice bounds where the audit searches, never what it may read: a
    # cross-service finding must stay reachable (spec.md monorepo requirement).
    assert all("bounds where you search, never what you may read" in call["prompt"] for call in audit_calls)
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert "limited to `core`" in report
    assert "**catalog** — `apps/catalog`" in report

async def _run_improve(make_config: MakeConfig, repo: Path, **overrides: Any) -> int:
    return await run(make_config(repo, flow_name="improve", **overrides))

async def _run_publish(make_config: MakeConfig, repo: Path, **overrides: Any) -> int:
    return await _run_improve(
        make_config, repo, pr_repo="acme/widgets", file_config=DaydreamFileConfig(improve_github_publish_issues=True),
        **overrides,
    )

async def _run_plan_subverb(make_config: MakeConfig, repo: Path) -> int:
    return await _run_improve(make_config, repo, improve_plan_description="add rate limiting")

@pytest.mark.anyio
async def test_plan_subverb_repairs_invalid_plan_once_then_blocks_without_truncating(
    improve_monorepo_target: Path, stub: ImproveStubBackend, make_config: MakeConfig,
) -> None:
    """A secret-bearing invalid plan gets one sanitized repair; a still over-length plan blocks whole."""
    stub.return_secret_invalid_enum_once = True
    stub.plan_instruction_override = "Replace service_name. " + "x" * 4000
    code = await _run_plan_subverb(make_config, improve_monorepo_target)
    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 1
    assert len(plan_calls) == 2
    repair_prompt = plan_calls[1]["prompt"]
    assert "PRIVATE_SCHEMA_SECRET" not in repair_prompt
    assert "TOKEN=" not in repair_prompt
    assert all(
        call["output_schema"] == PLAN_AUTHOR_SCHEMA and call["read_only"] is True and call["persist_session"] is False
        for call in plan_calls
    )
    diagnostics = _plan_write_diagnostics(improve_monorepo_target)
    errors = [error for attempt in diagnostics["attempts"] for error in attempt["errors"]]
    assert any(error["pointer"] == "/steps/0/changes/0/instruction" for error in errors), errors
    assert not list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (improve_monorepo_target / ".daydream", improve_monorepo_target / "daydream_plans",)
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "PRIVATE_SCHEMA_SECRET" not in persisted
    assert "TOKEN=" not in persisted

@pytest.mark.anyio
async def test_an_edited_file_left_unquoted_is_repaired_before_the_plan_lands(
    improve_monorepo_target: Path, stub: ImproveStubBackend, make_config: MakeConfig,
) -> None:
    """Reject and repair missing quotes before publishing an executable drift comparison.

    A non-retryable writer crash replaces only that generation; it spends no repair.
    """
    stub.plan_crash_attempts = 1
    stub.plan_unquoted_path_attempts = 2
    code = await _run_plan_subverb(make_config, improve_monorepo_target)
    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 0
    assert len(plan_calls) == 3
    assert "EXISTING_PATH_NOT_QUOTED" not in plan_calls[1]["prompt"]
    repair_prompt = plan_calls[2]["prompt"]
    assert "EXISTING_PATH_NOT_QUOTED" in repair_prompt
    assert "/scope/existing_paths/0/path" in repair_prompt
    assert "context_excerpts" in repair_prompt
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    current_state = plan_text.split("## Current state\n\n", 1)[1].split("\n\n## Commands")[0]
    assert "- `apps/billing/api.py:1-2`" in current_state
    assert "def service_name" in current_state
    diagnostics = _plan_write_diagnostics(improve_monorepo_target)
    assert [attempt["disposition"] for attempt in diagnostics["attempts"]] == ["retried", "success"]
    first_attempt = diagnostics["attempts"][0]
    assert first_attempt["stage"] == "authoring"
    assert {(error["code"], error["pointer"]) for error in first_attempt["errors"]} == {
        ("EXISTING_PATH_NOT_QUOTED", "/scope/existing_paths/0/path")
    }

@pytest.mark.anyio
async def test_retryable_plan_writer_failure_is_not_regenerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_agent already spent the retry budget on a retryable failure; authoring must not restart it."""
    calls: list[str] = []

    class _RateLimited(RuntimeError):
        retryable = True

    async def _generate_once(_ctx: Any, _backend: Any, prompt: str) -> tuple[Any, str | None]:
        calls.append(prompt)
        raise _RateLimited("provider rate limit")

    monkeypatch.setattr(authoring, "_generate_once", _generate_once)
    with pytest.raises(_RateLimited):
        await authoring._generate_with_crash_retry(cast(FlowContext, None), cast(Backend, None), "plan")
    assert calls == ["plan"]


def test_apply_vet_verdicts_restamps_a_corrected_location(tmp_path: Path) -> None:
    for root, source in (("apps/billing", "first\n" * 10), ("apps/catalog", "first\n" * 20), ("frontend", "x = 1\n")):
        (tmp_path / root).mkdir(parents=True)
        (tmp_path / root / "api.py").write_text(source)
    services = [Service("billing", Path("apps/billing"), "test"), Service("catalog", Path("apps/catalog"), "test"),
        Service("frontend", Path("frontend"), "test")]
    partitions = [Partition("billing", "apps/billing", "test", "billing", ()),
        Partition("catalog", "apps/catalog", "test", "catalog", ()),
        Partition("frontend", "frontend", "directory", None, ())]
    findings: list[dict[str, Any]] = [
        {"fingerprint": "original", "title": "Misattributed finding", "category": "correctness",
            "path": "apps/billing/api.py", "line": 10, "body": "The evidence belongs to catalog.",
            "evidence": ["`apps/billing/api.py:10`"]},
        {"fingerprint": "frontend", "title": "Frontend finding", "category": "correctness", "severity": "high",
            "path": "./frontend/api.py", "line": 1, "evidence": ["./frontend/api.py:1"], "provenance": "inherited"},
        {"fingerprint": "unknown", "title": "Unknown severity", "severity": "CRITICAL"},
    ]
    kept, rejected = _apply_vet_verdicts(findings, [
        {"vet_id": 1, "keep": True, "path": "apps/catalog/api.py", "line": 20, "severity": "medium"},
        {"vet_id": 2, "keep": True, "path": "./frontend/api.py", "line": 1}, {"vet_id": 3, "keep": True},
    ], rejected_at_sha="sha", repo=tmp_path, default_provenance="introduced", services=services,
        partitions=partitions)
    assert rejected == []
    assert [(finding["evidence"], finding["services"], finding["partition"], finding["severity"],
        finding["provenance"]) for finding in kept[:2]] == [
        (["`apps/catalog/api.py:20`"], ["catalog"], "catalog", "MED", "introduced"),
        (["./frontend/api.py:1"], ["frontend"], "frontend", "HIGH", "inherited"),
    ]
    assert len(kept) == 3 and kept[0]["fingerprint"] != "original" and kept[1]["fingerprint"] != "frontend"
    assert kept[2] == {"fingerprint": "unknown", "title": "Unknown severity", "provenance": "introduced"}


def test_stamp_finding_rejects_unconfined_missing_and_out_of_range_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "valid.py").write_text("first\nsecond\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("secret\n")
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    for citation in ("missing.py:1", "valid.py:2:3", "escape/secret.py:1"):
        assert _stamp_finding({"evidence": [citation]}, "correctness", [], [], repo=repo,) is None, citation

def _nested_service_repo(tmp_path: Path) -> Path:
    """A repo whose ``services/api/inner`` nests inside ``services/api``."""
    repo = tmp_path / "repo"
    for root in ("services/api", "services/api/inner"):
        (repo / root).mkdir(parents=True)
        (repo / root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return repo

def test_scope_narrowing_does_not_widen_for_a_repo_root_service(tmp_path: Path) -> None:
    repo = _nested_service_repo(tmp_path)
    services = enumerate_services(
        repo, DaydreamFileConfig(improve_service_roots=["services/*", "services/api/inner", "."])
    )
    diff = (
        "diff --git a/services/api/inner/main.py b/services/api/inner/main.py\n"
        "--- a/services/api/inner/main.py\n"
        "+++ b/services/api/inner/main.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"
    )

    text, files = _restrict_diff_to_services(diff, services)
    assert files == ["services/api/inner/main.py"]
    assert text.count("diff --git") == 1

    scoped = _stacks_for_services(
        [StackAssignment("python", ["services/api/inner/main.py", "README.md"], False)], services
    )
    assert [(stack.stack_name, list(stack.files)) for stack in scoped] == [("python", ["services/api/inner/main.py"])]

@pytest.mark.anyio
async def test_publication_failures_are_not_reported_as_planning_failures(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    """Issue-create and pathless-plan failures are publication failures, never planning failures."""
    repo = improve_monorepo_target
    install_improve_stub(monkeypatch, repo, n_findings=1,)
    _no_open_issues(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_create", lambda *args, **kwargs: (_ for _ in ()).throw(GitError("offline")),
    )

    code = await _run_publish(make_config, repo)
    publication = _load_improve_json(repo, "published-issues.json",)
    output = capsys.readouterr().out
    report = improve_artifact(repo, "report.md",).read_text(encoding="utf-8")
    assert code == 1
    assert publication["status"] == "failed"
    assert publication["published"] == []
    assert [entry["stage"] for entry in publication["failed"]] == ["issue-create"]
    assert "Improve issue publishing failed" in output
    assert "Improve planning failed" not in output
    assert "GitHub publication failures: 1" in report

    # An indexed package whose plan file vanished fails before GitHub reconciliation.
    next((repo / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")).unlink()
    monkeypatch.setattr("daydream.git_ops.gh_issue_list_strict",
        lambda *args, **kwargs: pytest.fail("a pathless package must fail before GitHub reconciliation"),
    )
    code = await _run_publish(make_config, repo)
    publication = _load_improve_json(repo, "published-issues.json",)
    output = capsys.readouterr().out
    assert code == 1
    assert publication["status"] == "failed"
    assert publication["published"] == []
    assert [(entry["stage"], entry["plan_path"]) for entry in publication["failed"]] == [
        ("plan-reconciliation", None)]
    assert "Improve issue publishing failed" in output
    assert "Improve planning failed" not in output
