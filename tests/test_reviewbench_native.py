"""Native ReviewBench contracts using real Git and the ordinary runner."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, CostEvent, EffectiveRequestConfig, RequestEvent
from daydream.benchmark.reviewbench.candidate import selected_extension_identity
from daydream.benchmark.reviewbench.contract import Case
from daydream.benchmark.reviewbench.native import convert_findings, native_configuration, review_case
from daydream.config_file import DaydreamFileConfig
from daydream.deep.artifacts import diff_key
from daydream.review_result import FinalizedReview, deliver_finalized_review
from daydream.run_context import InteractionPolicy, RunContext, bind_run_context
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.dataset_hub import FakeDatasetHub
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, init_repo, tracked_source_state, write_and_stage
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.review_result import terminal_result
from tests.harness.stub_backend import review_stage_state, stage_result


def frozen_case(tmp_path: Path) -> tuple[Path, Case]:
    repo = tmp_path / "frozen-input"
    init_repo(repo)
    write_and_stage(repo, "README.md", "Synthetic native ReviewBench integration fixture.\n")
    base = commit(repo, "base")
    git(repo, "checkout", "-b", "candidate")
    write_and_stage(repo, "api.py", "def resolve(value):\n    return value[0]\n")
    head = commit(repo, "candidate")
    git(repo, "remote", "add", "origin", "https://github.com/example/project.git")
    return repo, Case("https://github.com/example/project", 7, base, head)


def finding(**overrides: Any) -> dict[str, Any]:
    return {"id": 1, "file": "api.py", "line": 2, "description": "Empty input raises IndexError",
            "rationale": "The API accepts an empty sequence.", "evidence": "api.py:2 indexes value[0].",
            "severity": "medium", "confidence": "MEDIUM", **overrides}


@pytest.mark.parametrize(("outcome", "status"), [
    ("empty", "completed_empty"), ("finding", "complete"), ("partial", "incomplete"),
    ("failed", "failed"), ("relocated", "conversion_limited"), ("pipeline_failure", "incomplete"),
])
async def test_native_attempt_uses_terminal_authority_and_preserves_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, outcome: str, status: str,
    dataset_store: Path, dataset_hub: FakeDatasetHub, operator_dataset_sentinel: Path,
) -> None:
    """Frozen native attempts preserve terminal authority, isolation and available usage."""
    repo, case = frozen_case(tmp_path)
    original = tracked_source_state(repo)
    roots: list[Path] = []

    @dataclass(frozen=True)
    class CustomConfig(EffectiveRequestConfig):
        arbitrary: str = "custom-unstructured-canary"

    class MeasuredBackend(EmptyReviewBackend):
        async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            yield RequestEvent(prompt="Unpersisted request prompt", config=CustomConfig())
            yield CostEvent(cost_usd=0.0, input_tokens=None, output_tokens=0, cached_tokens=None)
            async for event in super().execute(*args, **kwargs):
                yield event

    nonempty = outcome in {"finding", "partial", "relocated", "pipeline_failure"}

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == "pi"
        checkout = kwargs["cwd"]
        assert checkout != repo and not git_ops.remote_urls(checkout)
        roots.append(checkout)
        backend_type = EmptyReviewBackend if outcome == "empty" else MeasuredBackend
        backend = backend_type(checkout, forbid_merge=False, forbid_supervise=False,
            review_by_stack={"structure": [finding(line=500 if outcome == "relocated" else 2)]} if nonempty else {},
            fail_stack="python" if outcome == "partial" else None,
            responder=(lambda prompt: [RuntimeError("offline failure")] if (
                outcome == "failed" and ("you are reviewing the " in prompt.lower()
                                         or "you are the structural reviewer" in prompt.lower())
                or outcome == "pipeline_failure" and "supervisor adjudication" in prompt.lower()) else None))
        backend.merge_echo_records = True
        return backend

    monkeypatch.setattr(runner, "create_backend", create)
    config = empty_review_config(repo, tmp_path / "ignored-trajectory.json", backend="pi",
        model="deepseek/deepseek-v4.1-flash", dataset_capture=True, output_mode="loop",
        dataset_store_path=operator_dataset_sentinel, trajectory_hub_repo="fixture/forbidden-benchmark-capture",
        findings_out=str(tmp_path / "must-not-publish.json"))
    result = await review_case(case, repo, config, "native-fixture")
    assert result.status == status, result.diagnostics
    assert result.terminal_result is not None and result.terminal_result["run_id"]
    assert result.analyzed_comparison == {"head_sha": case.head, "merge_base_sha": case.base,
        "diff_key": diff_key(git_ops.diff_paths(repo, case.base, case.head, ["."]))}
    assert tracked_source_state(repo) == original and not (repo / ".daydream").exists()
    assert roots and all(not root.exists() for root in roots)
    assert not (tmp_path / "must-not-publish.json").exists()
    assert not (tmp_path / "ignored-trajectory.json").exists()
    assert not dataset_store.exists() and dataset_hub.commits == []
    assert not fake_gh.calls("POST")
    assert result.configuration["phases"]["per_stack_review"]["model"] == config.model
    assert result.usage == ({} if outcome == "empty" else {"completion_tokens": 0, "cost_usd": 0.0})
    if outcome != "empty":
        assert result.configuration["observed_requests"]
        assert all(row["controls"] == {} for row in result.configuration["observed_requests"])
    assert "custom-unstructured-canary" not in json.dumps(asdict(result))
    if status in {"failed", "incomplete"}:
        assert result.candidate is None
    else:
        assert result.candidate is not None
        assert result.candidate["pr"] == asdict(case)
        assert bool(result.candidate["findings"]) == (outcome == "finding")
        if outcome == "empty":
            assert "usage" not in result.candidate
        else:
            assert result.candidate["usage"] == result.usage
    if nonempty:
        assert result.retained_findings[0]["description"] == finding()["description"]
    if outcome == "relocated":
        assert result.conversion_omissions[0]["reason"] == "native_location_was_relocated_or_distrusted"
    if outcome == "finding":
        # Optional external proof uses the pinned upstream's actual no-inference
        # parser on this real native output, not a handcrafted candidate envelope.
        if source := os.environ.get("DAYDREAM_REVIEWBENCH_SOURCE"):
            candidate, metadata = tmp_path / "produced.json", tmp_path / "pr.json"
            candidate.write_text(json.dumps(result.candidate))
            metadata.write_text(json.dumps(asdict(case)))
            checked = subprocess.run(["npm", "run", "harbor:validate-findings", "--", "--candidate", str(candidate),
                                      "--pr", str(metadata)], cwd=source, capture_output=True, text=True, timeout=60)
            assert checked.returncode == 0, checked.stdout + checked.stderr
            assert "No judging or inference performed" in checked.stdout


@pytest.mark.parametrize("fault", ["dirty", "identity", "missing", "mirror", "no_origin"])
async def test_frozen_input_is_verified_before_native_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """Reject dirty, mismatched or unavailable frozen sources before dispatch."""
    repo, case = frozen_case(tmp_path)
    if fault == "dirty":
        (repo / "untracked.py").write_text("not part of the frozen case")
    elif fault == "identity":
        git(repo, "remote", "set-url", "origin", "https://github.com/another/project.git")
    elif fault == "missing":
        case = replace(case, head="f" * 40)
    elif fault == "no_origin":
        git(repo, "remote", "remove", "origin")
    else:
        git(repo, "remote", "set-url", "origin", "https://github.com/review-bench/example_project.git")

    def create(*_args: Any, **kwargs: Any) -> EmptyReviewBackend:
        assert fault == "mirror", "invalid frozen inputs reached backend creation"
        return EmptyReviewBackend(kwargs["cwd"])

    monkeypatch.setattr(runner, "create_backend", create)
    result = await review_case(case, repo, empty_review_config(repo, tmp_path / "trajectory", backend="pi"), "fixture")
    assert result.status == ("completed_empty" if fault == "mirror" else "failed"), result.diagnostics
    if fault != "mirror":
        assert result.candidate is None and result.terminal_result is None and result.diagnostics


def test_head_conversion_retains_logical_records_and_exposes_invalid_locations(tmp_path: Path) -> None:
    """Convert proven HEAD locations and retain explicit omissions for unsafe hints."""
    repo, initial = frozen_case(tmp_path)
    # A later frozen PR includes a same-number replacement, unchanged context,
    # a pure-deleted file, and new-only lines beyond the old hunk's bounds.
    write_and_stage(repo, "deleted.py", "removed\n")
    base = commit(repo, "second base")
    write_and_stage(repo, "api.py", "def resolve(value):\n    return value[-1]\n# new\n# tail\n")
    git(repo, "rm", "deleted.py")
    head = commit(repo, "second candidate")
    case = replace(initial, base=base, head=head)
    items = [finding(id=1, line=1), finding(id=2, line=2), finding(id=3, line=3),
        finding(id=4, file="deleted.py", line=1), finding(id=5, line=0), finding(id=6, line=999),
        finding(id=7, line=3, location_cited_line=2), finding(id=8, file="../escape", line=1),
        finding(id=9, line=4, description="", rationale="", evidence="")]
    converted, omitted = convert_findings(items, repository=repo, case=case, producer="fixture",
        comparison=git_ops.diff_paths(repo, base, head, ["."]))
    assert [(row["start_line"], row["end_line"]) for row in converted] == [(1, 1), (2, 2), (3, 3)]
    assert all(row["producer"] == "fixture" and row["file"] == "api.py" for row in converted)
    assert {row["id"]: row["reason"] for row in omitted} == {
        4: "location_absent_from_head_regular_file",
        5: "missing_repository_relative_positive_location", 6: "location_absent_from_head_regular_file",
        7: "native_location_was_relocated_or_distrusted", 8: "missing_repository_relative_positive_location",
        9: "missing_finding_message"}


async def _packaged_extension_journey(repo: Path, case: Case, extension: Path, mutation: str, report: Path) -> None:
    """Run inside the isolated candidate import root, replacing only generation."""
    created: list[tuple[str, EmptyReviewBackend]] = []
    config = empty_review_config(repo, report.parent / 'trajectory', file_config=DaydreamFileConfig(
        backend='pi', supervisor='llm', phases={'extension-review': {'backend': 'claude', 'model': 'claude-sonnet-5'}}))
    expected_extension = selected_extension_identity()
    frozen = native_configuration(config)
    if mutation == 'before':
        extension.write_text(extension.read_text() + '\n# changed before generation\n')

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert kwargs['execution_input'].retry_policy.base_delay_s == (10.0 if name == 'pi' else 2.0)
        if mutation == 'during' and not created:
            extension.write_text(extension.read_text() + '\n# changed during the admitted run\n')
        backend = EmptyReviewBackend(kwargs.get('cwd', repo), forbid_merge=False, forbid_supervise=False,
            review_by_stack={'structure': [finding()]} if mutation == 'during' else {})
        backend.model = kwargs['model']
        backend.merge_echo_records = True
        created.append((name, backend))
        return backend

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, 'create_backend', create)
        result = await review_case(case, repo, config, 'fixture', runtime=frozen)
    report.write_text(json.dumps({'execution': asdict(result), 'frozen_extension': expected_extension,
        'created': len(created), 'custom_phase_called': any(name == 'claude' and backend.calls
            for name, backend in created), 'prompt_override_called': any('PACKAGED_PROMPT_OVERRIDE' in call['prompt']
            for _, backend in created for call in backend.calls)}))


@pytest.mark.parametrize('mutation', ['none', 'before', 'during'])
async def test_native_packaged_extension_is_bound_before_and_after_real_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, mutation: str,
) -> None:
    """Keep ordinary packaged customization and reject changed executable inputs.

    The earlier ExtDir phase test could prove backend policy while allowing a
    mutable, untracked extension through both benchmark identity checks. Run an
    actual owned default extension in an isolated copy of the candidate, with real
    Git and runner execution, then edit its real source before/during admission.
    """
    import daydream

    repo, case = frozen_case(tmp_path)
    monkeypatch.delenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', raising=False)
    monkeypatch.delenv('DAYDREAM_EXT_DIR', raising=False)
    candidate = tmp_path / 'candidate'
    shutil.copytree(Path(daydream.__file__).parent, candidate / 'daydream',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    extension = candidate / 'daydream_ext' / '__init__.py'
    extension.parent.mkdir()
    extension.write_text(
        'DAYDREAM_EXT_API = 8\n'
        'from daydream.agent import run_agent\n'
        'from daydream.extensions import FlowStep\n'
        'from daydream.trajectory import DaydreamPhase\n'
        'async def probe(ctx):\n'
        '    await run_agent(ctx.backend_for("extension-review"), ctx.work.repo, "NATIVE_EXTENSION", '
        'phase=DaydreamPhase.REVIEW, run_context=ctx.run_context)\n'
        'def register(registry):\n'
        '    intent = registry.prompt("intent")\n'
        '    registry.override_prompt("intent", lambda *args, **kwargs: '
        'intent(*args, **kwargs) + "\\nPACKAGED_PROMPT_OVERRIDE")\n'
        '    registry.register_phase(FlowStep(name="native-probe", run=probe))\n'
        '    registry.insert_before("deep", anchor="intent", step="native-probe")\n')
    home = tmp_path / 'private-home'
    home.mkdir()
    report = tmp_path / 'extension-result.json'
    script = (
        'import asyncio,json,sys; from pathlib import Path; '
        'from tests.test_reviewbench_native import _packaged_extension_journey; '
        'from daydream.benchmark.reviewbench.contract import Case; '
        'asyncio.run(_packaged_extension_journey(Path(sys.argv[1]), Case.from_dict(json.loads(sys.argv[2])), '
        'Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])))'
    )
    subprocess.run([sys.executable, '-c', script, str(repo), json.dumps(asdict(case)), str(extension), mutation,
                    str(report)], cwd=tmp_path, env={**os.environ, 'HOME': str(home),
        'PYTHONPATH': os.pathsep.join((str(candidate), str(Path(__file__).resolve().parent.parent)))},
        check=True, capture_output=True, text=True, timeout=60)
    observed = json.loads(report.read_text())
    result = observed['execution']
    assert observed['frozen_extension']['source_kind'] == 'packaged'
    if mutation == 'none':
        assert result['status'] == 'completed_empty', result['diagnostics']
        assert result['candidate'] is not None
    else:
        assert result['status'] == 'failed' and result['candidate'] is None
    if mutation == 'before':
        assert observed['created'] == 0 and result['terminal_result'] is None
        assert 'native extension differs from frozen configuration' in result['diagnostics'][0]
    else:
        assert observed['custom_phase_called'] and observed['prompt_override_called']
        assert result['configuration']['phases']['extension-review']['model'] == 'claude-sonnet-5'
        assert result['configuration']['native']['extension'] == observed['frozen_extension']
    if mutation == 'during':
        assert result['terminal_result']['analysis_state'] == 'complete'
        assert result['retained_findings']
        assert 'native extension changed during review' in result['diagnostics'][0]
    assert not fake_gh.calls('POST')


async def test_native_explicit_runtime_cannot_admit_external_extension_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public callers supplying capabilities must still pass host extension admission.

    Native configuration is bypassed by an explicit runtime tuple; this boundary
    must reject an external selector before scanning arbitrary operator files or
    invoking its register function, even when backend capabilities are frozen.
    """
    repo, case = frozen_case(tmp_path)
    config = empty_review_config(repo, tmp_path / 'trajectory', backend='codex')
    frozen = native_configuration(config)
    extension = tmp_path / 'operator-directory'
    extension.mkdir()
    marker = tmp_path / 'extension-executed'
    (extension / '__init__.py').write_text('from pathlib import Path\nPath(' + repr(str(marker))
                                          + ').write_text("must not execute")\n')
    (extension / 'auth.json').write_text('{"refresh_token":"do-not-read-or-fingerprint"}')
    monkeypatch.setenv('DAYDREAM_EXT_DIR', str(extension))
    read_bytes = Path.read_bytes

    def bounded_read(path: Path) -> bytes:
        if path.is_relative_to(extension):
            pytest.fail('benchmark admission read external extension or credential bytes')
        return read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', bounded_read)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('external extension selection reached generation')

    monkeypatch.setattr(runner, 'create_backend', forbidden)
    result = await review_case(case, repo, config, 'fixture', runtime=frozen)
    assert result.status == 'failed' and result.candidate is None and result.terminal_result is None
    assert not marker.exists()
    assert 'DAYDREAM_EXT_DIR' in result.diagnostics[0]
    assert 'do-not-read-or-fingerprint' not in json.dumps(asdict(result))


async def test_frozen_pi_defaults_refuse_case_thinking_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pi must reject case-local effort when the frozen operator defaults omit it.

    The native success cases freeze explicit effort; they miss inheritance of a
    case setting when no operator effort was selected. Exercise the actual
    backend guard that owns native argument selection before spawning Pi.
    """
    from daydream.backends.pi import PiBackend, PiError

    repo, _ = frozen_case(tmp_path)
    write_and_stage(repo, '.pi/settings.json', '{"defaultThinkingLevel":"high"}')
    commit(repo, 'case-local thinking default')
    operator = tmp_path / 'operator'
    (operator / '.pi').mkdir(parents=True)
    (operator / '.pi/settings.json').write_text(
        '{"defaultProvider":"nous","defaultModel":"deepseek/deepseek-v4.1-flash"}')
    monkeypatch.delenv('PI_THINKING', raising=False)
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'pi-agent'))

    config = empty_review_config(operator, tmp_path / 'trajectory', backend='pi')
    _, execution = native_configuration(config)
    backend = PiBackend(cwd=repo, execution_input=execution.backends['pi'])
    with pytest.raises(PiError, match='Case Pi thinking conflicts with unavailable operator effort'):
        async for _ in backend.execute(repo, 'Review this change'):
            pytest.fail('case-local effort reached the Pi transport')


@pytest.mark.parametrize(("selection", "provider"), [
    ("explicit", "nous"), ("project", "nous"), ("project_phase", "nous"),
    ("project_cli", "openrouter"), ("project_cli_missing", "nous"), ("project_env", "nous"), ("agent_relative", "nous"),
])
async def test_native_pi_retains_nous_provider_model_and_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, selection: str, provider: str,
) -> None:
    """Freeze operator configuration and cwd-relative credentials across isolated runs."""
    repo, case = frozen_case(tmp_path)
    if selection in {"explicit", "project_env"}:
        monkeypatch.setenv("PI_PROVIDER", "nous")
    else:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    monkeypatch.delenv("PI_THINKING", raising=False)
    monkeypatch.setenv("NOUS_API_KEY", "offline-native-auth-canary")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-other-auth-canary")
    agent_dir = tmp_path / 'operator-pi'
    if selection == 'agent_relative':
        operator_cwd = tmp_path / 'operator-cwd'
        operator_cwd.mkdir()
        monkeypatch.chdir(operator_cwd)
        agent_dir = operator_cwd / '.operator-pi'
        monkeypatch.delenv('NOUS_API_KEY')
        monkeypatch.delenv('OPENROUTER_API_KEY')
    agent_dir.mkdir()
    auth_body = '{"nous":{"type":"api_key","key":"offline-file-auth-canary"}}'
    (agent_dir / 'auth.json').write_text(auth_body)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", '.operator-pi' if selection == 'agent_relative' else str(agent_dir))
    real_exec: Any = asyncio.create_subprocess_exec
    captured: list[tuple[list[str], dict[str, Any]]] = []
    auth_bodies: list[str | None] = []
    expected_usage: list[tuple[int, int, int, float]] = []
    operator = tmp_path / "operator"
    operator.mkdir()
    overridden = selection in {"project_cli", "project_phase", "project_cli_missing"}
    if selection != "explicit":
        settings_dir = agent_dir if selection == 'agent_relative' else operator / '.pi'
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings = {"defaultModel": "operator-default-model" if overridden else "deepseek/deepseek-v4.1-flash",
                    "defaultThinkingLevel": "low", "apiKey": "must-not-fingerprint-this-secret"}
        if selection != "project_cli_missing":
            settings["defaultProvider"] = "openrouter" if selection == "project_env" else provider
        (settings_dir / 'settings.json').write_text(json.dumps(settings))
        write_and_stage(repo, ".pi/settings.json", '{"defaultProvider":"other","defaultModel":"wrong-model"}')
        case = replace(case, head=commit(repo, "case native settings cannot replace operator settings"))

    async def native_process(*args: object, **kwargs: Any) -> MagicMock:
        if not args or args[0] != "pi":
            return cast(MagicMock, await real_exec(*args, **kwargs))
        argv = [str(arg) for arg in args]
        captured.append((argv, kwargs))
        auth_path = Path(kwargs['cwd']) / kwargs['env']['PI_CODING_AGENT_DIR'] / 'auth.json'
        auth_bodies.append(auth_path.read_text() if auth_path.is_file() else None)
        prompt = Path(argv[-1][1:]).read_text()
        stage = review_stage_state(prompt)
        if stage is None and "--extension" not in argv:
            expected_usage.append((110, 50, 10, 0.0003))
            return make_mock_process_from_fixture("simple_text.jsonl")
        expected_usage.append((13, 3, 2, 0.125))
        payload = stage_result(stage) if stage is not None else {"issues": []}
        call = {"type": "toolCall", "id": "submit", "name": "structured_output", "arguments": payload}
        assistant = {"role": "assistant", "content": [call], "stopReason": "toolUse",
            "usage": {"input": 11, "output": 3, "cacheRead": 2, "cost": {"total": 0.125}}}
        tool = {"content": [{"type": "text", "text": "Submitted."}], "details": payload}
        return make_mock_process([json.dumps(row) for row in [
            {"type": "session", "sessionId": "offline-fixture"}, {"type": "agent_start"}, {"type": "turn_start"},
            {"type": "message_end", "message": assistant},
            {"type": "tool_execution_start", "toolCallId": "submit", "toolName": "structured_output", "args": payload},
            {"type": "tool_execution_end", "toolCallId": "submit", "toolName": "structured_output",
             "isError": False, "result": tool},
            {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "submit",
             "toolName": "structured_output", "isError": False, **tool}},
            {"type": "turn_end", "message": assistant}, {"type": "agent_end", "messages": []},
            {"type": "agent_settled"}]])

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", native_process)
    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi",
        file_config=DaydreamFileConfig(model="deepseek/deepseek-v4.1-flash", reasoning_effort="low")
        if selection == "explicit" else DaydreamFileConfig())
    if selection != "explicit":
        config.target = str(operator)
    if selection in {"project_cli", "project_cli_missing"}:
        config.model = "deepseek/deepseek-v4.1-flash"
    elif selection == "project_phase":
        config.file_config = DaydreamFileConfig(phases={"per_stack_review": {"model": "deepseek/deepseek-v4.1-flash"}})
    result = await review_case(case, repo, config, "native-pi-fixture")
    assert result.status == "completed_empty", result.diagnostics
    assert captured
    assert auth_bodies == [auth_body] * len(captured)
    expected_models = {"deepseek/deepseek-v4.1-flash"}
    for argv, kwargs in captured:
        assert argv[argv.index("--provider") + 1] == provider
        assert argv[argv.index("--thinking") + 1] in ({"low"} if selection == "explicit" else {"low", "medium"})
        if selection != 'agent_relative':
            assert kwargs["env"]["NOUS_API_KEY"] == "offline-native-auth-canary"
            if provider == "openrouter":
                assert kwargs["env"]["OPENROUTER_API_KEY"] == "offline-other-auth-canary"
    assert {argv[argv.index("--model") + 1] for argv, _ in captured} == expected_models
    assert result.candidate is not None
    assert result.candidate['usage'] == {"prompt_tokens": sum(row[0] for row in expected_usage),
        "completion_tokens": sum(row[1] for row in expected_usage),
        "cached_tokens": sum(row[2] for row in expected_usage),
        "cost_usd": pytest.approx(sum(row[3] for row in expected_usage)),
        "total_tokens": sum(row[0] + row[1] for row in expected_usage)}
    assert {row['provider'] for row in result.configuration['observed_requests']} == {provider}
    assert {row['model'] for row in result.configuration['observed_requests']} == expected_models
    assert {'wonder', 'arbiter', 'parse'} <= result.configuration['phases'].keys()
    if selection == "explicit":
        assert config.file_config is not None and config.file_config.model == "deepseek/deepseek-v4.1-flash"
        assert os.environ["PI_PROVIDER"] == "nous"
    else:
        if selection != "project_cli_missing":
            assert result.configuration['native']['pi']['defaults']['defaultProvider'] == (
                "openrouter" if selection == "project_env" else provider)
        assert 'must-not-fingerprint-this-secret' not in json.dumps(asdict(result), default=str)
        if selection == "project_env":
            assert os.environ["PI_PROVIDER"] == "nous"
        else:
            assert 'PI_PROVIDER' not in os.environ
    assert "offline-native-auth-canary" not in json.dumps(asdict(result), default=str)
    assert "offline-other-auth-canary" not in json.dumps(asdict(result), default=str)
    assert 'offline-file-auth-canary' not in json.dumps(asdict(result), default=str)
    assert result.configuration['native']['pi']['selectors']['PI_CODING_AGENT_DIR'] == str(agent_dir)
    assert os.environ['PI_CODING_AGENT_DIR'] == ('.operator-pi' if selection == 'agent_relative' else str(agent_dir))


@pytest.mark.parametrize("fault", ["missing_terminal", "malformed_projection", "ignored_patch", "moved_head"])
async def test_native_artifact_failure_or_changed_comparison_never_becomes_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """Artifact faults and changed comparisons cannot produce scorable empty success."""
    repo, case = frozen_case(tmp_path)
    original = tracked_source_state(repo)
    injected: list[str] = []
    real_read = Path.read_text
    real_replace = os.replace

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        contents = real_read(path, *args, **kwargs)
        if fault == "malformed_projection" and path.name == "merged-items.json":
            injected.append(path.name)
            return '{"items": "not canonical findings"}'
        return contents

    def replace_file(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        # atomic_write_json stages complete coverage bytes before installing them
        # into place; failure there occurs before the real terminal handoff.
        if fault == "missing_terminal" and Path(destination).name == "review-coverage.json":
            injected.append(Path(destination).name)
            raise OSError("synthetic terminal coverage installation failure")
        real_replace(source, destination, *args, **kwargs)

    def create(*_args: Any, **kwargs: Any) -> EmptyReviewBackend:
        checkout = kwargs["cwd"]
        if fault == "moved_head":
            git_ops.checkout_detach(checkout, case.base)
            injected.append("head")
        return EmptyReviewBackend(checkout)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "replace", replace_file)
    monkeypatch.setattr(runner, "create_backend", create)
    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi",
        ignore_paths=["api.py"] if fault == "ignored_patch" else [])
    result = await review_case(case, repo, config, "fixture")
    assert result.status == "failed", result.diagnostics
    assert result.candidate is None and result.diagnostics
    assert tracked_source_state(repo) == original
    if fault == "missing_terminal":
        assert injected and result.terminal_result is None
        assert "native reviewer did not produce a finalized terminal result" in result.diagnostics
    elif fault == "malformed_projection":
        assert injected and result.terminal_result is not None
        assert not result.terminal_result["projection_valid"]
    elif fault == "ignored_patch":
        assert result.terminal_result is not None, result.diagnostics
        assert any("comparison differs" in diagnostic for diagnostic in result.diagnostics)
    else:
        assert injected
        assert any("changed the frozen checkout revision" in diagnostic for diagnostic in result.diagnostics)


async def test_unadvertised_divergent_base_survives_frozen_endpoint_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Materialize both frozen endpoints even when base is unadvertised and divergent."""
    repo, initial = frozen_case(tmp_path)
    git(repo, "checkout", "--detach", initial.base)
    write_and_stage(repo, "base-progress.txt", "The upstream branch advanced independently.\n")
    advanced_base = commit(repo, "advanced base tip")
    git(repo, "checkout", "--detach", initial.head)
    case = replace(initial, base=advanced_base)
    assert advanced_base not in git(repo, "show-ref")
    original = tracked_source_state(repo)

    def create(*_args: Any, **kwargs: Any) -> EmptyReviewBackend:
        checkout = kwargs["cwd"]
        assert git_ops.commit_exists(checkout, advanced_base)
        assert not git_ops.remote_urls(checkout)
        assert not git_ops.list_local_branches(checkout)
        return EmptyReviewBackend(checkout)

    monkeypatch.setattr(runner, "create_backend", create)
    result = await review_case(case, repo, empty_review_config(repo, tmp_path / "trajectory", backend="pi"), "fixture")
    assert result.status == "completed_empty", result.diagnostics
    assert result.analyzed_comparison["merge_base_sha"] == initial.base
    assert result.candidate is not None and result.candidate["pr"]["base"] == advanced_base
    assert tracked_source_state(repo) == original


async def test_result_observer_preserves_terminal_report_and_invocation_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Observer mutations leave native reporting and later invocation scope intact."""
    repo, case = frozen_case(tmp_path)
    received: list[str] = []
    backend = EmptyReviewBackend(repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={"structure": [finding()]})
    backend.merge_echo_records = True
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)

    def observe(result: FinalizedReview) -> None:
        received.append(result.terminal_result["run_id"])
        result.terminal_result["run_id"] = "observer-mutated-run"
        result.terminal_result["analysis_state"] = "failed"

    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi", base=case.base,
                                 model="native-original-model", pr_number=None)
    assert await runner.run(config, review_result_handler=observe) == 0
    assert len(received) == 1
    coverage = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    report = (repo / ".review-output.md").read_text()
    assert coverage["run_id"] == received[0]
    assert all(scope["status"] == "complete" for scope in coverage["stack_outcomes"])
    assert "observer-mutated" not in report and received[0] in report
    assert "<code>complete</code>" in report
    assert await runner.run(config) == 0
    assert len(received) == 1, "a later ordinary invocation must not inherit the observer"


def test_result_delivery_preserves_caller_owned_nested_values() -> None:
    """Observer mutations leave nested caller-owned findings and configuration intact."""
    items = [finding(source_uids=["native-source"])]
    configuration = {"phases": {"per_stack_review": {"model": "native-model"}}}
    received: list[FinalizedReview] = []

    def observe(result: FinalizedReview) -> None:
        received.append(result)
        result.items[0]["source_uids"].append("observer-source")
        result.configuration["phases"]["per_stack_review"]["model"] = "observer-model"

    with bind_run_context(RunContext(InteractionPolicy(), review_result_handler=observe)):
        deliver_finalized_review(terminal_result(), items, configuration)
    assert len(received) == 1
    assert items[0]["source_uids"] == ["native-source"]
    assert configuration["phases"]["per_stack_review"]["model"] == "native-model"


async def test_result_delivery_follows_nested_and_concurrent_run_contexts() -> None:
    """Nested ordinary runs mask observers; sibling tasks keep their own deliveries."""
    received: dict[str, list[str]] = {name: [] for name in ("first", "second")}

    def fail(_result: FinalizedReview) -> None:
        raise RuntimeError("consumer failed")

    async def deliver(name: str) -> None:
        def observe(result: FinalizedReview) -> None:
            received[name].append(result.configuration["owner"])

        with bind_run_context(RunContext(InteractionPolicy(), review_result_handler=observe)):
            with bind_run_context(RunContext(InteractionPolicy())):
                deliver_finalized_review(terminal_result(), [], {"owner": "nested"})
            await asyncio.sleep(0)
            deliver_finalized_review(terminal_result(), [], {"owner": name})
            with pytest.raises(RuntimeError, match="consumer failed"):
                with bind_run_context(RunContext(InteractionPolicy(), review_result_handler=fail)):
                    deliver_finalized_review(terminal_result(), [], {})
            deliver_finalized_review(terminal_result(), [], {"owner": name})

    async with asyncio.TaskGroup() as tasks:
        for name in received:
            tasks.create_task(deliver(name))
    deliver_finalized_review(terminal_result(), [], {"owner": "outside"})
    assert received == {name: [name, name] for name in received}
