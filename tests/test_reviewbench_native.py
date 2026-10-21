"""Native ReviewBench boundary regressions using real Git and the ordinary runner.

The existing terminal-result journeys own coverage derivation; these tests add the
missing adapter risks: frozen source identity, conservative HEAD coordinates,
nonpublishing isolated execution, and preservation of native Pi authentication.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, CostEvent, EffectiveRequestConfig, RequestEvent
from daydream.benchmark.reviewbench.contract import Case
from daydream.benchmark.reviewbench.native import convert_findings, review_case
from daydream.config_file import DaydreamFileConfig
from daydream.deep.artifacts import diff_key
from daydream.review_result import FinalizedReview
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.dataset_hub import FakeDatasetHub
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, init_repo, tracked_source_state, write_and_stage
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
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
    """Exit zero/empty output cannot hide failed scopes or discarded native findings.

    Existing test_review_completion exercises publication; this adapter journey
    proves the same terminal authority reaches nonpublishing historical cases.
    """
    repo, case = frozen_case(tmp_path)
    original = tracked_source_state(repo)
    created: list[EmptyReviewBackend] = []
    roots: list[Path] = []
    nonempty = outcome in {"finding", "partial", "relocated", "pipeline_failure"}

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == "pi"
        checkout = kwargs["cwd"]
        assert checkout != repo and not git_ops.remote_urls(checkout)
        roots.append(checkout)
        backend = EmptyReviewBackend(checkout, forbid_merge=False, forbid_supervise=False,
            review_by_stack={"structure": [finding(line=500 if outcome == "relocated" else 2)]} if nonempty else {},
            fail_stack="python" if outcome == "partial" else None,
            responder=(lambda prompt: [RuntimeError("offline failure")] if (
                outcome == "failed" and ("you are reviewing the " in prompt.lower()
                                         or "you are the structural reviewer" in prompt.lower())
                or outcome == "pipeline_failure" and "supervisor adjudication" in prompt.lower()) else None))
        backend.merge_echo_records = True
        created.append(backend)
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
    assert created and all(not root.exists() for root in roots)
    assert not (tmp_path / "must-not-publish.json").exists()
    assert not (tmp_path / "ignored-trajectory.json").exists()
    assert not dataset_store.exists() and dataset_hub.commits == []
    assert not fake_gh.calls("POST")
    assert result.configuration["phases"]["per_stack_review"]["model"] == config.model
    if status in {"failed", "incomplete"}:
        assert result.candidate is None
    else:
        assert result.candidate is not None
        assert result.candidate["pr"] == asdict(case)
        assert bool(result.candidate["findings"]) == (outcome == "finding")
        assert "usage" not in result.candidate
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


@pytest.mark.parametrize("fault", ["dirty", "identity", "missing", "mirror"])
async def test_frozen_input_is_verified_before_native_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """A supplied SHA label must not admit the wrong repository or dirty source.

    Existing workspace tests resolve refs for ordinary runs; they do not validate
    supplied benchmark identities or reject dirty caches before cloning.
    """
    repo, case = frozen_case(tmp_path)
    if fault == "dirty":
        (repo / "untracked.py").write_text("not part of the frozen case")
    elif fault == "identity":
        git(repo, "remote", "set-url", "origin", "https://github.com/another/project.git")
    elif fault == "missing":
        case = replace(case, head="f" * 40)
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


def test_head_conversion_retains_logical_records_and_exposes_ambiguous_locations(tmp_path: Path) -> None:
    """Native line hints cannot turn old/deleted coordinates into HEAD findings.

    Native location-validator tests own snapping; this distinct external format
    boundary must reject that snap and numeric old/new ambiguity explicitly.
    """
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
    assert [(row["start_line"], row["end_line"]) for row in converted] == [(1, 1), (3, 3)]
    assert all(row["producer"] == "fixture" and row["file"] == "api.py" for row in converted)
    assert {row["id"]: row["reason"] for row in omitted} == {
        2: "ambiguous_old_or_new_side_line_hint", 4: "location_absent_from_head_regular_file",
        5: "missing_repository_relative_positive_location", 6: "location_absent_from_head_regular_file",
        7: "native_location_was_relocated_or_distrusted", 8: "missing_repository_relative_positive_location",
        9: "missing_finding_message"}


def test_head_conversion_checks_old_coordinates_outside_captured_hunks(tmp_path: Path) -> None:
    """An old-file line remains ambiguous when insertion moves it outside the old hunk.

    The replacement fixture above has overlapping old/new hunk ranges and misses
    a bare native hint that is old-side-valid elsewhere in the actual old file.
    """
    repo, initial = frozen_case(tmp_path)
    old_contents = "".join(f"# original {number}\n" for number in range(1, 41))
    write_and_stage(repo, "api.py", old_contents)
    base = commit(repo, "long old file")
    write_and_stage(repo, "api.py", "".join(f"# inserted {number}\n" for number in range(1, 11)) + old_contents)
    head = commit(repo, "insert lines at start")
    case = replace(initial, base=base, head=head)
    comparison = git_ops.diff_paths(repo, base, head, ["."])
    assert "@@ -1,3 +1,13 @@" in comparison
    converted, omitted = convert_findings([finding(line=10)], repository=repo, case=case,
        producer="fixture", comparison=comparison)
    assert converted == []
    assert omitted == [{"uid": None, "id": 1, "reason": "ambiguous_old_or_new_side_line_hint"}]


@pytest.mark.parametrize('incomplete', [False, True])
async def test_native_usage_preserves_observed_zero_without_inventing_missing_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, incomplete: bool,
) -> None:
    """Cumulative usage-only events cannot manufacture unavailable prompt/cache counts.

    Positive Pi totals below miss both observed zero and absent dimensions when
    invocation costs and sibling trajectories are folded into the native result.
    """
    repo, case = frozen_case(tmp_path)

    @dataclass(frozen=True)
    class CustomConfig(EffectiveRequestConfig):
        arbitrary: str = 'custom-unstructured-canary'

    class MeasuredBackend(EmptyReviewBackend):
        async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            yield RequestEvent(prompt='Unpersisted request prompt', config=CustomConfig())
            yield CostEvent(cost_usd=0.0, input_tokens=None, output_tokens=0, cached_tokens=None)
            async for event in super().execute(*args, **kwargs):
                yield event

    monkeypatch.setattr(runner, 'create_backend', lambda *_args, **kwargs: MeasuredBackend(
        kwargs['cwd'], fail_stack='python' if incomplete else None))
    result = await review_case(case, repo, empty_review_config(repo, tmp_path / 'trajectory', backend='pi'), 'fixture')
    assert result.status == ('incomplete' if incomplete else 'completed_empty'), result.diagnostics
    assert result.usage == {'completion_tokens': 0, 'cost_usd': 0.0}
    if not incomplete:
        assert result.candidate is not None and result.candidate['usage'] == result.usage
    else:
        assert result.candidate is None
    assert result.configuration['observed_requests']
    assert all(row['controls'] == {} for row in result.configuration['observed_requests'])
    assert 'custom-unstructured-canary' not in json.dumps(asdict(result))


async def test_native_extension_phase_retains_its_own_backend_execution_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, ext_dir: ExtDir,
) -> None:
    """A configured extension phase must not inherit another backend's retry policy.

    Homogeneous native cases and existing singular execution-input tests cannot
    expose a runtime map that omits extension-owned phase configuration.
    """
    repo, case = frozen_case(tmp_path)
    monkeypatch.delenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', raising=False)
    ext_dir.write_module(
        'from daydream.agent import run_agent\n'
        'from daydream.extensions import FlowStep\n'
        'from daydream.trajectory import DaydreamPhase\n'
        'async def probe(ctx):\n'
        '    await run_agent(ctx.backend_for("extension-review"), ctx.work.repo, "NATIVE_EXTENSION", '
        'phase=DaydreamPhase.REVIEW, run_context=ctx.run_context)\n'
        'def register(registry):\n'
        '    registry.register_phase(FlowStep(name="native-probe", run=probe))\n'
        '    registry.insert_before("deep", anchor="intent", step="native-probe")\n')
    created: list[tuple[str, EmptyReviewBackend]] = []

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert kwargs['execution_input'].retry_policy.base_delay_s == (10.0 if name == 'pi' else 2.0)
        backend = EmptyReviewBackend(kwargs.get('cwd', repo))
        backend.model = kwargs['model']
        created.append((name, backend))
        return backend

    monkeypatch.setattr(runner, 'create_backend', create)
    config = empty_review_config(repo, tmp_path / 'trajectory', file_config=DaydreamFileConfig(
        backend='pi', supervisor='llm', phases={'extension-review': {'backend': 'claude', 'model': 'claude-sonnet-5'}}))
    result = await review_case(case, repo, config, 'fixture')
    assert result.status == 'completed_empty', result.diagnostics
    assert any(name == 'claude' and backend.calls for name, backend in created)
    assert result.configuration['phases']['extension-review']['model'] == 'claude-sonnet-5'


async def test_case_pi_thinking_cannot_replace_unavailable_operator_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Unknown operator effort must not inherit a historical case's project default.

    The explicit/project selection tests below have known effort and cannot
    expose native Pi's fallback to case-local settings when no effort is pinned.
    """
    repo, case = frozen_case(tmp_path)
    write_and_stage(repo, '.pi/settings.json', '{"defaultThinkingLevel":"high"}')
    case = replace(case, head=commit(repo, 'case-local thinking default'))
    operator = tmp_path / 'operator'
    (operator / '.pi').mkdir(parents=True)
    (operator / '.pi/settings.json').write_text(
        '{"defaultProvider":"nous","defaultModel":"deepseek/deepseek-v4.1-flash"}')
    monkeypatch.delenv('PI_THINKING', raising=False)
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'pi-agent'))
    spawned: list[str] = []
    real_exec = asyncio.create_subprocess_exec

    async def no_model(*args: Any, **kwargs: Any) -> Any:
        if args[0] == 'pi':
            spawned.append('pi')
            raise AssertionError('a conflicting case must fail before model work')
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr('daydream.backends._transport.asyncio.create_subprocess_exec', no_model)
    config = empty_review_config(operator, tmp_path / 'trajectory', backend='pi')
    result = await review_case(case, repo, config, 'fixture')
    assert result.status == 'failed' and result.candidate is None and not spawned
    assert result.terminal_result is None
    assert 'Case Pi thinking conflicts with unavailable operator effort' in result.diagnostics


@pytest.mark.parametrize("selection", ["explicit", "project"])
async def test_native_pi_retains_nous_provider_model_and_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, selection: str,
) -> None:
    """The adapter must not inherit Harbor's forced OpenRouter provider policy.

    Existing Pi runner tests establish transport precedence; this real adapter
    journey proves its isolation and configuration copy preserve that boundary.
    """
    repo, case = frozen_case(tmp_path)
    if selection == "explicit":
        monkeypatch.setenv("PI_PROVIDER", "nous")
    else:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    monkeypatch.setenv("NOUS_API_KEY", "offline-native-auth-canary")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "operator-pi"))
    real_exec: Any = asyncio.create_subprocess_exec
    captured: list[tuple[list[str], dict[str, Any]]] = []
    expected_usage: list[tuple[int, int, int, float]] = []
    operator = tmp_path / "operator"
    if selection == "project":
        (operator / ".pi").mkdir(parents=True)
        (operator / ".pi/settings.json").write_text(json.dumps({"defaultProvider": "nous",
            "defaultModel": "deepseek/deepseek-v4.1-flash", "defaultThinkingLevel": "low",
            "apiKey": "must-not-fingerprint-this-secret"}))
        write_and_stage(repo, ".pi/settings.json", '{"defaultProvider":"other","defaultModel":"wrong-model"}')
        case = replace(case, head=commit(repo, "case native settings cannot replace operator settings"))

    async def native_process(*args: object, **kwargs: Any) -> MagicMock:
        if not args or args[0] != "pi":
            return cast(MagicMock, await real_exec(*args, **kwargs))
        argv = [str(arg) for arg in args]
        captured.append((argv, kwargs))
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
    if selection == "project":
        config.target = str(operator)
    result = await review_case(case, repo, config, "native-pi-fixture")
    assert result.status == "completed_empty", result.diagnostics
    assert captured
    for argv, kwargs in captured:
        assert argv[argv.index("--provider") + 1] == "nous"
        assert argv[argv.index("--model") + 1] == "deepseek/deepseek-v4.1-flash"
        assert argv[argv.index("--thinking") + 1] in ({"low"} if selection == "explicit" else {"low", "medium"})
        assert kwargs["env"]["NOUS_API_KEY"] == "offline-native-auth-canary"
    assert result.candidate is not None
    assert result.candidate['usage'] == {"prompt_tokens": sum(row[0] for row in expected_usage),
        "completion_tokens": sum(row[1] for row in expected_usage),
        "cached_tokens": sum(row[2] for row in expected_usage),
        "cost_usd": pytest.approx(sum(row[3] for row in expected_usage)),
        "total_tokens": sum(row[0] + row[1] for row in expected_usage)}
    assert {row['provider'] for row in result.configuration['observed_requests']} == {'nous'}
    assert {row['model'] for row in result.configuration['observed_requests']} == {'deepseek/deepseek-v4.1-flash'}
    assert {'wonder', 'arbiter', 'parse'} <= result.configuration['phases'].keys()
    if selection == "explicit":
        assert config.file_config is not None and config.file_config.model == "deepseek/deepseek-v4.1-flash"
        assert os.environ["PI_PROVIDER"] == "nous"
    else:
        assert result.configuration['native']['pi']['defaults']['defaultProvider'] == 'nous'
        assert 'must-not-fingerprint-this-secret' not in json.dumps(asdict(result), default=str)
        assert 'PI_PROVIDER' not in os.environ
    assert "offline-native-auth-canary" not in json.dumps(asdict(result), default=str)


@pytest.mark.parametrize("fault", ["missing_terminal", "malformed_projection", "ignored_patch", "moved_head"])
async def test_native_artifact_failure_or_changed_comparison_never_becomes_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """Lost artifacts and a different actual patch cannot be scored under supplied SHAs.

    Existing terminal tests own atomic artifact publication; this adapter contract
    rejects missing callbacks, invalid canonical projection, and mismatched native
    comparisons using real filesystem/Git operations instead of fabricated results.
    """
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
    """SHA-only mirror caches need both endpoints, even when base is not a HEAD ancestor.

    The ancestor fixtures above cannot expose ordinary cloning dropping an
    unadvertised base tip; this real Git graph matches the supplied fetch workflow.
    """
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


async def test_result_observer_cannot_mutate_native_evidence_or_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """The supported handoff owns copies rather than aliases into native authority.

    Existing terminal artifact tests have no embedded observer; a callback that
    edits its result must not corrupt the final coverage, canonical items or report.
    """
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
        result.items[0]["description"] = "observer-mutated-finding"
        result.configuration["phases"]["per_stack_review"]["model"] = "observer-mutated-model"

    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi", base=case.base,
                                 model="native-original-model", pr_number=None)
    assert await runner.run(config, review_result_handler=observe) == 0
    assert len(received) == 1
    coverage = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    items = json.loads((repo / ".daydream/deep/merged-items.json").read_text())
    report = (repo / ".review-output.md").read_text()
    assert coverage["run_id"] == received[0]
    assert all(scope["status"] == "complete" for scope in coverage["stack_outcomes"])
    assert items["items"][0]["description"] == finding()["description"]
    assert "observer-mutated" not in report and received[0] in report
    assert "<code>complete</code>" in report and config.model == "native-original-model"
    assert await runner.run(config) == 0
    assert len(received) == 1, "a later ordinary invocation must not inherit the observer"
