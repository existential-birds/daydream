"""Bounded review stages through runner.run, real Git and the external Backend seam."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream import clock
from daydream.backends import (
    AgentEvent,
    ResultEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexLifecycleError
from daydream.backends.lifecycle import BackendLifecycleFailure
from daydream.review_source_refs import SOURCE_SELECTORS_SCHEMA
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.stub_backend import review_stage_state, stage_result
from tests.test_deep_orchestrator import _profile_with_pipeline, _sanctioned_inputs

StageResponse = Callable[[dict[str, Any], dict[str, Any]],
                         Iterable[AgentEvent] | AsyncGenerator[AgentEvent, None] | None]


class StagedBackend(EmptyReviewBackend):
    """A provider that remembers nothing beyond the current request."""

    supports_review_instructions = True
    supports_budget_preamble = False

    def __init__(self, repo: Path) -> None:
        super().__init__(repo, forbid_merge=False, forbid_supervise=False)
        self.merge_echo_records = True
        self.stages: list[dict[str, Any]] = []
        self.stage_response: StageResponse | None = None
        self.scripts: dict[str, tuple[StageResponse, bool]] = {}

    def script(self, scope_id: str = 'python', *, terminal: bool = True) -> Callable[[StageResponse], StageResponse]:
        """Register only provider behavior; the real runner owns admission and lifecycle."""
        def register(response: StageResponse) -> StageResponse:
            self.scripts[scope_id] = (response, terminal)
            return response
        return register

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        stage = review_stage_state(prompt)
        if stage is not None:
            self.stages.append(stage)
            self.calls.append({'cwd': cwd, 'prompt': prompt,
                               'output_schema': args[0] if args else kwargs.get('output_schema'), **kwargs})
            output = stage_result(stage)
            response, terminal = self.scripts.get(stage['scope_id'], (self.stage_response, True))
            events = response(stage, output) if response else None
            if isinstance(events, AsyncGenerator):
                async with aclosing(events):
                    async for event in events:
                        yield event
            else:
                for event in events or ():
                    yield event
            if not terminal:
                return
            yield ResultEvent(structured_output=output, continuation=None)
            return
        kwargs.pop('review_instructions', None)
        kwargs.pop('wall_budget_s', None)
        kwargs.pop('tool_call_budget', None)
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


class InvestigationRun(ReviewRun):
    backend: StagedBackend

    def __init__(self, repo: Path, tmp: Path, patch: pytest.MonkeyPatch) -> None:
        super().__init__(repo, tmp, patch)
        self.backend = StagedBackend(repo)

    async def finish(self, scope_id: str, *, reason: str | None = None, findings: tuple[str, ...] = (),
                     statuses: tuple[str, ...] | None = None, expected_exit: int = 0,
                     **overrides: Any) -> dict[str, Any]:
        assert await self.run(**overrides) == expected_exit
        data = self.load()
        assert [finding['title'] for finding in data['findings']] == list(findings)
        inventory = scopes(data)
        scope = inventory[scope_id]
        assert all(s['status'] == 'complete' for name, s in inventory.items() if name != scope_id)
        assert scope['status'] in (statuses or ('incomplete' if reason else 'complete',)), scope['reason_codes']
        assert scope['reason_codes'] == ([reason] if reason else [])
        assert not any(call.get('tools_disabled') for call in self.backend.calls)
        return data


@pytest.fixture
def investigation(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InvestigationRun:
    return InvestigationRun(multi_stack_target, tmp_path, monkeypatch)


def candidate(*, disposition: str = 'open', candidate_id: str = '',
              finding: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    return {'candidate_id': candidate_id, 'file': 'api.py', 'line': 2,
            'trigger': 'Calling hello()', 'consequence': 'The returned greeting may violate callers.',
            'grounds': "api.py:2 returns 'universe'; current callers accept that greeting.",
            'disposition': disposition, 'finding': finding, **fields}


def read_events(repo: Path, path: str, *, event_id: str) -> Iterable[AgentEvent]:
    """Read real source, retaining the provider's start/result event boundary."""
    source = (repo / path).read_text()
    yield ToolStartEvent(id=event_id, name='Read', input={'file_path': path, 'offset': 1,
                                                       'limit': len(source.splitlines())})
    yield ToolResultEvent(id=event_id, output=source, is_error=False)


async def test_scope_population_keeps_assignment_and_triage_decisions_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad sibling cannot borrow IDs, optional tool authority, or another scope's findings."""
    discovery = {
        "unknown": "malformed_output",
        "read-all": "malformed_output",
        "handoff-bytes": "evidence_incomplete",
        "truncated-tool-output": None,
        "optional-failures": None,
        "invalid-disposition": "malformed_output",
    }
    triage = {
        "confirmed": None,
        "unresolved": "evidence_incomplete",
        "contradiction": "evidence_incomplete",
        "extra-target": "malformed_output",
    }
    cases = {
        **{"discovery-" + key: reason for key, reason in discovery.items()},
        **{"triage-" + key: reason for key, reason in triage.items()},
    }
    repo = tmp_path / "decision_population"
    before = {f"{case}/api.py": "def hello():\n    return 'world'\n" for case in cases}
    seed_feature_branch(
        repo, base=before, feature={path: body.replace("world", "universe") for path, body in before.items()}
    )
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    scope_cases: dict[str, str] = {}
    closed: dict[str, str] = {}

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage["scope_id"] == "structure":
            return
        scope_id = stage['scope_id']
        case = stage["assigned_files"][0].split("/")[0] if stage["stage"] == "first_pass" else scope_cases[scope_id]
        scope_cases[scope_id] = case
        family, fault = case.split("-", 1)
        path = f"{case}/api.py"
        finding = dict(record(), file=path, description="Confirmed " + case, evidence=f"{path}:2")
        concrete = candidate(disposition="confirmed", file=path, finding=finding)
        if family == "discovery":
            if fault == "unknown":
                output["targets"][0]["target_id"] = "foreign.py"
            elif fault == "duplicate":
                output["targets"] *= 2
            elif fault in {"omitted", "read-all"}:
                output["targets"] = []
            elif fault == "handoff-bytes":
                output["notes"] = "x" * 70_000
            else:
                output["candidates"] = [concrete] * (129 if fault == "handoff-items" else 1)
                if fault == "missing-grounds":
                    concrete["grounds"] = ""
                elif fault == "invalid-disposition":
                    concrete["disposition"] = []
            if fault in {"read-all", "truncated-tool-output"}:
                yield ToolStartEvent(id=case, name="Read", input={"file_path": path})
                yield ToolResultEvent(
                    id=case,
                    output=(repo / path).read_text(),
                    is_error=False,
                    truncated=fault == "truncated-tool-output",
                )
            elif fault == "unmatched-tool-result":
                yield ToolResultEvent(id=case, output=(repo / path).read_text(), is_error=False)
            elif fault == "optional-failures":
                yield ToolStartEvent(id=case, name="Bash", input={"command": "rg definitely_absent ."})
                yield ToolResultEvent(
                    id=case, output="x" * (2 * 1024 * 1024 + 1), is_error=True, exit_code=1, truncated=True
                )
                yield ToolResultEvent(id="unassociated", output="failed optional lookup", is_error=True)
            return
        contract = stage['response_contract']
        assert contract["schema"] == review.backend.calls[-1]["output_schema"]
        validator = Draft202012Validator(contract['schema'])
        if stage["stage"] == "first_pass":
            output["candidates"] = [
                concrete if fault == "contradiction" else candidate(disposition="rejected", file=path),
                candidate(file=path),
            ]
            assert validator.is_valid(output)
            assert not validator.is_valid(dict(output, candidates=[candidate(candidate_id="invented-native-id")]))
            return
        pending, = stage['candidates']
        (closed_id,) = stage["closed_candidate_ids"]
        closed[scope_id] = closed_id
        assert closed_id != pending["candidate_id"]
        assert stage["progress"] == stage["assigned_target_ids"] == [] and contract["skeleton"]["targets"] == []
        assert stage["assigned_candidate_ids"] == [pending["candidate_id"]] and pending["disposition"] == "open"
        disposition = (
            "confirmed"
            if fault in {"confirmed", "missing-finding"}
            else "unresolved"
            if fault == "unresolved"
            else "rejected"
        )
        output["candidates"] = [
            candidate(
                disposition=disposition,
                candidate_id=pending["candidate_id"],
                file=path,
                finding=finding if fault == "confirmed" else None,
            )
        ]
        if fault == "unknown":
            output["candidates"][0]["candidate_id"] = closed_id
        elif fault == "duplicate":
            output["candidates"] *= 2
        elif fault == "omitted":
            output["candidates"] = []
        elif fault == "contradiction":
            output["notes"] = f"{path}:2 already returns a greeting that current callers accept."
            output["contradictions"] = [closed_id]
        elif fault == "unknown-contradiction":
            output["contradictions"] = ["foreign-candidate"]
        elif fault == "extra-target":
            output["targets"] = [{"target_id": path, "status": "reviewed", "reason": ""}]
        if fault in {"rejected", "extra-target"}:
            assert validator.is_valid(output) is (fault == "rejected")

    review.backend.stage_response = response
    assert (
        await review.run(
            deep_shard_enabled=True, deep_shard_max_files=1, deep_shard_max_bytes=10**9, deep_shard_fanout_cap=32
        )
        == 0
    )
    data = review.load()
    assert len(scope_cases) == len(cases) and set(scope_cases.values()) == set(cases)
    expected = {"Confirmed discovery-" + fault for fault, reason in discovery.items() if reason is None}
    expected |= {"Confirmed triage-confirmed", "Confirmed triage-contradiction"}
    assert len(data["findings"]) == len(expected) and {finding["title"] for finding in data["findings"]} == expected
    assert scopes(data)["structure"]["status"] == "complete"
    for scope_id, case in scope_cases.items():
        reason = cases[case]
        outcome = scopes(data)[scope_id]
        assert outcome["files"] == [f"{case}/api.py"]
        assert outcome["status"] == ("incomplete" if reason else "complete")
        assert outcome["reason_codes"] == ([reason] if reason else [])
        events = stage_ends(review, scope_id)
        family, fault = case.split("-", 1)
        assert [event["metadata"]["attempt"] for event in events] == ([1, 1] if family == "triage" else [1])
        if family == "triage":
            assert [stage["stage"] for stage in review.backend.stages if stage["scope_id"] == scope_id] == [
                "first_pass",
                "triage",
            ]
            assert (
                closed[scope_id]
                != next(stage for stage in reversed(review.backend.stages) if stage["scope_id"] == scope_id)[
                    "assigned_candidate_ids"
                ][0]
            )
            if fault in {"rejected", "extra-target"}:
                assert events[-1]["metadata"]["failure_class"] == ("assignment_mismatch" if reason else None)
        else:
            assert (events[0]["status"], events[0].get("reason_code")) == (
                ("failed", "domain_failure") if reason else ("succeeded", None)
            )
            if fault == "no-tool-read":
                assert events[0]["metadata"]["observed_tool_starts"] == 0
            elif fault == "optional-failures":
                assert events[0]["metadata"]["admitted"] is True
                assert events[0]["metadata"]["observed_tool_starts"] == 1


def stage_ends(review: ReviewRun, scope_id: str) -> list[dict[str, Any]]:
    trajectory = json.loads((review.tmp / 'trajectory.json').read_text())
    return [event for sub in trajectory['extra']['subtrajectories'] for event in sub.get('phase_events', [])
            if event['event'] == 'phase_end' and event['metadata'].get('review_scope_id') == scope_id]


def assert_host_dispatch_measurements(review: InvestigationRun) -> list[dict[str, Any]]:
    """Bind persisted stage attempts to actual Backend arguments without summing fork copies."""
    trajectory = json.loads((review.tmp / 'trajectory.json').read_text())
    summaries = trajectory['extra']['subtrajectories']
    invocations = [invocation for summary in summaries
                   for invocation in ([summary] if 'invocation_id' in summary else summary.get('invocations', []))]
    assert len({(item['trajectory_id'], item['invocation_id']) for item in invocations}) == len(invocations)
    assert all('host_dispatch' not in summary for summary in summaries if 'invocation_id' not in summary)
    events = {event['scope_id']: event for summary in summaries for event in summary.get('phase_events', [])
              if event['event'] == 'phase_end' and 'logical_stage' in event.get('metadata', {})}
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for invocation in invocations:
        dispatch = invocation.get('host_dispatch', {})
        if 'stage_scope_id' in dispatch:
            event = events[dispatch['stage_scope_id']]
            assert invocation['trajectory_id'] == event['trajectory_id']
            by_scope.setdefault(event['metadata']['review_scope_id'], []).append(dispatch)
    calls_by_scope: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for call in review.backend.calls:
        if (stage := review_stage_state(call['prompt'])) is not None:
            calls_by_scope.setdefault(stage['scope_id'], []).append((stage, call))
    assert set(by_scope) == set(calls_by_scope)
    measurements: list[dict[str, Any]] = []
    for scope, calls in calls_by_scope.items():
        assert len(by_scope[scope]) == len(calls)
        attempts: dict[str, int] = {}
        for (stage, call), dispatch in zip(calls, by_scope[scope], strict=True):
            stage_scope = dispatch['stage_scope_id']
            attempts[stage_scope] = attempts.get(stage_scope, 0) + 1
            event = events[stage_scope]
            assert (event['metadata']['logical_stage'], event['metadata']['attempt']) == (
                stage['stage'], stage['attempt'])
            assert dispatch == {
                'stage_scope_id': stage_scope, 'attempt': attempts[stage_scope],
                'user_utf8_bytes': len(call['prompt'].encode('utf-8')),
                'system_disposition': 'supplied',
                'system_utf8_bytes': len(call['review_instructions'].encode('utf-8')),
            }
            measurements.append(dispatch)
    return measurements


def many_file_review(tmp_path: Path, patch: pytest.MonkeyPatch, *, count: int = 17,
                     lines: int = 1, docs_count: int = 4, feature_message: str = 'feature',
                     feature_overrides: dict[str, str] | None = None,
                     supporting_sources: dict[str, str] | None = None) -> InvestigationRun:
    before = {'api.py': "def hello():\n    return 'world'\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(count - 1)})
    before.update({f'00_guide_{i}.md': '# Guide\nExisting behavior\n' for i in range(docs_count)})
    after = {name: content + ''.join(f'# changed {i}\n' for i in range(lines)) for name, content in before.items()}
    after.update(feature_overrides or {})
    before.update(supporting_sources or {})
    after.update(supporting_sources or {})
    repo = tmp_path / 'many_files'
    seed_feature_branch(repo, base=before, feature=after, feature_message=feature_message)
    review = InvestigationRun(repo, tmp_path, patch)
    # These journeys exercise serial stages under one cumulative reviewer allowance.
    review.config = replace(review.config, deep_shard_enabled=False)
    return review


async def test_optional_supporting_paths_are_confined_without_rejecting_valid_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable references do not veto findings or acquire source/review authority."""
    review = many_file_review(tmp_path, monkeypatch, count=5, docs_count=0)
    outside = tmp_path / 'outside.py'
    outside.write_text('PRIVATE_OUTSIDE_SOURCE\n')
    (review.repo / 'linked.py').symlink_to(outside)
    git(review.repo, 'add', 'linked.py')
    git(review.repo, 'commit', '-m', 'Add an outward symlink source control')
    review = InvestigationRun(review.repo, tmp_path, monkeypatch)
    review.config = replace(review.config, deep_shard_enabled=False)
    notes: list[dict[str, Any]] = []
    supporting_reference: dict[str, Any] = {}

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if 'api.py' in stage['assigned_files']:
            supporting_reference.update({
                'artifact_sha256': hashlib.sha256((review.repo / 'module_03.py').read_bytes()).hexdigest(),
                'part_sha256': None, 'revision': stage['analyzed_revision']['head_sha'],
                'path': 'module_03.py', 'side': 'new', 'source_range': {'start': 1, 'end': 1},
                'fragment': None, 'basis': 'source_inspection', 'role': 'supporting',
            })
            # The provider selects a supporting path; the host pins its whole-blob identity.
            output['targets'][0]['evidence_refs'].append({
                'path': 'module_03.py', 'side': 'new', 'source_range': {'start': 1, 'end': 1},
                'basis': 'source_inspection'})
            output.update(notes='Supporting paths are optional model declarations.', supporting_files=[
                './module_03.py', 'module_03.py', '../outside.py', str(outside),
                'linked.py', 'missing.py', '.git/config', '.daydream/deep/stack-python-records.json',
                'module_03.py\n', 'module_03.py/../api.py'])
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        elif 'module_03.py' in stage['assigned_files']:
            notes.extend(stage['note_context'])
            assert stage['candidates'] == []
            assert 'module_03.py' not in stage['completed_target_ids']

    data = await review.finish('python', findings=('Grounded defect',))
    note, = notes
    assert note['supporting_files'] == ['module_03.py']
    assert 'api.py' in note['files']
    decisions = scopes(data)['python']['target_decisions']
    api_decision = next(decision for decision in decisions if decision['target_id'] == 'api.py')
    assert supporting_reference in api_decision['evidence_refs']
    assert any(reference['role'] == 'assigned' for reference in api_decision['evidence_refs'])
    assert all('PRIVATE_OUTSIDE_SOURCE' not in call['prompt'] for call in review.backend.calls)


class _LookalikeLifecycleError(CodexLifecycleError):
    """A polymorphic payload that must not receive the exact adapter projection."""


_REAP_FAILURE = BackendLifecycleFailure(stage="process_reap", allowance_s=5.0, terminal_received=True,
                                        stdout_eof=True, process_returncode=None)


@pytest.mark.parametrize(("error", "diagnostic"), [
    (RuntimeError("PRIVATE_PROVIDER_DETAIL"), "RuntimeError: review invocation failed (backend_failure)."),
    (CodexLifecycleError(_REAP_FAILURE),
     "Codex process_reap timeout after 5s: actual child exit was not observed; turn.completed received, "
     "stdout EOF observed, child exit not observed; review incomplete. Automatic retry disabled after completion."),
    (_LookalikeLifecycleError(_REAP_FAILURE),
     "_LookalikeLifecycleError: review invocation failed (backend_failure)."),
], ids=["generic", "codex_lifecycle", "lookalike"])
async def test_unadmitted_backend_failure_remains_failed_after_a_shared_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, diagnostic: str,
) -> None:
    """A pipeline-budget stop after a failed batch cannot relabel that attempted work as unattempted.

    Only the exact Codex lifecycle error projects its closed diagnostic; other failures stay generic."""
    review = many_file_review(tmp_path, monkeypatch, count=9, docs_count=0)
    review.backend.fanout_concurrency = 1
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    timer = FakeClock(monotonic_value=clock.monotonic()).install(monkeypatch)
    patch_retry_sleep(monkeypatch, timer)
    attempts = 0

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        nonlocal attempts
        attempts += 1
        timer.advance(61)
        raise error

    assert await review.run(review_profile=_profile_with_pipeline(
        reviewer_investigation_budget_s=480, review_wall_budget_s=60)) == 0
    data = review.load()
    assert attempts == 1
    scope = scopes(data)['python']
    assert scope['status'] == 'failed' and scope['partial_evidence'] is False
    assert scope['reason_codes'] == ['backend_failure', 'host_pipeline_budget_exhaustion']
    assert data['findings'] == []
    assert all(event['metadata']['admitted'] is False for event in stage_ends(review, 'python'))
    assert f'python: {diagnostic}\nReview limit exhausted: pipeline_budget_exceeded' in data['review_warnings']


async def test_stage_builders_preserve_transport_and_bounded_semantic_handoffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir) -> None:
    """Custom builders keep exact-path transport, and a long closed conclusion is clipped in later handoffs."""
    ext_dir.write_module(
        "import json\n"
        "from daydream.deep.prompts import build_per_stack_prompt, build_structural_prompt\n"
        "def wrap(builder, kw):\n"
        "    state = kw['review_stage']\n"
        "    assert isinstance(' '.join(state['notes']), str)\n"
        "    assert kw['files'] == state['assigned_files']\n"
        "    return builder(**kw) + '\\nCUSTOM_UTF8=é🙂\\nBuilder stage arguments:\\n' + json.dumps({\n"
        "        'files': kw['files'], 'diff': kw.get('inline_diff'), 'stage': state['stage']})\n"
        "def register(r):\n"
        "    r.override_prompt('per-stack', lambda **kw: wrap(build_per_stack_prompt, kw))\n"
        "    r.override_prompt('structural', lambda **kw: wrap(build_structural_prompt, kw))\n"
    )
    review = many_file_review(tmp_path, monkeypatch, count=9,
                             feature_message='fix: SETTLED_UNRELATED_HISTORY\n\nDaydream-Run: previous-fixture')
    later_assignments: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        call = review.backend.calls[-1]
        prose = call['prompt'].split('Host review stage:\n')[0]
        assert set(call['output_schema']['properties']) == {
            'targets', 'notes', 'supporting_files', 'candidates', 'contradictions'}
        # Non-Pi backends select sources through the strict transport; only the host builds canonical refs.
        assert 'supporting_files' in call['output_schema']['required']
        assert call['output_schema']['properties']['targets']['items']['properties'][
            'evidence_refs'] == SOURCE_SELECTORS_SCHEMA
        assert stage['submission_transport'] == 'final_output' and 'source_refs' not in stage
        assert all('source_refs' not in part for part in stage['assignment_parts'])
        assert 'rejects the whole stage with no correction attempt' in call['review_instructions']
        assert 'correct the submission in this conversation' not in call['review_instructions']
        if output['targets']:
            assert Draft202012Validator(call['output_schema']).is_valid(output)
        assert 'with issues' not in prose
        assert 'AI-generated code' not in prose
        assert 'Be strict' not in prose
        assert 'Host review stage budget fields are dispatch-time snapshots:' in prose
        assert 'remaining_tool_calls is the reviewer' in prose
        assert 'hard_tool_call_allowance is the current invocation allocation' in prose
        assert 'advisory_tool_call_target guides pace' in prose
        assert stage['hard_tool_call_allowance'] <= stage['remaining_tool_calls']
        assert stage['advisory_tool_call_target'] <= stage['hard_tool_call_allowance']
        assert 'Error Handling Semantics (QUAL-04)' in prose
        assert 'Confidence and Convention Rules' in prose
        assert 'HIGH: directly verified by the supplied change, ordinary investigation, or Exploration Context' in prose
        if stage['stage'] == 'first_pass':
            assert 'SETTLED_UNRELATED_HISTORY' in prose
            assert len(stage['assigned_files']) <= 4
            assert stage['context_transport'] == 'exact_paths'
            pointers = _sanctioned_inputs(call['prompt'])
            assert set(pointers) == set(stage['context_inputs'])
            for label in ('diff', 'hunk-index'):
                assert pointers[label].is_file() and pointers[label].read_text()
            assigned_line = next(line for line in prose.splitlines() if 'Assigned files:' in line)
            assert assigned_line.split('Assigned files:', 1)[1].strip() == ', '.join(stage['assigned_files'])
            assert stage['candidates'] == []
            if 'api.py' in stage['assigned_files']:
                confirmed = candidate(disposition='confirmed', finding=record())
                confirmed['grounds'] += ' Additional explanation: ' + 'é' * 400
                rejected = candidate(disposition='rejected',
                                     grounds='The greeting is allowed by the updated consumer contract.')
                output['candidates'] = [confirmed, rejected] + [candidate() for _ in range(9)]
            if stage['progress']:
                later_assignments.append(stage)
                decisions = stage['closed_decisions']
                assert [(decision['candidate_id'], decision['file'], decision['line'], decision['disposition'])
                        for decision in decisions] == [('python:candidate:1', 'api.py', 2, 'confirmed'),
                                                      ('python:candidate:2', 'api.py', 2, 'rejected')]
                assert decisions[0]['conclusion'].startswith('Grounded defect: ' + candidate()['grounds'])
                assert len(decisions[0]['conclusion'].encode()) <= 512
                assert decisions[0]['conclusion'].endswith('[conclusion clipped]')
                assert decisions[1]['conclusion'] == 'The greeting is allowed by the updated consumer contract.'
                assert all('evidence_references' not in decision for decision in decisions)
                assert all(decision['judgment_origin'] == 'reviewer' for decision in decisions)
                assert stage['candidates'] == []
                instruction = review.backend.calls[-1]['review_instructions']
                assert 'The greeting is allowed by the updated consumer contract.' in instruction
                assert 'Host-retained receipt authority' not in instruction
                policy = ' '.join(instruction.split())
                assert 'Host admission validates the submission contract, not the correctness' in policy
                assert 'Source evidence can disprove the original reasoning without a changed premise' in policy
        else:
            assert stage['stage'] == 'triage'
            assert 'SETTLED_UNRELATED_HISTORY' not in prose
            assert [item['candidate_id'] for item in stage['candidates']] == stage['assigned_candidate_ids']
            assert all(item['disposition'] == 'open' for item in stage['candidates'])
            assert 'Assigned files:' not in prose
            assert stage['assigned_files'] == ['api.py']
            assert all(item['grounds'] and item['trigger'] and item['consequence'] for item in stage['candidates'])
            assert not any(label.startswith('source-') for label in stage['context_inputs'])
            assert 'Sanctioned phase inputs (read only these exact files):' not in call['prompt']
            output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]

    await review.finish('python', findings=('Grounded defect',))
    assert len(later_assignments) == 2
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    assert len([stage for stage in stages if stage['stage'] == 'first_pass']) > 1
    assert [stage['stage'] for stage in stages].count('triage') == 2
    assert [len(stage['assigned_candidate_ids']) for stage in stages if stage['stage'] == 'triage'] == [8, 1]
    # Custom composition and transport expansion happen before the measured host boundary.
    assert_host_dispatch_measurements(review)
    assert all('CUSTOM_UTF8=é🙂' in call['prompt'] for call in review.backend.calls
               if (stage := review_stage_state(call['prompt'])) is not None
               and stage['scope_id'] in {'python', 'structure'})


async def test_stage_preparation_cannot_renew_the_original_invocation_wall_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir,
) -> None:
    """An extension's preparation time spends the same invocation cap as dispatched investigation."""
    review = many_file_review(tmp_path, monkeypatch, count=5)
    review.backend.fanout_concurrency = 1
    elapsed = tmp_path / 'preparation-elapsed.txt'
    elapsed.write_text('0')
    timer = FakeClock(monotonic_value=clock.monotonic())
    monkeypatch.setattr(clock, 'monotonic', lambda: timer.monotonic() + float(elapsed.read_text()))
    from daydream.review_investigation import ReviewInvestigation

    real_run = ReviewInvestigation.run

    async def explicitly_bounded(self: ReviewInvestigation, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop('wall_budget_s', None)
        kwargs['wall_budget_s'] = 3600
        return await real_run(self, *args, **kwargs)

    monkeypatch.setattr(ReviewInvestigation, 'run', explicitly_bounded)
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream.deep.prompts import build_per_stack_prompt\n"
        "def scoped(**kw):\n"
        "    if kw['review_stage']['scope_id'] == 'python':\n"
        f"        elapsed = Path({str(elapsed)!r})\n"
        "        elapsed.write_text(str(float(elapsed.read_text()) + 2000))\n"
        "    return build_per_stack_prompt(**kw)\n"
        "def register(r): r.override_prompt('per-stack', scoped)\n"
    )

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        assert stage['progress'] == [], 'A later stage dispatched after preparation spent the original cap'
        assert 'hard wall allowance: 1600 seconds' in review.backend.calls[-1]['prompt']
        yield from read_events(review.repo, 'api.py', event_id='admitted-read')
        output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    data = await review.finish('python', reason='host_wall_budget_exhaustion', findings=('Grounded defect',),
                              review_profile=_profile_with_pipeline(
                                  reviewer_investigation_budget_s=10000, review_wall_budget_s=10000))
    assert float(elapsed.read_text()) == 4000
    assert len([stage for stage in review.backend.stages if stage['scope_id'] == 'python']) == 1
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    for event in events:
        assert event['metadata']['investigation_effective_allowance_s'] == pytest.approx(3600, rel=0, abs=0.01)
        assert event['metadata']['investigation_limit'] == 'invocation_wall_budget'
    assert events[-1]['metadata']['retained_findings'] == 1
    assert scopes(data)['python']['partial_evidence'] is True
    saved = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert 'wall_budget_exceeded' in saved['diagnostics']['scopes']['python']


async def test_runner_cancellation_closes_stream_and_does_not_admit_an_aborted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling a whole-stack review closes its provider stream and admits no aborted checkpoint."""
    review = many_file_review(tmp_path, monkeypatch, count=10, docs_count=0)
    checkpoint_emitted = asyncio.Event()
    blocked_scopes: set[str] = set()
    closed_scopes: set[str] = set()
    expected = {'python'}

    class CancellationBackend(StagedBackend):
        cancel_done = False

        async def cancel(self) -> None:
            await asyncio.sleep(0)
            self.cancel_done = True

    backend = CancellationBackend(review.repo)
    backend.fanout_concurrency = 2
    review.backend = backend

    async def response(stage: dict[str, Any], output: dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        scope_id = stage['scope_id']
        if scope_id not in expected:
            return
        path = stage['assigned_files'][0]
        line = 2 if path == 'api.py' else 1
        finding = dict(record(line), file=path, evidence=f'{path}:{line}')
        if stage['progress']:
            # The settled first stage's finding is carried into this later request.
            assert len(stage['closed_decisions']) == 1
            assert stage['closed_decisions'][0]['conclusion'].startswith('Grounded defect: ')
            try:
                for event in read_events(review.repo, path, event_id=f'{scope_id}-cancelled-read'):
                    yield event
                output['candidates'] = [candidate(disposition='confirmed', file=path, line=line,
                    grounds=finding['evidence'], finding=dict(finding, description='Aborted checkpoint'))]
                yield ResultEvent(structured_output=output, continuation=None)
                blocked_scopes.add(scope_id)
                if blocked_scopes == expected:
                    checkpoint_emitted.set()
                await asyncio.Event().wait()
            finally:
                closed_scopes.add(scope_id)
            return
        for assigned in stage['assigned_files']:
            for event in read_events(review.repo, assigned, event_id=f'{scope_id}-admitted-{assigned}'):
                yield event
        output['candidates'] = [candidate(disposition='confirmed', file=path, line=line,
                                         grounds=finding['evidence'], finding=finding)]

    backend.stage_response = response
    task = asyncio.create_task(review.run(deep_shard_enabled=False,
                                          review_cache_enabled=True))
    try:
        await asyncio.wait_for(checkpoint_emitted.wait(), timeout=60)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert closed_scopes == expected and backend.cancel_done
    for scope_id in expected:
        completed, interrupted = stage_ends(review, scope_id)
        assert completed['metadata']['admitted'] is True
        assert interrupted['metadata']['admitted'] is False
        assert interrupted['metadata']['failure_class'] == 'cancellation'
        assert interrupted['metadata']['stop_reason'] == 'interruption'
        assert interrupted['metadata']['retained_findings'] == completed['metadata']['retained_findings'] == 1
        assert interrupted['metadata']['observed_tool_starts'] == 5
    assert not review.output.exists()
    coverage_paths = list(review.repo.glob('.daydream/**/review-coverage.json'))
    for coverage_path in coverage_paths:
        coverage = json.loads(coverage_path.read_text())
        assert not any(scope['scope_id'] in expected and scope['status'] == 'complete'
                       for scope in coverage['stack_outcomes'])
    # Cancellation must not publish any aborted scope as a reusable complete result.
    manifests = list(review.repo.glob('.daydream/review-cache/entries/*/manifest.json'))
    assert all(json.loads(path.read_text())['unit'] not in {f'shard:{scope}' for scope in expected}
               for path in manifests)


def test_assessment_rejects_domain_faults_and_irreparable_native_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each domain rejection keeps prior state; native capacity failures stop instead of rewriting history.

    Runner journeys reach only the faults a scripted provider happens to submit;
    this owner check proves every irreparable bound without a review run.
    """
    from daydream import review_result
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import StackAssignment
    from daydream.host_tools import HostToolFatal
    from daydream.review_budget import ReviewLimits
    from daydream.review_investigation import HANDOFF_MAX_BYTES, HANDOFF_MAX_ITEMS, ReviewInvestigation
    from tests.harness.review_result import review_coverage

    coverage = review_coverage(scope_ids=('python',), files=('api.py',))
    part = {'target_id': 'api.py', 'file': 'api.py', 'source_refs': [], 'source_options': []}

    def fresh(batches: list[list[dict[str, Any]]] | None = None, *, scope_id: str = 'python',
              limits: ReviewLimits | None = None) -> ReviewInvestigation:
        return ReviewInvestigation(StackAssignment(scope_id, ['api.py']), {}, coverage.revision.to_dict(),
                                   assignment_batches=batches or [[part]], coverage=coverage, limits=limits)

    judgment = {'trigger': 'x', 'consequence': 'x', 'grounds': 'x', 'disposition': 'open', 'candidate_id': ''}
    reviewed = {'target_id': 'api.py', 'status': 'reviewed', 'reason': ''}
    valid = {'targets': [reviewed], 'candidates': [], 'notes': '', 'contradictions': []}
    new = {**judgment, 'file': 'api.py'}
    rejected = [
        ({**valid, 'targets': [{**reviewed, 'status': 'not_reviewed', 'reason': ' '}]}, 'nonblank explanation'),
        ({**valid, 'targets': [reviewed, reviewed]}, 'duplicate assigned target'),
        ({**valid, 'contradictions': ['never-closed']}, 'previously closed_candidate_ids'),
        ({**valid, 'candidates': [{**new, 'candidate_id': 'c1'}]}, 'no host identity'),
        ({**valid, 'candidates': [{**new, 'disposition': 'confirmed', 'finding': None}]}, 'complete finding'),
    ]
    for output, diagnostic in rejected:
        investigation = fresh()
        assert investigation._assessment(output, 'first_pass', ['api.py'], [], files=['api.py']) == (
            None, 'malformed_output')
        assert diagnostic in (investigation.assertion_failure_diagnostic or '')
    blank = fresh()
    assert blank._assessment({**valid, 'candidates': [{**new, 'grounds': ' '}]}, 'first_pass', ['api.py'], [],
                             files=['api.py']) == (None, 'evidence_incomplete')
    assert blank.assertion_failure_class == 'admission_failure'
    # Triage judges exactly its assigned open candidates and keeps first-pass progress.
    triaged = fresh()
    triaged.handoff['candidates'] = [{**new, 'candidate_id': 'c1', 'line': 1, 'finding': None}]
    for decisions, diagnostic in (([{**new, 'candidate_id': 'c9'}], 'outside assigned_candidate_ids'),
                                  ([], 'missing or duplicate assigned_candidate_ids')):
        assert triaged._assessment({**valid, 'targets': [], 'candidates': decisions}, 'triage', [], ['c1'],
                                   files=[]) == (None, 'malformed_output')
        assert diagnostic in (triaged.assertion_failure_diagnostic or '')
    plan, reason = triaged._assessment({**valid, 'targets': [], 'candidates': [
        {**new, 'candidate_id': 'c1', 'disposition': 'rejected'}]}, 'triage', [], ['c1'], files=[])
    assert reason is None and plan is not None
    assert plan.handoff['progress'] == triaged.handoff['progress']
    assert plan.handoff['candidates'][0]['disposition'] == 'rejected'
    # Shape failures permit only independently proven bounds; otherwise the result is unassessable.
    investigation = fresh()
    unshaped = {'targets': [reviewed], 'candidates': [judgment]}
    assert investigation._assessment(unshaped, 'first_pass', ['api.py'], [], files=[]) == (None, None)
    assert investigation._assessment({**unshaped, 'candidates': [judgment] * (HANDOFF_MAX_ITEMS + 1)},
                                     'first_pass', ['api.py'], [], files=[]) == (None, 'evidence_incomplete')
    investigation._source_root = tmp_path
    assert investigation._supporting_files({'supporting_files': 'api.py'}) == []
    # Native admissions raise irreparable fatals rather than dropping retained work.
    investigation.handoff['candidates'] = [{**judgment, 'file': 'api.py', 'candidate_id': 'c1',
                                            'disposition': 'rejected', 'finding': None, 'line': 1}]
    with pytest.raises(HostToolFatal) as contradiction:
        investigation._assessment({**valid, 'contradictions': ['c1']}, 'first_pass', ['api.py'], [],
                                  files=['api.py'], admitting=True, native_sources=True)
    assert contradiction.value.code == 'evidence_incomplete'
    investigation = fresh()
    investigation.handoff['notes'] = [{'files': [], 'supporting_files': [], 'text': ''}] * HANDOFF_MAX_ITEMS
    with pytest.raises(HostToolFatal, match='no capacity for the required stage note') as notes:
        investigation._assessment(valid, 'first_pass', ['api.py'], [], files=['api.py'], native_sources=True)
    assert notes.value.code == 'capacity_exhausted'
    assert investigation._assessment(valid, 'first_pass', ['api.py'], [], files=['api.py'])[1] == (
        'evidence_incomplete')
    investigation = fresh()
    pending = {**judgment, 'file': 'api.py', 'candidate_id': 'c2', 'line': 1, 'finding': None}
    investigation.handoff.update(candidates=[pending], notes=[
        {'files': ['api.py'], 'supporting_files': [], 'text': 'x' * HANDOFF_MAX_BYTES}])
    with pytest.raises(HostToolFatal, match='cannot fit the required stage context') as triage:
        investigation._assessment({**valid, 'targets': [], 'candidates': [{**pending, 'disposition': 'unresolved'}]},
                                  'triage', [], ['c2'], files=[], native_sources=True)
    assert triage.value.code == 'capacity_exhausted'
    # A later batch is admitted only while shared budgets remain; exhaustion is named, not retried.
    batches = fresh()
    from daydream import clock
    from daydream.run_context import InteractionPolicy, RunContext

    context = RunContext(InteractionPolicy())
    assert batches._continuation_admitted({'run_context': context}) is True
    for _ in range(100):
        context.outage_circuit.record_failure(clock.monotonic())
    assert batches._continuation_admitted({'run_context': context}) is False
    assert (batches.reason, batches.failure_class) == ('backend_failure', 'outage_circuit')
    batches.budget.observed_tool_starts = batches.budget.limits.tool_calls
    assert batches._continuation_admitted({}) is False
    assert (batches.reason, batches.failure_class, batches.failure_diagnostic) == (
        'tool_call_budget_exceeded', 'quantitative_exhaustion', 'Review limit exhausted: tool_call_budget_exceeded')
    # Ordinary per-batch spend still reaches every batch of a many-batch scope; a zero allowance
    # still stops dispatch, and Structure's single integration stage keeps the base allowance.
    many = [[{**part, 'target_id': f'api.py#{index}'}] for index in range(12)]
    scoped = fresh(many)
    for _ in many:
        assert scoped._budget_stop_reason() is None
        scoped.budget.observed_tool_starts += 9
    assert fresh(many, limits=ReviewLimits(tool_calls=0))._budget_stop_reason() == 'tool_call_budget_exceeded'
    assert fresh(many, scope_id=STRUCTURE_STACK_NAME).budget.limits == ReviewLimits()
    monkeypatch.setattr(review_result, 'RUN_TARGET_DECISIONS_MAX_BYTES', 1)
    with pytest.raises(HostToolFatal, match='required target IDs') as targets:
        fresh()._assessment(valid, 'first_pass', ['api.py'], [], files=['api.py'], native_sources=True)
    assert targets.value.code == 'capacity_exhausted'


def test_bounded_review_context_and_tool_evidence_drop_oldest_whole_entries() -> None:
    """Source notes are clipped and redacted; rationale and tool context drop whole entries and say so.

    Runner reports stay far below these bounds; without this owner check an
    unbounded note list or tool-start map would only fail on real oversized runs.
    """
    from daydream.deep.detection import StackAssignment
    from daydream.review_evidence import ReviewEvidence
    from daydream.review_investigation import ReviewInvestigation
    from daydream.review_rationale import REPORT_INPUT_MAX_BYTES, bounded_scope_review_context_bytes
    from tests.harness.review_result import review_coverage

    # Persisted source notes are reviewer assertions: bounded, clipped and credential-redacted.
    coverage = review_coverage(scope_ids=('python',), files=('api.py',))
    investigation = ReviewInvestigation(StackAssignment('python', ['api.py']), {}, coverage.revision.to_dict(),
                                        assignment_batches=[], coverage=coverage)
    investigation.handoff['notes'] = [{
        'text': 'Source-only assertion ghp_1234567890 {"token": "private-value"} ' + 'x' * 700,
        'files': [f'{index}/' + 'é' * 300 for index in range(40)], 'supporting_files': [],
        'origin': {'scope_id': 'python', 'unexpected': True}} for _ in range(9)]
    context = investigation.review_context()
    assert len(context['notes']) == 8 and context['notes_omitted'] == 1
    for note in context['notes']:
        assert len(note['text'].encode()) <= 512 and note['text'].endswith('[note clipped]')
        assert 'ghp_1234567890' not in note['text'] and 'private-value' not in note['text']
        assert '[REDACTED_API_KEY]' in note['text']
        assert len(note['files']) == 32 and all(len(path.encode()) <= 512 for path in note['files'])
        assert note['origin'] is None, 'malformed attribution stays null instead of becoming host provenance'

    notes = [{'text': str(index) * (REPORT_INPUT_MAX_BYTES // 3)} for index in range(4)]
    envelope: dict[str, Any] = {'scope_id': 'python', 'review_context': {'notes': notes, 'notes_omitted': 0}}
    bounded = json.loads(bounded_scope_review_context_bytes(envelope))
    assert bounded['review_context']['notes'] == notes[:2] and bounded['review_context']['notes_omitted'] == 2
    assert envelope['review_context']['notes'] == notes, 'the caller envelope is not mutated'
    with pytest.raises(ValueError, match='exceeds report input budget'):
        bounded_scope_review_context_bytes({'scope_id': 'x' * REPORT_INPUT_MAX_BYTES,
                                            'review_context': {'notes': [], 'notes_omitted': 0}})
    evidence = ReviewEvidence(None)
    for index in range(65):
        evidence.observe(ToolStartEvent(id=f'read-{index}', name='Read', input={'file_path': 'api.py'}))
    assert len(evidence.pending) == 64 and 'read-0' not in evidence.pending
    assert evidence.omitted == 1 and evidence.clipped
