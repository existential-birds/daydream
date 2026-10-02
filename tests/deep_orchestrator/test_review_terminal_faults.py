"""Production terminal exports after provider and filesystem boundary faults."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream import json_utils, runner
from daydream.backends import AgentEvent, ResultEvent
from daydream.findings import FindingsValidationError, load_findings_artifact
from daydream.phases.review import ReviewOutputError
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.console import collapse_panel_text
from tests.test_deep_orchestrator import _pin_findings_pr, _record


def _survivor() -> dict[str, Any]:
    return _record(description='Surviving sibling defect', file='App.tsx', line=1, severity='medium',
                   confidence='MEDIUM', rationale='Grounded sibling evidence', evidence='App.tsx:1')


def _load(output: Path, head: str) -> dict[str, Any]:
    artifact = load_findings_artifact(output, expected_repo='o/r', expected_pr_number=7, expected_head_sha=head)
    assert artifact.schema_version == 2
    data = json.loads(output.read_text())
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(('fault', 'reason'), [('missing', 'missing_artifact'), ('corrupt', 'malformed_artifact')])
async def test_persisted_scope_fault_preserves_completed_sibling_findings(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, reason: str,
) -> None:
    """Corrupt the actual reviewer write before the real parse stage consumes it."""
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = EmptyReviewBackend(multi_stack_target, review_by_stack={'react': [_survivor()]})
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: backend)
    original_write = Path.write_text
    injected = False
    def faulty_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        nonlocal injected
        if path.name == 'stack-python-records.json' and not injected:
            injected = True
            count = original_write(path, text if fault == 'missing' else '{invalid', *args, **kwargs)
            if fault == 'missing':
                path.unlink()
            return count
        return original_write(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', faulty_write)
    output = tmp_path / 'fault-findings.json'
    code = await runner.run(empty_review_config(multi_stack_target, tmp_path / 'trajectory.json',
                                                findings_out=str(output), pr_number=7))
    assert injected
    assert code == 1
    data = _load(output, pr.head_sha)
    result = data['terminal_result']
    scopes = {scope['scope_id']: scope for scope in result['stack_outcomes']}
    assert set(scopes) == {'python', 'react', 'generic', 'structure'}
    assert scopes['python']['status'] == 'failed'
    assert scopes['python']['reason_codes'] == [reason]
    assert all(s['status'] == 'complete' for name, s in scopes.items() if name != 'python')
    assert result['pipeline_state'] == 'failed'
    assert result['analysis_state'] == 'incomplete'
    assert reason in result['reason_codes']
    assert [f['title'] for f in data['findings']] == ['Surviving sibling defect']
    assert data['findings'][0]['placement'] == 'inline'
    assert not any('cross-stack merge agent' in c['prompt'].lower() for c in backend.calls)


@pytest.mark.parametrize(('fault', 'reason'), [('missing', 'missing_output'), ('malformed', 'malformed_output')])
async def test_required_supervision_invalid_output_cannot_complete_review(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, reason: str,
) -> None:
    class BadSupervisorBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if 'supervisor adjudication' in prompt.lower():
                self.calls.append({'prompt': prompt, 'model': self.model})
                yield ResultEvent(structured_output=None if fault == 'missing' else {'verdicts': [None]},
                                  continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = BadSupervisorBackend(multi_stack_target, review_by_stack={'structure': [_survivor()]})
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: backend)
    output = tmp_path / 'supervision-findings.json'
    with pytest.raises(ReviewOutputError):
        await runner.run(empty_review_config(multi_stack_target, tmp_path / 'trajectory.json',
                                             findings_out=str(output), pr_number=7))
    data = _load(output, pr.head_sha)
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['supervision']['status'] == 'failed'
    assert phases['supervision']['reason_codes'] == [reason]
    assert data['terminal_result']['pipeline_state'] == 'failed'
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    assert [f['title'] for f in data['findings']] == ['Surviving sibling defect']
    assert any('supervisor adjudication' in c['prompt'].lower() for c in backend.calls)


async def test_omitted_supervision_verdict_is_incomplete_evidence(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OmittedSupervisorBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if 'supervisor adjudication' in prompt.lower():
                self.calls.append({'prompt': prompt, 'model': self.model})
                yield ResultEvent(structured_output={'verdicts': []}, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = OmittedSupervisorBackend(multi_stack_target, review_by_stack={'structure': [_survivor()]})
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: backend)
    output = tmp_path / 'omitted-supervision.json'
    assert await runner.run(empty_review_config(multi_stack_target, tmp_path / 'trajectory.json',
                                                findings_out=str(output), pr_number=7)) == 0
    data = _load(output, pr.head_sha)
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['supervision']['status'] == 'incomplete'
    assert phases['supervision']['reason_codes'] == ['evidence_incomplete']
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    assert data['terminal_result']['pipeline_state'] == 'completed'
    assert len(data['findings']) == 1


async def test_terminal_filesystem_failure_preserves_original_provider_error_and_no_current_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    original_error = RuntimeError('intent provider failed first')
    class FailingIntentBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if 'present your understanding concisely' in prompt.lower():
                raise original_error
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = FailingIntentBackend(multi_stack_target)
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: backend)
    original_stage = json_utils._stage_bytes
    injected = False
    def fail_coverage_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal injected
        if path.name == 'review-coverage.json':
            injected = True
            raise OSError('coverage staging failed during finalization')
        return original_stage(path, content, **kwargs)
    monkeypatch.setattr(json_utils, '_stage_bytes', fail_coverage_stage)
    output = tmp_path / 'absent-current-result.json'
    with pytest.raises(RuntimeError, match='intent provider failed first') as raised:
        await runner.run(empty_review_config(multi_stack_target, tmp_path / 'trajectory.json',
                                             findings_out=str(output), pr_number=7))
    assert raised.value is original_error
    assert injected
    assert 'Terminal review finalization failed: OSError' in collapse_panel_text(capsys)
    assert not output.exists()
    with pytest.raises(FindingsValidationError):
        load_findings_artifact(output, expected_repo='o/r', expected_pr_number=7, expected_head_sha=pr.head_sha)


async def test_later_fix_failure_keeps_frozen_review_coverage_complete(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe real coverage writes around the external fix-provider failure."""
    fix_started = False
    captured: list[tuple[bool, bytes]] = []
    original_stage = json_utils._stage_bytes
    def observe_coverage_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        if path.name == 'review-coverage.json':
            captured.append((fix_started, content))
        return original_stage(path, content, **kwargs)
    monkeypatch.setattr(json_utils, '_stage_bytes', observe_coverage_stage)
    class FailedFixBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal fix_started
            if prompt.lower().startswith(('fix this issue', 'fix these')):
                fix_started = True
                self.calls.append({'prompt': prompt, 'model': self.model})
                raise RuntimeError('later fix provider unavailable')
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = FailedFixBackend(multi_stack_target, review_by_stack={'structure': [_survivor()]},
                               forbid_supervise=False)
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: backend)
    code = await runner.run(empty_review_config(multi_stack_target, tmp_path / 'trajectory.json',
                                                output_mode='loop', pr_number=7))
    assert code == 1
    assert fix_started
    assert captured
    assert all(not after_fix for after_fix, _ in captured), 'frozen review cannot be revised by later fix failure'
    final_path = multi_stack_target / '.daydream' / 'deep' / 'review-coverage.json'
    assert final_path.read_bytes() == captured[-1][1]
    saved = json.loads(final_path.read_text())
    assert all(scope['status'] == 'complete' for scope in saved['stack_outcomes'])
    assert all(phase['status'] == 'complete' for phase in saved['phase_outcomes'])
    assert 'pipeline' not in saved['required_phases']


async def test_host_salvage_rejection_exports_failed_projection_and_preserves_phase_error(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host projection rejection cannot hide terminal metadata or replace the phase error."""
    from daydream.phases import findings

    original_error = RuntimeError('supervision provider failed first')
    original_write = Path.write_text
    original_salvage = findings._write_single_stack_merged_items
    merged_path: Path | None = None
    failed_phase = False
    salvage_rejected = False

    def observe_merged_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        nonlocal merged_path
        if path.name == 'merged-items.json':
            merged_path = path
        return original_write(path, text, *args, **kwargs)

    def reject_failed_salvage(*args: Any, **kwargs: Any) -> None:
        nonlocal salvage_rejected
        if failed_phase:
            salvage_rejected = True
            raise ValueError('host rejected surviving record projection')
        original_salvage(*args, **kwargs)

    class FailedSupervisorBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal failed_phase
            if 'supervisor adjudication' in prompt.lower():
                assert merged_path is not None
                assert json.loads(merged_path.read_text())['items']
                merged_path.unlink()
                failed_phase = True
                raise original_error
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = FailedSupervisorBackend(multi_stack_target, review_by_stack={'structure': [_survivor()]})
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(Path, 'write_text', observe_merged_write)
    monkeypatch.setattr(findings, '_write_single_stack_merged_items', reject_failed_salvage)
    output = tmp_path / 'rejected-host-projection.json'
    with pytest.raises(RuntimeError, match='supervision provider failed first') as raised:
        await runner.run(empty_review_config(multi_stack_target, tmp_path / 'trajectory.json',
                                             findings_out=str(output), pr_number=7))
    assert raised.value is original_error
    assert salvage_rejected
    data = _load(output, pr.head_sha)
    result = data['terminal_result']
    assert result['pipeline_state'] == 'failed'
    assert result['analysis_state'] == 'failed'
    assert result['projection_valid'] is False
    assert data['findings'] == []
    assert all(scope['status'] == 'complete' for scope in result['stack_outcomes'])
    phases = {phase['phase']: phase for phase in result['phase_outcomes']}
    assert phases['supervision']['status'] == 'failed'
    assert phases['supervision']['reason_codes'] == ['backend_failure']
    assert phases['findings']['status'] == 'failed'
    assert phases['findings']['reason_codes'] == ['malformed_artifact']
