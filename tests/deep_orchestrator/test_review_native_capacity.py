"""Production-sized assignment transport with a late cross-file judgment."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, PiRequestConfig, RequestEvent, ToolResultEvent, ToolStartEvent
from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, stage_ends
from tests.harness.git_helpers import seed_feature_branch


def _workload(wired: bool) -> tuple[dict[str, str], dict[str, str]]:
    before, after = {}, {}
    for index in range(41):
        path = f'package_{index // 5}/module_{index:02}.py'
        old, new = '', ''
        sections = 8 if index == 13 else 4 if index < 13 else 0
        for section in range(sections):
            stable = ''.join(f'# Unchanged enclosing context {section}:{line}\n' for line in range(18))
            old += stable + f'WINDOW_{section} = 0\n' + ''.join(
                f'# Old contract {section}:{line:02} ' + 'o' * 72 + '\n' for line in range(18))
            new += stable + f'WINDOW_{section} = 1\n' + ''.join(
                f'# New contract {section}:{line:02} ' + 'n' * 72 + '\n' for line in range(18))
        if not sections:
            old, new = 'VALUE = 0\n', 'VALUE = 1\n'
        if index == 0:
            old += 'def parse_flags(args):\n    return {}\n'
            new += 'def parse_flags(args):\n    return {"dry_run": "--dry-run" in args}\n'
        if index == 40:
            old += 'def build_request(options):\n    return {}\n'
            new += ("from package_0.module_00 import parse_flags\ndef build_request(options):\n"
                    + ('    return {"dry_run": options["dry_run"]}\n' if wired else '    return {}\n'))
        before[path], after[path] = old, new
    before['00_change-guide.md'] = '# Request contract\nDry-run must propagate to the request.\n'
    after['00_change-guide.md'] = before['00_change-guide.md'] + ''.join(
        f'Change context {index:04}: ' + 'c' * 100 + '\n' for index in range(800))
    return before, after


@pytest.mark.parametrize('wired', [False, True], ids=['late-cross-file-defect', 'correctly-wired-clean'])
async def test_late_cross_file_judgment_fits_cumulative_native_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wired: bool) -> None:
    repo = tmp_path / 'production_sized_work'
    before, after = _workload(wired)
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    late = 'package_8/module_40.py'
    assigned_files: set[str] = set()
    assignment_units = 0

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal assignment_units
        yield RequestEvent(prompt=review.backend.calls[-1]['prompt'],
                           output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        assert stage['remaining_tool_calls'] <= 192
        assigned_files.update(stage['assigned_files'])
        assignment_units += len(stage['assignment_parts'])
        output['notes'] = 'The late request builder is assessed against the changed dry-run contract.'
        if not wired and late in stage['assigned_files']:
            finding = {'id': 1, 'file': late, 'line': 4, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': 'Late request builder drops the parsed dry-run flag',
                       'rationale': 'The changed request builder omits the dry-run value.',
                       'evidence': f'{late}:4 omits dry_run required by the changed request contract.'}
            output['candidates'] = [{'candidate_id': '', 'file': late, 'line': 4,
                                     'trigger': 'Parse --dry-run and build the request',
                                     'consequence': 'The request omits dry_run', 'grounds': finding['evidence'],
                                     'disposition': 'confirmed', 'finding': finding}]
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    data = await review.finish('python', deep_shard_enabled=False,
                               findings=() if wired else ('Late request builder drops the parsed dry-run flag',))
    assert len(scopes(data)['python']['files']) == 41 and late in scopes(data)['python']['files']
    assert assigned_files == set(after for after in before if after.endswith('.py'))
    assert assignment_units == 48
    metadata = [phase['metadata'] for phase in stage_ends(review, 'python')]
    assert metadata[0]['hard_tool_call_allowance'] == 192
    assert metadata[-1]['observed_tool_starts'] <= 192
    assert sum(phase['submission_starts'] for phase in metadata) == len(metadata)
    assert all(phase['admitted'] for phase in metadata)
    assert len((repo / '.daydream/diff.patch').read_bytes()) > 256 * 1024
