"""Individual frozen source commands remain admissible through the real runner."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.config_file import DaydreamFileConfig
from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, stage_ends
from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend
from tests.harness.git_helpers import seed_feature_branch, tracked_source_state


class BoundCommandBackend(NativeSourceBackend):
    """Expose the existing inline full-SHA source access at the provider seam."""

    supports_source_recipe = False
    read_only_disposable_clone = True


async def test_individual_bound_commands_preserve_literal_paths_findings_and_folded_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'bound_commands'
    route = 'user.$username.tsx'
    before = {
        route: 'export const profileNavigation = "user.$username";\n',
        'App.tsx': 'export const actualPath = "/profile/alex";\n',
        'Profile.tsx': 'export const expectedPath = "/profile/alex";\n',
        'Routes.tsx': 'export const profileRoute = "/profile/$username";\n',
        'README.md': '# Profile navigation\n',
    }
    after = {path: body + '// Changed profile boundary.\n' for path, body in before.items()}
    after['App.tsx'] = 'export const actualPath = "/profile/";\n'
    seed_feature_branch(repo, base=before, feature=after)
    original = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = BoundCommandBackend(repo)
    review.backend.sandbox = True
    commands: dict[str, list[str]] = {'react': [], 'structure': []}
    decisions: list[tuple[str, str, str]] = []
    title = 'Profile navigation loses the username required by the route'

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        scope = stage['scope_id']
        if scope not in commands:
            return
        call = review.backend.calls[-1]
        windows = [window for window in stage['source_access'] if window['side'] == 'after'
                   and window['file'].endswith('.tsx')]
        assert len(windows) == 4
        supplied = [window['access']['arguments']['cmd'] for window in windows]
        for window, command in zip(windows, supplied, strict=True):
            assert window['access']['tool'] == 'exec_command'
            assert shlex.split(command) == ['git', 'show', f"{review.pr.head_sha}:{window['source_path']}"]
        literal, = [command for window, command in zip(windows, supplied, strict=True) if window['file'] == route]
        assert literal.endswith(f"'{review.pr.head_sha}:{route}'")

        # Model the observed batching choice at the external provider boundary.
        # Explicit guidance in both the invocation and persistent instructions
        # causes separate results; otherwise successful compound output still
        # cannot establish the independent frozen-source association.
        separated = all('one bound source read per tool result' in instructions
                        and 'preserve supplied shell quoting' in instructions
                        for instructions in (call['prompt'], call.get('review_instructions', '')))
        issued = supplied if separated else [' && '.join(supplied)]
        captured = ''
        for index, command in enumerate(issued):
            commands[scope].append(command)
            event_id = f'bound-{index}'
            yield ToolStartEvent(id=event_id, name='exec_command', input={'cmd': command})
            executed = subprocess.run(command, shell=True, cwd=call['cwd'], capture_output=True, text=True)
            assert executed.returncode == 0, executed.stderr
            captured += executed.stdout
            yield ToolResultEvent(id=event_id, output=executed.stdout, is_error=False)
        actual = re.search(r'export const actualPath = ("[^"]+");', captured)
        expected = re.search(r'export const expectedPath = ("[^"]+");', captured)
        assert actual is not None and expected is not None
        values = json.loads(actual[1]), json.loads(expected[1])
        decisions.append((scope, *values))
        output['notes'] = 'Navigation was compared with the concrete profile route contract.'
        if scope == 'react' and values[0] != values[1]:
            finding = {'id': 1, 'file': 'App.tsx', 'line': 1, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': title, 'rationale': 'Navigation omits the username segment.',
                       'evidence': 'App.tsx:1 supplies /profile/; Profile.tsx:1 requires /profile/alex.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'App.tsx', 'line': 1,
                                     'trigger': 'Navigate to the profile for alex',
                                     'consequence': 'The destination cannot identify the profile',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    data = await review.finish('react', findings=(title,), file_config=DaydreamFileConfig(supervisor='off'))
    assert data['terminal_result']['analysis_state'] == 'complete'
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == 'complete'
    assert set(decisions) == {('react', '/profile/', '/profile/alex'), ('structure', '/profile/', '/profile/alex')}
    for scope in commands:
        assert len(commands[scope]) == 4
        terminal, = stage_ends(review, scope)
        assert terminal['metadata']['admitted']
        assert terminal['metadata']['fresh_source_reads'] == 4
        assert terminal['metadata']['attempt'] == 1
        assert terminal['metadata']['remaining_tool_calls'] == 44
    assert tracked_source_state(repo) == original
