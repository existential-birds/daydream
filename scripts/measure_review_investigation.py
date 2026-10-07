"""Replay bounded review control through runner.run with a lossy external backend.

Run identical fixtures against an isolated source tree and the current checkout:
  uv run python scripts/measure_review_investigation.py --source /tmp/before --output /tmp/before.json
  uv run python scripts/measure_review_investigation.py --source . --output /tmp/after.json
This measures controller enforcement, not model quality or universal budget sufficiency.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def fixture_sources() -> tuple[dict[str, str], dict[str, str]]:
    base = {f'module_{index:02}.py': 'VALUE = 0\n' for index in range(17)}
    changed = {name: 'VALUE = 1\n' + ''.join(f'# changed line {i}\n' for i in range(70)) for name in base}
    return base, changed


async def replay(source: Path) -> dict[str, Any]:
    source = source.resolve()
    source_digest = hashlib.sha256()
    for path in sorted((source / 'daydream').rglob('*.py')):
        source_digest.update(str(path.relative_to(source)).encode())
        source_digest.update(path.read_bytes())
    sys.path.insert(0, str(source))
    import pytest

    from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
    from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
    from tests.deep_orchestrator.test_review_completion import ReviewRun, scopes
    from tests.harness.git_helpers import seed_feature_branch

    with (tempfile.TemporaryDirectory(prefix='daydream-stage-replay-') as directory,
          pytest.MonkeyPatch.context() as patch):
        root = Path(directory).resolve()
        repo = root / 'repo'
        base, changed = fixture_sources()
        seed_feature_branch(repo, base=base, feature=changed)
        review = ReviewRun(repo, root, patch)
        requests: list[dict[str, Any]] = []
        reads: dict[str, list[str]] = {}

        class LossyBackend(EmptyReviewBackend):
            """Each invocation starts over; only host-supplied state survives."""

            async def execute(self, cwd, prompt, *args, **kwargs):
                stage = None
                if 'Host review stage:\n' in prompt:
                    stage = json.JSONDecoder().raw_decode(prompt.split('Host review stage:\n', 1)[1])[0]
                    scope = stage['scope_id']
                    paths = stage['assigned_target_ids'] if stage['stage'] == 'first_pass' else sorted(changed)[:2]
                    if stage['stage'] == 'triage':
                        paths = [sorted(changed)[0]]
                elif 'you are reviewing the python stack' in prompt.lower():
                    scope, paths = 'python', sorted(changed) * 5
                elif 'you are the structural reviewer' in prompt.lower():
                    scope, paths = 'structure', sorted(changed) * 5
                else:
                    async for event in super().execute(cwd, prompt, *args, **kwargs):
                        yield event
                    return
                request = {'scope_id': scope, 'stage': stage['stage'] if stage else 'unbounded_discovery',
                           'remaining_tool_calls': stage['remaining_tool_calls'] if stage else None,
                           'tool_call_allowance': stage['tool_call_allowance'] if stage else None,
                           'progress_count': len(stage['progress']) if stage else 0,
                           'candidate_dispositions': [c['disposition'] for c in stage['candidates']] if stage else [],
                           'observed_tool_starts': 0, 'no_tools': bool(kwargs.get('no_tools'))}
                requests.append(request)
                self.calls.append({'prompt': prompt, **kwargs})
                for index, path in enumerate(paths):
                    request['observed_tool_starts'] += 1
                    reads.setdefault(scope, []).append(path)
                    yield ToolStartEvent(id=f'read-{index}', name='Read', input={'file_path': path})
                    yield ToolResultEvent(id=f'read-{index}', output=(repo / path).read_text(), is_error=False)
                candidates = []
                if stage:
                    if scope == 'structure' and stage['stage'] == 'first_pass' and not stage['progress']:
                        candidates = [{'candidate_id': '', 'file': sorted(changed)[0], 'line': 1,
                                       'trigger': 'Import changed constant', 'consequence': 'Potential caller drift',
                                       'grounds': 'The constant is now 1; targeted caller checks remain.',
                                       'disposition': 'open', 'finding': None}]
                        candidates.insert(0, dict(candidates[0], disposition='rejected'))
                    elif stage['stage'] == 'triage':
                        candidates = [dict(c, disposition='rejected') for c in stage['candidates']
                                      if c['disposition'] == 'open']
                    output = {'targets': [{'target_id': t, 'status': 'reviewed', 'reason': ''}
                                          for t in stage['assigned_target_ids']],
                              'notes': 'Assigned constant changes inspected.', 'candidates': candidates,
                              'contradictions': []}
                else:
                    output = {'issues': []}
                yield ResultEvent(structured_output=output, continuation=None)

        review.backend = LossyBackend(repo)
        exit_code = await review.run()
        artifact = review.load()
        return {'exit_code': exit_code, 'fixture': '17-file-lossy-context',
                'source_sha256': source_digest.hexdigest(),
                'replay_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fixture_sha256': hashlib.sha256(json.dumps([base, changed], sort_keys=True).encode()).hexdigest(),
                'stage_requests': requests,
                'observed_tool_starts': {scope: len(paths) for scope, paths in reads.items()},
                'repeated_evidence_requests': {scope: sum(n - 1 for n in Counter(paths).values())
                                               for scope, paths in reads.items()},
                'findings': len(artifact['findings']),
                'coverage': {name: value['status'] for name, value in scopes(artifact).items()},
                'interpretation': 'Deterministic controller enforcement only; no model-quality claim.'}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = await replay(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'stage_requests'}))


if __name__ == '__main__':
    asyncio.run(main())
