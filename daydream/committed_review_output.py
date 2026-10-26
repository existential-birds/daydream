"""Render and install the sealed committed-review result through its run owner.

The required JSON pairs findings and typed coverage in one atomic installation.
Presentation has a separate outcome and cannot grant publication authority.
"""
from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

from daydream.artifact_visibility import ArtifactSession
from daydream.committed_review_model import validate_committed_result
from daydream.json_utils import atomic_write_bytes, atomic_write_json


def _code(value: object) -> str:
    return '<code>' + html.escape(str(value)) + '</code>'


def render_committed_review(payload: Mapping[str, Any]) -> str:
    """Present retained claims without treating an empty list as completed work."""
    terminal = payload['terminal_result']
    revision = payload['analyzed_revision']
    lines = [
        '# Review', '', '## Review outcome', '',
        f"Analysis: {_code(terminal['analysis_state'])}.",
        f"Run: {_code(payload['run_id'])}; planned work: {_code(payload['planned_work_id'])}.",
        f"Captured head: {_code(revision['head_sha'])}; merge base: {_code(revision['merge_base_sha'])}.",
        f"Captured diff: {_code(revision['diff_key'])}.",
        'Source correspondence does not independently prove reading, understanding, or executed checks.',
    ]
    if terminal['reason_codes']:
        lines.append('Unfinished work: ' + ', '.join(_code(reason) for reason in terminal['reason_codes']) + '.')
    lines.extend(['', '## Findings', ''])
    confirmed = [claim for claim in payload['claims'] if claim['disposition'] == 'confirmed']
    if not confirmed:
        lines.append('No confirmed findings.')
    for disposition, heading in (
        ('confirmed', None), ('unresolved', 'Unresolved concerns'), ('rejected', 'Rejected concerns'),
    ):
        claims = [claim for claim in payload['claims'] if claim['disposition'] == disposition]
        if heading is not None and claims:
            lines.extend(['', '## ' + heading, ''])
        for claim in claims:
            semantic = claim['payload']
            location = _code(semantic['file'])
            if semantic['line'] is not None:
                location += ':' + _code(semantic['line'])
            lines.extend([
                f"### {claim['id']}. {claim['title']}", '',
                f"Location: {location}; "
                f"claim {_code(claim['uid'])}, revision {_code(claim['revision'])}.", '',
                semantic['claim'], '',
                '**Trigger:** ' + semantic['trigger'], '',
                '**Consequence:** ' + semantic['consequence'], '',
                '**Supporting grounds:** ' + semantic['grounds'],
            ])
            for challenge in claim['challenge_evidence']:
                lines.extend([
                    '', f"Challenge {_code(challenge['verdict'])} for revision {_code(challenge['revision'])}:", '',
                    challenge['grounds'],
                ])
    lines.extend(['', '## Native execution', ''])
    for execution in payload['execution']:
        lines.append(
            f"- {_code(execution['assignment'])}: accepted {_code(execution['accepted'])}; "
            f"settled {_code(execution['settled'])}; cleanup complete {_code(execution['cleanup_complete'])}."
        )
        if execution['reason']:
            lines.append('  ' + str(execution['reason']))
    return '\n'.join(lines) + '\n'


def write_committed_review_output(session: ArtifactSession, payload: dict[str, Any]) -> None:
    """Install one validated findings/coverage bundle after staging its report.

    The session owns isolation, conflict preservation and final publication.
    Writing JSON last leaves no successful current bundle if report staging fails.
    """
    validate_committed_result(payload)
    if payload['run_id'] != session.layout.session_id:
        raise ValueError('committed review output differs from its artifact run')
    report = render_committed_review(payload).encode('utf-8')
    atomic_write_bytes(session.review_output, report)
    atomic_write_json(session.daydream_dir / 'committed-review.json', payload, trailing_newline=True)
