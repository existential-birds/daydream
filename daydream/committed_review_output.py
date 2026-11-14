"""Render the report and atomically stage its validated findings/coverage bundle."""
from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from daydream.artifact_visibility import ArtifactSession
from daydream.committed_review_model import validate_committed_result
from daydream.json_utils import atomic_write_bytes, atomic_write_json

# The [pipeline] profile keys that raise the allowance behind each exhaustion reason; the wall caps the time.
_ALLOWANCE_KEYS = {'host_wall_budget_exhaustion': ('reviewer_investigation_budget_s', 'review_wall_budget_s'),
                   'host_tool_budget_exhaustion': ('reviewer_tool_call_budget',)}


def _code(value: object) -> str:
    return '<code>' + html.escape(str(value)) + '</code>'


def allowance_sentence(reason_codes: list[str], code: Callable[[object], str] = _code) -> str:
    """Name the profile keys that raise a spent allowance, from sealed reason codes; empty when none was spent."""
    keys = [code(key) for reason, names in _ALLOWANCE_KEYS.items() if reason in reason_codes for key in names]
    if not keys:
        return ''
    wall = 'host_wall_budget_exhaustion' in reason_codes
    return ('The review used up its allowance before it finished; raise ' + ' and '.join(keys)
            + f" under {code('[pipeline]')} in the review profile. The review multiplies "
            + f"{code('reviewer_tool_call_budget')} and {code('reviewer_investigation_budget_s')} by the "
            + f"{code('scale')} its {code('settings.limits')} record"
            + (f"; {code('review_wall_budget_s')} caps the whole review and grows with the diff in the "
               "default profile only." if wall else "."))


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
    ]
    if terminal['reason_codes']:
        lines.append('Unfinished work: ' + ', '.join(_code(reason) for reason in terminal['reason_codes']) + '.')
    if sentence := allowance_sentence(terminal['reason_codes']):
        lines.append(sentence)
    lines.extend(['', '## Findings', ''])
    if not payload['findings']:
        lines.append('No confirmed findings.')
    for disposition, heading in (
        ('confirmed', None), ('unresolved', 'Unresolved concerns'), ('rejected', 'Rejected concerns'),
    ):
        claims = [claim for claim in payload['claims'] if claim['disposition'] == disposition]
        if heading is not None and claims:
            lines.extend(['', '## ' + heading, ''])
        for claim in claims:
            semantic = claim['payload']
            location = ':'.join(_code(semantic[key]) for key in ('file', 'line') if semantic[key] is not None)
            lines.extend([
                f"### {claim['id']}. {claim['title']}", '',
                f"Location: {location}.", '',
                semantic['claim'],
            ])
            for key, label in (
                ('trigger', 'Trigger'), ('consequence', 'Consequence'), ('grounds', 'Supporting grounds'),
            ):
                lines.extend(['', f'**{label}:** {semantic[key]}'])
            for challenge in claim['challenge_evidence']:
                lines.extend([
                    '', f"Challenge {_code(challenge['verdict'])} for revision {_code(challenge['revision'])}:", '',
                    challenge['grounds'],
                ])
    return '\n'.join(lines) + '\n'


def write_committed_review_output(
    session: ArtifactSession | None, payload: dict[str, Any], *, report_path: Path, result_path: Path,
) -> None:
    """Stage the report first; an owning session supplies conflict-safe final publication."""
    validate_committed_result(payload)
    if session is not None and payload['run_id'] != session.layout.session_id:
        raise ValueError('committed review output differs from its artifact run')
    atomic_write_bytes(report_path, render_committed_review(payload).encode('utf-8'))
    atomic_write_json(result_path, payload, trailing_newline=True)
