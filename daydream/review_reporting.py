"""Pure presentation of recorded review evidence, without acquiring or certifying it."""
from __future__ import annotations

import html
import json
import re
from typing import Any

from daydream.output_schema import strict_object
from daydream.redaction import redact_structured_text
from daydream.review_rationale import SOURCE_RATIONALE_SCHEMA
from daydream.review_result import ANALYZED_REVISION_SCHEMA

_TEXT = {'type': 'string', 'maxLength': 512}
REVIEW_CONTEXT_SCHEMA = strict_object({
    'scopes': {'type': 'array', 'maxItems': 32, 'items': strict_object({
        'scope_id': _TEXT, 'originating_run_id': {'anyOf': [_TEXT, {'type': 'null'}]},
        'analyzed_revision': {'anyOf': [ANALYZED_REVISION_SCHEMA, {'type': 'null'}]},
        'review_context': {'anyOf': [SOURCE_RATIONALE_SCHEMA, {'type': 'null'}]},
    })},
    'scopes_omitted': {'type': 'integer', 'minimum': 0},
    'verification': strict_object({
        'status': {'enum': ['available', 'unproduced', 'failed']},
        'value': {'type': ['object', 'null'], 'maxProperties': 16},
        'omitted_artifacts': {'type': 'array', 'maxItems': 16, 'items': _TEXT},
    }),
    'author_reported': {'anyOf': [strict_object({
        'source': {'const': 'pull_request_body'}, 'head_sha': {'anyOf': [_TEXT, {'type': 'null'}]},
        'body': {'type': 'string', 'maxLength': 4096}, 'truncated': {'type': 'boolean'},
    }), {'type': 'null'}]},
})
REVIEW_CONTEXT_SCHEMA['properties']['verification']['required'].remove('omitted_artifacts')

MAX_REVIEW_CONTEXT_CHARS = 20_000


def render_github_review_context(
    terminal_result: dict[str, Any] | None, *, review_warnings: tuple[str, ...] = (),
) -> str:
    """Bound GitHub diagnostics while retaining detailed evidence in local reports."""
    from daydream.review_result import validate_terminal_result

    if terminal_result is not None:
        validate_terminal_result(terminal_result)
    state = terminal_result['analysis_state'] if terminal_result is not None else None
    unfinished = [outcome for kind in ('stack_outcomes', 'phase_outcomes')
                  for outcome in (terminal_result or {}).get(kind, []) if outcome['status'] != 'complete']
    warnings = dict.fromkeys(review_warnings)
    if state == 'complete' and not warnings:
        return ''
    heading = ('Review completeness unavailable' if state is None else
               f'Review {state}' if state != 'complete' else 'Review incomplete')
    lines = [f'⚠️ **{heading}.**']
    diagnostics: dict[str, None] = {}
    for outcome in unfinished:
        key = outcome.get('scope_id', outcome.get('phase'))
        diagnostic = next((warning for warning in warnings
                           if warning.startswith(f'{key}: ')),
                          f"{key}: " + ', '.join(reason.replace('_', ' ') for reason in outcome['reason_codes']))
        diagnostics[diagnostic] = None
        warnings.pop(diagnostic, None)
    diagnostics.update(warnings)
    lines.extend('- ' + _text(warning, 512) for warning in list(diagnostics)[:8])
    if len(diagnostics) > 8:
        lines.append(f'- {len(diagnostics) - 8} additional notices; see run artifacts.')
    if state not in {None, 'complete'} and not unfinished:
        lines.append('Review did not finish; see run artifacts.')
    return '\n\n'.join(lines)


def render_review_run_id(terminal_result: dict[str, Any] | None) -> str:
    """Render an already-validated run identity without letting it escape its fence."""
    if terminal_result is None:
        return ''
    run_id = redact_structured_text(terminal_result['run_id'])[:512].replace('\r', ' ').replace('\n', ' ')
    longest_fence = max((len(match.group()) for match in re.finditer(r'`+', run_id)), default=0)
    fence = '`' * max(3, longest_fence + 1)
    return f'Run:\n{fence}text\n{run_id}\n{fence}'


def _text(value: Any, limit: int = 1024) -> str:
    text = redact_structured_text(str(value))
    return html.escape(text[:limit] + (' [clipped]' if len(text) > limit else '')).replace('\n', '&#10;')


def _code(value: Any) -> str:
    return '<code>' + _text(value).replace('\n', ' ') + '</code>'


def _payload(value: Any, limit: int = 4096) -> str:
    return '<pre>' + _text(json.dumps(value, sort_keys=True, ensure_ascii=False), limit) + '</pre>'


def _host_attempt(attempt: Any) -> bool:
    if not isinstance(attempt, dict) or attempt.get('kind') != 'host':
        return False
    identity, command = attempt.get('identity'), attempt.get('command')
    return (isinstance(identity, dict) and identity.get('kind') == 'host'
            and isinstance(command, list) and bool(command) and command == identity.get('argv')
            and all(isinstance(identity.get(key), str) and identity[key] for key in ('session_id', 'head_sha'))
            and isinstance(identity.get('outcome'), str)
            and identity.get('outcome') in {'passed', 'failed', 'timed-out', 'truncated'}
            and type(attempt.get('passed')) is bool
            and (identity['outcome'] == 'truncated' or attempt['passed'] == (identity['outcome'] == 'passed')))


def _render_review_context(
    context: dict[str, Any], terminal_result: dict[str, Any] | None = None, *, findings_count: int = 0,
) -> str:
    """Render host evidence and attributed assertions; missing facts remain unavailable."""
    lines = ['## Review outcome',
             'No confirmed findings.' if not findings_count else f'Confirmed findings: {findings_count}.']
    if terminal_result is None:
        lines.append('Terminal completion is not yet available.')
    else:
        scopes = terminal_result['stack_outcomes']
        complete = sum(scope['status'] == 'complete' for scope in scopes)
        lines.extend([
            f"Recorded analysis: {_code(terminal_result['analysis_state'])}; "
            f"pipeline: {_code(terminal_result['pipeline_state'])}.",
            f'Recorded scope completion: {complete}/{len(scopes)}.',
            f"Run: {_code(terminal_result['run_id'])}.",
            'Analyzed snapshot: ' + _payload(terminal_result['analyzed_revision']),
            'Reasons: ' + (', '.join(_code(reason) for reason in terminal_result['reason_codes']) or 'none recorded.'),
        ])
        unfinished = [scope for scope in scopes if scope['status'] != 'complete']
        for scope in unfinished[:64]:
            lines.append(f"Scope {_code(scope['scope_id'])}: {_code(scope['status'])}; "
                         + ', '.join(_code(reason) for reason in scope['reason_codes']) + '.')
        if len(unfinished) > 64:
            lines.append(f'{len(unfinished) - 64} additional unfinished scopes omitted.')
    lines.append('Completion records admitted assignment decisions; it does not independently prove semantic coverage.')
    lines.extend(['', '<details><summary>Reviewer source rationale</summary>', '',
                  'Reviewer assertions and source inspection are not executable verification.'])
    scope_contexts = context.get('scopes')
    scope_contexts = scope_contexts if isinstance(scope_contexts, list) else []
    if not scope_contexts:
        lines.append('Source rationale unavailable.')
    for scope in scope_contexts[:32]:
        if not isinstance(scope, dict):
            lines.append('Source rationale attribution unavailable: malformed scope record.')
            continue
        scope_id = scope.get('scope_id')
        lines.extend(['', f"Scope {_code(scope_id or 'unavailable')}; originating run "
                      f"{_code(scope.get('originating_run_id') or 'unavailable')}.",
                      'Source snapshot: ' + (_payload(scope['analyzed_revision'])
                          if scope.get('analyzed_revision') is not None else 'unavailable.')])
        detail = scope.get('review_context')
        if not isinstance(detail, dict):
            lines.append('Source rationale unavailable.')
            continue
        lines.append('Admitted stages and target decisions: ' + _payload({
            key: detail.get(key) for key in ('admitted_stages', 'target_decisions')}))
        notes = detail.get('notes')
        notes = notes if isinstance(notes, list) else []
        if not notes:
            lines.append('Source rationale unavailable.')
        for note in notes[:3]:
            if not isinstance(note, dict) or not isinstance(note.get('text'), str) or not note['text'].strip():
                lines.append('Source rationale unavailable: malformed or empty note.')
                continue
            origin = note.get('origin')
            if not isinstance(origin, dict):
                attribution = 'Attribution unavailable.'
            elif (origin.get('scope_id') != scope_id
                  or origin.get('analyzed_revision') != scope.get('analyzed_revision')
                  or origin.get('judgment_origin') != 'reviewer'):
                attribution = 'Anomalous source attribution; this assertion is not independently validated.'
            else:
                attribution = 'Reviewer source conclusion; attribution is recorded, semantic fidelity is unproven.'
            lines.extend([attribution, '<pre>' + _text(note['text']) + '</pre>',
                          'Recorded attribution: ' + _payload(origin),
                          'Referenced files (not read receipts): ' + _payload({
                              key: note.get(key) for key in ('files', 'supporting_files')})])
        notes_omitted = detail.get('notes_omitted')
        omitted = len(notes[3:]) + (notes_omitted if type(notes_omitted) is int else 0)
        if omitted:
            lines.append(f'{omitted} additional source notes omitted.')
    scopes_omitted = context.get('scopes_omitted')
    omitted = max(0, len(scope_contexts) - 32) + (scopes_omitted if type(scopes_omitted) is int else 0)
    if omitted:
        lines.append(f'{omitted} additional scope rationale records omitted.')
    lines.extend(['', '</details>', '', '### Verification evidence'])
    verification = context.get('verification')
    verification = verification if isinstance(verification, dict) else {}
    value = verification.get('value')
    lines.append('Verification artifact availability: ' + _code(verification.get('status', 'unproduced')) + '.')
    if verification.get('omitted_artifacts'):
        lines.append('Verification artifacts omitted due to the report evidence budget: '
                     + _payload(verification['omitted_artifacts']))
    if not isinstance(value, dict) or not value:
        lines.append('Executable verification unavailable; check artifacts: '
                     + _code(verification.get('status', 'unproduced')) + '.')
    else:
        lines.append('Recorded check artifacts are shown below; '
                     'no checks or remote CI were refreshed for this summary.')
        for name, artifact in sorted(value.items())[:16]:
            if name == 'item_associations':
                continue
            lines.append(f'<details><summary>Recorded artifact: {_text(name, 128)}</summary>')
            if name == 'test-verdict.json' and isinstance(artifact, dict):
                attempts = artifact.get('attempts')
                attempts = attempts if isinstance(attempts, list) else []
                host_attempts = [attempt for attempt in attempts if _host_attempt(attempt)]
                if host_attempts:
                    lines.append('Checks actually executed by the recorded host runner:')
                    lines.extend(_payload({key: attempt.get(key) for key in ('command', 'passed', 'identity')})
                                 for attempt in host_attempts[:8])
                if len(host_attempts) != len(attempts) or not attempts:
                    lines.append('Other test assertions or no-execution records do not establish host execution; '
                                 'executable verification for those records is unavailable.')
            elif name == 'remote-ci-verdict.json':
                lines.append('Recorded remote CI observation; this summary does not refresh GitHub status.')
            else:
                lines.append('Recorded evidence; artifact presence alone does not establish executable verification.')
            lines.extend([_payload(artifact, 8192), '</details>'])
    author = context.get('author_reported')
    if isinstance(author, dict):
        lines.extend(['', '<details><summary>Author-reported checks and context</summary>',
                      f"Source: {_code(author.get('source', 'unavailable'))}; "
                      f"reported head: {_code(author.get('head_sha') or 'unavailable')}.",
                      'Author-supplied assertions are not independently executed or refreshed check results.',
                      '<pre>' + _text(author.get('body') or 'Author context unavailable.', 4096) + '</pre>',
                      '</details>'])
    else:
        lines.append('Author-reported check context unavailable.')
    return '\n\n'.join(lines)


def _escaped_excerpt(text: str, budget: int) -> str:
    """Escape and bound presentation text without leaving a partial HTML entity."""
    escaped = html.escape(redact_structured_text(text))
    if len(escaped) <= budget:
        return escaped
    escaped = escaped[:max(0, budget - 22)]
    if escaped.rfind('&') > escaped.rfind(';'):
        escaped = escaped[:escaped.rfind('&')]
    return escaped + ' [details omitted]'


def render_review_context(
    context: dict[str, Any], terminal_result: dict[str, Any] | None = None, *, findings_count: int = 0,
) -> str:
    """Keep the entire host summary within a shared publication budget.

    Rich source detail remains in the JSON artifact. Oversized presentations use
    balanced shells and escaped excerpts, prioritizing snapshot and check identity.
    """
    rendered = _render_review_context(context, terminal_result, findings_count=findings_count)
    if len(rendered) <= MAX_REVIEW_CONTEXT_CHARS:
        return rendered
    header = rendered.split('<details>', 1)[0].rstrip()
    if terminal_result is not None:
        # The first seven paragraphs contain outcome, state, counts, run,
        # snapshot and reasons; the optional unfinished inventory follows them.
        header = '\n\n'.join(rendered.split('\n\n')[:7])
    if len(header) > 8_000:
        header = ('## Review outcome\n\n'
                  + ('No confirmed findings.' if not findings_count else f'Confirmed findings: {findings_count}.')
                  + '\n\n<pre>' + _escaped_excerpt(json.dumps(terminal_result, ensure_ascii=False), 7_500)
                  + '</pre>')
    verification = context.get('verification')
    value = verification.get('value') if isinstance(verification, dict) else None
    verdict = value.get('test-verdict.json') if isinstance(value, dict) else None
    attempts = verdict.get('attempts') if isinstance(verdict, dict) else None
    executed = [{key: attempt.get(key) for key in ('command', 'passed', 'identity')}
                for attempt in attempts if _host_attempt(attempt)] if isinstance(attempts, list) else []
    checks = ('Checks actually executed by the recorded host runner: '
              + json.dumps(executed[:8], ensure_ascii=False) if executed else
              'Executable verification details are in the recorded evidence excerpt below.')
    if isinstance(verification, dict):
        availability = 'Verification artifact availability: ' + str(verification.get('status', 'unproduced')) + '.\n'
        omitted_artifacts = verification.get('omitted_artifacts')
        if omitted_artifacts:
            availability += ('Verification artifacts omitted due to the report evidence budget: '
                             + json.dumps(omitted_artifacts) + '\n')
        checks = availability + checks
    prefix = (header + '\n\nCompletion records admitted decisions; semantic fidelity is unproven.\n\n'
              '<details><summary>Recorded verification</summary>\n\n<pre>'
              + _escaped_excerpt(checks, 5_000) + '</pre>\n\n</details>\n\n'
              'Additional details omitted to keep report context within its publication budget.\n\n'
              '<details><summary>Recorded evidence excerpt</summary>\n\n<pre>')
    suffix = '</pre>\n\n</details>'
    excerpt = _escaped_excerpt(rendered, MAX_REVIEW_CONTEXT_CHARS - len(prefix) - len(suffix))
    return prefix + excerpt + suffix
