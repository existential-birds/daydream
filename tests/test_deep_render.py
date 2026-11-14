from typing import Any

from daydream.deep.render import insert_diagrams_section, insert_review_context, render_report
from tests.harness.review_result import terminal_result


def _item(id: Any, lens: Any, file: Any, line: Any, sev: Any, desc: Any) -> dict[str, Any]:
    return {"id": id, "lens": lens, "file": file, "line": line, "severity": sev,
            "description": desc, "confidence": "HIGH", "rationale": "r"}

# ---------------------------------------------------------------------------
# ## Diagrams section (issue #1113). The blocks themselves are rendered by
# deep/diagram_render.py and golden-tested in tests/test_diagram_render.py;
# these tests pin the pure text surgery render.py performs on the report.
# ---------------------------------------------------------------------------

_BLOCKS = "<details><summary><h3>Flowchart</h3></summary>\n\n```mermaid\nflowchart TD\n```\n\n</details>"

def _items() -> list[dict[str, Any]]:
    return [_item(1, "per-stack", "a.py", 9, "medium", "bug"), _item(2, "cross-stack", "b.py", 2, "high", "drift")]

def test_empty_report_retains_attributed_rationale_and_unproduced_checks_through_replacement() -> None:
    """Empty finding rendering previously discarded all existing scope reasoning and limits."""
    terminal = terminal_result()
    revision = terminal['analyzed_revision']
    context = {'scopes': [{'scope_id': 'python', 'originating_run_id': 'source-run',
        'analyzed_revision': revision, 'review_context': {'admitted_stages': 2,
        'target_decisions': {'required': 3, 'reviewed': 3, 'not_reviewed': 0}, 'notes_omitted': 1,
        'notes': [{'text': 'The caller handles the failure.\n## forged heading\nOPENROUTER_API_KEY=secret',
                   'files': ['a.py'], 'supporting_files': [], 'origin': {'scope_id': 'python',
                   'stage': 'first_pass', 'sequence': 1, 'judgment_origin': 'reviewer',
                   'analyzed_revision': revision}}]}}],
        'verification': {'status': 'unproduced', 'value': None},
        'author_reported': {'source': 'pull_request_body', 'head_sha': 'reported-head',
                            'body': 'CI and make check pass.'}}
    report = render_report([], review_context=context, terminal_result=terminal)
    assert 'No confirmed findings' in report and 'Recorded scope completion: 1/1' in report
    assert 'The caller handles the failure.' in report and 'source-run' in report
    assert 'Executable verification unavailable' in report and 'unproduced' in report
    assert 'Author-supplied assertions' in report and 'reported-head' in report
    assert 'secret' not in report and '[REDACTED' in report
    assert '\n## forged heading' not in report
    assert 'does not independently prove semantic coverage' in report
    assert 'additional source notes omitted' in report
    held = '## Section of another writer\n1. [held.py:8] retained through replacement'
    complete = 'Review warning\n\n' + report + '\n' + held + '\n\n## Coverage\nExisting coverage\n'
    once = insert_review_context(complete, context, terminal)
    assert insert_review_context(once, context, terminal) == once
    diagram = insert_diagrams_section(once, _BLOCKS)
    replaced = insert_diagrams_section(diagram, '<details>replacement</details>')
    assert insert_diagrams_section(replaced, '') == once
    assert insert_review_context(diagram, context, terminal).count('## Review outcome') == 1
    assert 'Review warning' in once and held in once and 'Existing coverage' in once
    # Without a review heading or trailing newline, the section keeps the report's conventions.
    out = insert_diagrams_section("## Coverage\n- Files in diff: 1\n", "<details>D</details>")
    assert out == "## Diagrams\n<details>D</details>\n\n## Coverage\n- Files in diff: 1\n"
    assert insert_diagrams_section(out, "<details>D</details>") == out
    # A report with no trailing newline keeps that convention.
    assert insert_diagrams_section("# Review", "<details>D</details>") == (
        "# Review\n\n## Diagrams\n<details>D</details>"
    )
    assert insert_diagrams_section("", "<details>D</details>") == "## Diagrams\n<details>D</details>"
    assert insert_diagrams_section("", "") == ""


def test_recorded_checks_keep_failure_identity_and_anomalous_rationale_explicit() -> None:
    """Artifact prose cannot supply execution identity or trustworthy source attribution."""
    outcome = {'unexpected': 'failed'}
    terminal = terminal_result('incomplete')
    context = {'scopes': [{'scope_id': 'python', 'originating_run_id': None,
        'analyzed_revision': terminal['analyzed_revision'], 'review_context': {
        'notes': [{'text': 'Everything passes', 'origin': {'scope_id': 'foreign'}}]}}],
        'verification': {'status': 'available', 'value': {'test-verdict.json': {
        'status': 'executed', 'passed': False, 'attempts': [{'kind': 'host', 'command': ['pytest', '-q'],
        'passed': False, 'identity': {'head_sha': 'executed-head', 'session_id': 'executed-session',
                                    'outcome': outcome, 'kind': 'host', 'argv': ['pytest', '-q']}},
        {'kind': 'agent', 'command': ['claimed-check'], 'passed': True}]},
        'remote-ci-verdict.json': {'status': 'failed', 'head_sha': 'remote-head'}}}}
    report = render_report([], review_context=context, terminal_result=terminal)
    assert 'incomplete' in report and 'backend_failure' in report and 'Run:' in report
    assert 'Anomalous source attribution' in report and 'originating run <code>unavailable</code>' in report
    assert 'Everything passes' in report and 'not executable verification' in report
    assert all(value in report for value in ('pytest', 'executed-head', 'executed-session', 'failed', 'remote-head'))
    assert 'Checks actually executed' not in report
    assert 'Other test assertions or no-execution records do not establish host execution' in report
    assert 'does not refresh GitHub status' in report
    assert 'Source rationale unavailable' in render_report([], review_context={})
    malformed = {'scopes': [{'scope_id': 'python', 'originating_run_id': 'run-1', 'analyzed_revision': None,
                             'review_context': {'admitted_stages': 1, 'target_decisions': None,
                                                'notes': [{'text': '  '}, 'not a note']}}]}
    assert render_report([], review_context=malformed).count(
        'Source rationale unavailable: malformed or empty note.') == 2
    assert 'Terminal completion is not yet available' in render_report([], review_context={})
