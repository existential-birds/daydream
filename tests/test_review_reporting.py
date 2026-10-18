"""GitHub presentation must preserve typed failures without exposing rich artifacts."""
import pytest

from daydream.review_reporting import render_github_review_context, render_review_run_id
from tests.harness.review_result import review_coverage


@pytest.mark.parametrize('shared_phase', [False, True], ids=['one-scope', 'scope-and-phase-share-key'])
def test_github_warning_retains_distinct_diagnostics_for_the_same_key(shared_phase: bool) -> None:
    """Matching a terminal key must not discard additional diagnoses or replay the first one."""
    coverage = review_coverage(scope_ids=('merge',), phases=('merge',) if shared_phase else ())
    coverage.record_scope('merge', 'failed', reasons=('backend_failure',))
    if shared_phase:
        coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',))
    warnings = ('merge: synthesis transport failed', 'merge: reviewer backend failed',
                'merge: additional evidence unavailable', 'merge: synthesis transport failed')
    rendered = render_github_review_context(coverage.finalize('failed'), review_warnings=warnings)
    assert rendered.startswith('⚠️ **Review failed.**')
    for warning in dict.fromkeys(warnings):
        assert rendered.count(warning) == 1


def test_github_warning_retains_phase_failures_bounds_and_escapes_diagnostics() -> None:
    """A missing legacy sidecar cannot hide phase-only failure; huge scope lists stay bounded."""
    coverage = review_coverage(scope_ids=(f'stack-{index:02}' for index in range(12)), phases=('merge',))
    coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',))
    result = coverage.finalize('failed')
    warnings = ('stack-00: <unsafe> & incomplete',
                *(f'stack-{index:02}: coverage unknown' for index in range(1, 12)),
                *(f'transport notice {index:02}' for index in range(11)))
    warning = render_github_review_context(result, review_warnings=warnings)
    assert warning.startswith('⚠️ **Review failed.**')
    assert 'stack-00: &lt;unsafe&gt; &amp; incomplete' in warning
    assert 'stack-01: coverage unknown' in warning
    assert warning.count('\n\n- ') == 18
    assert '5 additional unfinished review steps' in warning
    assert 'stack-08' not in warning and 'transport notice 07' in warning and 'transport notice 08' not in warning
    assert '3 additional notices' in warning
    coverage = review_coverage(scope_ids=(), phases=('merge',))
    coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',))
    assert 'merge: synthesis failure' in render_github_review_context(coverage.finalize('failed'))


def test_github_unavailable_completion_and_run_fence_are_safe() -> None:
    """Unusual admitted identities must remain copyable without escaping Markdown fences."""
    coverage = review_coverage(run_id='run```\n<script>', scope_ids=('python',), phases=())
    coverage.record_scope('python', 'complete')
    result = coverage.finalize('completed')
    assert render_github_review_context(result) == ''
    assert render_github_review_context(None) == '⚠️ **Review completeness unavailable.**'
    assert render_review_run_id(result) == 'Run:\n````text\nrun``` <script>\n````'
    assert render_review_run_id(None) == ''
