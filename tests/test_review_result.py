"""Truthful coverage derivation and strict terminal-result invariants."""
import copy
import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from daydream.review_result import ReviewCoverage


def model() -> ModuleType:
    assert importlib.util.find_spec('daydream.review_result') is not None, 'typed result model must exist'
    return importlib.import_module('daydream.review_result')


def coverage() -> ReviewCoverage:
    m = model()
    return ReviewCoverage('run-1', m.AnalyzedRevision('a' * 40, 'b' * 40, 'diff-1'),
                            [m.PlannedScope('python', 'python', ('a.py',)),
                             m.PlannedScope('structure', 'structure', ('a.py',))], ['merge'])


def test_complete_empty_requires_positive_scope_and_phase_evidence() -> None:
    c = coverage()
    c.record_scope('python', 'complete')
    c.record_scope('structure', 'complete')
    c.record_phase('merge', 'complete', noop=True)
    result = c.finalize('completed')
    assert result['analysis_state'] == 'complete'
    assert result['reason_codes'] == []
    assert [s['scope_id'] for s in result['stack_outcomes']] == ['python', 'structure']
    with pytest.raises(ValueError, match='frozen'):
        c.record_scope('python', 'failed', reasons=['backend_failure'])


@pytest.mark.parametrize('partial', [False, True])
def test_empty_partial_is_usable_but_never_complete(partial: bool) -> None:
    c = coverage()
    c.record_scope('python', 'incomplete' if partial else 'failed',
                   reasons=['host_wall_budget_exhaustion' if partial else 'backend_failure'],
                   partial_evidence=partial)
    c.record_phase('merge', 'complete', noop=True)
    result = c.finalize('completed')
    assert result['analysis_state'] == ('incomplete' if partial else 'failed')
    assert result['uncovered_stacks'] == ['python', 'structure']


def test_complete_sibling_preserves_incomplete_analysis_on_pipeline_failure() -> None:
    c = coverage()
    c.record_scope('python', 'complete')
    c.record_scope('structure', 'failed', reasons=['backend_failure'])
    c.record_phase('merge', 'failed', reasons=['synthesis_failure'])
    assert c.finalize('failed')['analysis_state'] == 'incomplete'


@pytest.mark.parametrize('mutate', [
    lambda r: r['stack_outcomes'].pop(),
    lambda r: r['planned_scopes'].append(copy.deepcopy(r['planned_scopes'][0])),
    lambda r: r.update(analysis_state='complete'),
    lambda r: r['reason_codes'].append('invented_reason'),
    lambda r: r.update(completed_stacks=['structure']),
    lambda r: r['stack_outcomes'][0].update(partial_evidence=True),
])
def test_terminal_rejects_inconsistent_or_unknown_evidence(mutate: Callable[[dict[str, Any]], None]) -> None:
    c = coverage()
    c.record_scope('python', 'complete')
    result = c.finalize('failed')
    mutate(result)
    with pytest.raises(ValueError):
        model().validate_terminal_result(result)


def test_projection_failure_is_failed_even_with_complete_analysis() -> None:
    c = coverage()
    c.record_scope('python', 'complete')
    c.record_scope('structure', 'complete')
    c.record_phase('merge', 'complete')
    assert c.finalize('failed', projection_valid=False)['analysis_state'] == 'failed'


def test_coverage_roundtrip_preserves_inventory_and_partial_evidence() -> None:
    c = coverage()
    c.record_scope('python', 'incomplete', reasons=['host_tool_budget_exhaustion'], partial_evidence=True)
    restored = model().ReviewCoverage.from_dict(c.to_dict())
    assert restored.to_dict() == c.to_dict()
    assert restored.finalize('completed')['analysis_state'] == 'incomplete'


def test_exception_classification_does_not_invent_authentication_from_prose() -> None:
    m = model()
    assert m.reason_for_exception(RuntimeError('invalid api key')) == m.ReasonCode.BACKEND_FAILURE
    class AuthError(RuntimeError):
        category = 'AUTH_CONFIG'
    assert m.reason_for_exception(AuthError('redacted')) == m.ReasonCode.AUTHENTICATION_FAILURE
    assert m.reason_for_budget('wall') == m.ReasonCode.HOST_WALL_BUDGET_EXHAUSTION
    assert m.reason_for_budget('tool') == m.ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION


def test_no_diff_requires_positive_host_noop_evidence() -> None:
    m = model()
    c = m.ReviewCoverage('no-diff', m.AnalyzedRevision('a' * 40, 'a' * 40, 'empty'), [], ['no_diff'])
    c.record_phase('no_diff', 'complete', noop=True)
    assert c.finalize('completed')['analysis_state'] == 'complete'
    missing = m.ReviewCoverage('missing', c.revision, [], [])
    assert missing.finalize('completed')['analysis_state'] == 'failed'


@pytest.mark.parametrize('bad', [None, [], {'schema_version': 1}, {'schema_version': True}])
def test_corrupt_coverage_is_a_controlled_validation_failure(bad: Any) -> None:
    with pytest.raises(ValueError):
        model().ReviewCoverage.from_dict(bad)


def test_nested_head_identity_is_checked() -> None:
    c = coverage()
    result = c.finalize('completed')
    with pytest.raises(ValueError, match='head'):
        model().validate_terminal_result(result, expected_head_sha='different-head')


def test_duplicate_and_unknown_scopes_are_rejected() -> None:
    m = model()
    with pytest.raises(ValueError, match='duplicate'):
        m.ReviewCoverage('run', m.AnalyzedRevision('a', 'b', 'diff'),
                         [m.PlannedScope('same', 'python'), m.PlannedScope('same', 'python')], [])
    with pytest.raises(ValueError, match='unknown'):
        coverage().record_scope('extra', 'complete')


def test_dynamic_phase_requires_successful_evidence() -> None:
    c = coverage()
    c.require_phase('adjudication')
    c.record_scope('python', 'complete')
    c.record_scope('structure', 'complete')
    c.record_phase('merge', 'complete', noop=True)
    assert c.finalize('completed')['analysis_state'] == 'incomplete'


def test_cancelled_projection_preserves_interruption() -> None:
    c = coverage()
    c.record_scope('python', 'complete')
    result = c.finalize('cancelled')
    assert result['analysis_state'] == 'incomplete'
    assert 'interruption' in result['reason_codes']


def test_validated_alternative_review_is_usable_even_without_successful_stack() -> None:
    c = coverage()
    c.require_phase('alternatives')
    c.record_scope('python', 'failed', reasons=['backend_failure'])
    c.record_scope('structure', 'failed', reasons=['backend_failure'])
    c.record_phase('merge', 'complete', noop=True)
    c.record_phase('alternatives', 'complete', usable_evidence=True)
    assert c.finalize('completed')['analysis_state'] == 'incomplete'


def test_partial_alternative_evidence_is_usable_but_incomplete() -> None:
    c = coverage()
    c.require_phase('alternatives')
    c.record_phase('alternatives', 'incomplete', reasons=['host_wall_budget_exhaustion'], usable_evidence=True)
    assert c.finalize('failed')['analysis_state'] == 'incomplete'


def test_failed_phase_cannot_claim_validated_usable_evidence() -> None:
    c = coverage()
    with pytest.raises(ValueError, match='usable'):
        c.record_phase('merge', 'failed', reasons=['synthesis_failure'], usable_evidence=True)


def test_missing_all_coverage_has_explicit_unknown_reason() -> None:
    m = model()
    c = m.ReviewCoverage('missing', m.AnalyzedRevision('a', 'b', 'diff'), [], [])
    assert c.finalize('completed')['reason_codes'] == ['coverage_unknown']


def test_transport_incomplete_reason_is_not_a_pipeline_budget() -> None:
    m = model()
    assert m.reason_for_budget('evidence_incomplete') == m.ReasonCode.EVIDENCE_INCOMPLETE


def test_missing_projection_preserves_missing_artifact_category() -> None:
    c = coverage()
    c.require_phase('findings')
    c.record_phase('findings', 'failed', reasons=['missing_artifact'])
    result = c.finalize('failed', projection_valid=False)
    assert 'missing_artifact' in result['reason_codes']
    assert 'malformed_artifact' not in result['reason_codes']


def test_finalized_status_is_readonly_and_exposes_no_mutable_terminal() -> None:
    c = coverage()
    assert c.is_finalized is False
    result = c.finalize('failed')
    assert c.is_finalized is True
    result['analysis_state'] = 'complete'
    assert c.is_finalized is True
    with pytest.raises(AttributeError):
        setattr(c, 'is_finalized', False)


def test_frozen_coverage_persistence_retains_the_frozen_snapshot() -> None:
    c = coverage()
    c.record_scope('python', 'complete')
    c.finalize('failed')
    persisted = c.to_dict()
    c.scopes['python']['status'] = 'uncovered'
    c.run_id = 'late-other-run'
    assert c.to_dict() == persisted


@pytest.mark.parametrize('declared', ['provider-specific-error', 429, {'provider': 'error'}])
def test_unrelated_provider_reason_attribute_does_not_mask_backend_failure(declared: Any) -> None:
    class ProviderError(RuntimeError):
        reason_code: Any = declared
    m = model()
    assert m.reason_for_exception(ProviderError('ordinary provider error')) == m.ReasonCode.BACKEND_FAILURE
