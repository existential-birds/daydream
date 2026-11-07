"""Truthful coverage derivation and strict terminal-result invariants."""
import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from daydream.review_result import (
    ReasonCode,
    ReviewCoverage,
    reason_for_exception,
    validate_terminal_result,
)
from tests.harness.review_result import review_coverage


def large_attributed_terminal() -> dict[str, Any]:
    """A valid storage input whose 34 distinct frozen-range assertions exceed the findings cap."""
    coverage = review_coverage(scope_ids=('python', *(f'python#{i:02}' for i in range(1, 17))),
                               files=('a.py', 'b.py'))
    for scope_id in coverage.scopes:
        coverage.record_scope(scope_id, 'complete')
    coverage.record_phase('merge', 'complete', noop=True)
    terminal = coverage.finalize('completed')
    for scope_index, scope in enumerate(terminal['stack_outcomes']):
        scope['target_decisions'] = [{
            'target_id': f'part:{index:06}', 'status': 'reviewed', 'reason': 'r' * (32 * 1024),
            'evidence_refs': [{
                'artifact_sha256': 'a' * 64, 'part_sha256': 'c' * 64,
                'revision': coverage.revision.head_sha, 'path': 'a.py' if index % 2 else 'b.py', 'side': 'new',
                'source_range': {'start': scope_index * 2 + index + 1, 'end': scope_index * 2 + index + 1},
                'fragment': None, 'basis': 'supplied_diff', 'role': 'assigned',
            }],
        } for index in range(2)]
    return terminal


def attribution_bytes(artifact: dict[str, Any]) -> int:
    """Independent exact published-size difference after removing only attribution fields."""
    core = copy.deepcopy(artifact)
    for scope in core.get('terminal_result', {}).get('stack_outcomes', []):
        scope.pop('target_decisions', None)
    def size(value: dict[str, Any]) -> int:
        return len((json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    return size(artifact) - size(core)


def test_terminal_state_requires_positive_usable_coverage() -> None:
    intent_only = review_coverage(scope_ids=(), phases=("intent",))
    intent_only.record_phase("intent", "complete")
    result = intent_only.finalize("completed")
    assert (result["analysis_state"], result["reason_codes"]) == ("failed", ["coverage_unknown"])
    cancelled = review_coverage()
    cancelled.record_scope("python", "complete")
    result = cancelled.finalize("cancelled")
    assert result["analysis_state"] == "incomplete" and "interruption" in result["reason_codes"]
    assert [scope["scope_id"] for scope in result["stack_outcomes"]] == ["python", "structure"]
    with pytest.raises(ValueError, match="frozen"):
        cancelled.require_phase("adjudication")


def test_terminal_rejects_inconsistent_or_unknown_evidence() -> None:
    mutations: list[Callable[[dict[str, Any]], None]] = [
        lambda r: r["planned_scopes"].append(copy.deepcopy(r["planned_scopes"][0])),
        lambda r: r.update(analysis_state="complete"),
        lambda r: r["reason_codes"].append("invented_reason"),
        lambda r: r.update(completed_stacks=["structure"]),
        lambda r: r["stack_outcomes"][0].update(partial_evidence=True),
        lambda r: r.update(schema_version=2),
        lambda r: r["planned_scopes"][0]["files"].reverse(),
        lambda r: r["stack_outcomes"][0].update(files=["a.py"]),
    ]
    for mutate in mutations:
        c = review_coverage(files=("a.py", "b.py"))
        c.record_scope("python", "complete")
        result = c.finalize("failed")
        validate_terminal_result(copy.deepcopy(result))
        mutate(result)
        with pytest.raises(ValueError):
            validate_terminal_result(result)


def test_retained_attribution_bounds_reject_atomically_after_a_valid_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage limits must reject overflow without losing already admitted decisions."""
    for bound in ('references-bytes', 'scope-bytes', 'run-bytes'):
        # Keep a reachable exact aggregate boundary while each scope stays independently valid.
        run_limit = 256 * 1024
        if bound == 'run-bytes':
            monkeypatch.setattr('daydream.review_result.RUN_TARGET_DECISIONS_MAX_BYTES', run_limit)
        c = review_coverage()
        reference: dict[str, Any] = {
            'artifact_sha256': 'a' * 64, 'part_sha256': 'c' * 64,
            'revision': c.revision.head_sha, 'path': 'a.py', 'side': 'new',
            'source_range': {'start': 1, 'end': 1}, 'fragment': None,
            'basis': 'supplied_diff', 'role': 'assigned',
        }
        valid: list[dict[str, Any]] = [
            {'target_id': 'a.py:part:1', 'status': 'reviewed', 'reason': '', 'evidence_refs': [reference]}]
        if bound == 'references-bytes':
            valid[0]['evidence_refs'] = [{**reference, 'path': 'context/' * 188 + f'source-{i}.py'}
                                         for i in range(4)]
        elif bound == 'run-bytes':
            valid[0]['reason'] = 'é\u2028' * 25000
            c.record_scope('structure', 'incomplete', reasons=['evidence_incomplete'], partial_evidence=True,
                           target_decisions=valid)
        else:
            valid[0]['reason'] = 'é' * 130000
        c.record_scope('python', 'incomplete', reasons=['evidence_incomplete'], partial_evidence=True,
                       target_decisions=valid)
        before = c.to_dict()
        invalid = copy.deepcopy(valid)
        if bound == 'references-bytes':
            for ref in invalid[0]['evidence_refs']:
                ref['path'] += '/context' * 40
            assert len(json.dumps(valid[0]['evidence_refs']).encode()) < 8192
            assert len(json.dumps(invalid[0]['evidence_refs']).encode()) > 8192
            expected = 'source references exceed retained byte limit'
        elif bound == 'run-bytes':
            without_decisions = copy.deepcopy(before)
            for scope in without_decisions['stack_outcomes']:
                scope.pop('target_decisions', None)
            def published_size(value: dict[str, Any]) -> int:
                return len(json.dumps({'terminal_result': value}, indent=2, ensure_ascii=False).encode('utf-8'))
            contribution = published_size(before) - published_size(without_decisions)
            invalid[0]['reason'] += 'x' * (run_limit - contribution + 1)
            assert sum(len(json.dumps(decisions, ensure_ascii=False).encode()) for decisions in [valid, invalid]) < (
                run_limit), 'compact JSON misses the nested whitespace overflow'
            corrupt = copy.deepcopy(before)
            corrupt_scope = next(scope for scope in corrupt['stack_outcomes'] if scope['scope_id'] == 'python')
            corrupt_scope['target_decisions'] = invalid
            assert published_size(corrupt) - published_size(without_decisions) == run_limit + 1
            terminal = ReviewCoverage.from_dict(before).finalize('completed')
            next(scope for scope in terminal['stack_outcomes'] if scope['scope_id'] == 'python')[
                'target_decisions'] = invalid
            with pytest.raises(ValueError, match='run source attribution exceeds retained byte limit'):
                validate_terminal_result(terminal)
            with pytest.raises(ValueError, match='run source attribution exceeds retained byte limit'):
                ReviewCoverage.from_dict(corrupt)
            expected = 'run source attribution exceeds retained byte limit'
        else:
            invalid[0]['reason'] += 'é' * 1200
            assert len(json.dumps(valid, ensure_ascii=False).encode()) < 256 * 1024
            assert len(json.dumps(invalid, ensure_ascii=False).encode()) > 256 * 1024
            expected = 'target decisions exceed retained byte limit'
        with pytest.raises(ValueError, match=expected):
            c.record_scope('python', 'incomplete', reasons=['backend_failure'], partial_evidence=True,
                           target_decisions=invalid)
        assert c.to_dict() == before
        assert ReviewCoverage.from_dict(before).to_dict() == before


def test_identity_and_invalid_recording_are_rejected() -> None:
    c = review_coverage()
    with pytest.raises(ValueError, match="head"):
        validate_terminal_result(c.finalize("completed"), expected_head_sha="different-head")
    with pytest.raises(ValueError, match="duplicate"):
        review_coverage(scope_ids=("same", "same"))
    with pytest.raises(ValueError, match="unknown"):
        review_coverage().record_scope("extra", "complete")
    with pytest.raises(ValueError, match="usable"):
        review_coverage().record_phase("merge", "failed", reasons=["synthesis_failure"], usable_evidence=True)
    corrupt: Any = None
    with pytest.raises(ValueError):
        ReviewCoverage.from_dict(corrupt)


def test_exception_classification_uses_host_attributes() -> None:
    class ProviderError(RuntimeError):
        reason_code: Any = "provider-specific-error"

    error = ProviderError("invalid api key")
    assert reason_for_exception(error) == ReasonCode.BACKEND_FAILURE
    error.reason_code = None
    error.category = "AUTH_CONFIG"  # type: ignore[attr-defined]
    assert reason_for_exception(error) == ReasonCode.AUTHENTICATION_FAILURE
