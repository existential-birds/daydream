"""Host-owned review coverage and the strict terminal-result contract.

Findings count and pipeline exit status never establish analysis completeness.
Only explicit validated scope and phase evidence can do so.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import jsonschema

from daydream.retry_policy import FailureClass, classify_failure


class ReasonCode(StrEnum):
    """Bounded public reasons assigned at trusted host boundaries."""

    BACKEND_FAILURE = 'backend_failure'
    AUTHENTICATION_FAILURE = 'authentication_failure'
    MALFORMED_OUTPUT = 'malformed_output'
    MISSING_OUTPUT = 'missing_output'
    MALFORMED_ARTIFACT = 'malformed_artifact'
    MISSING_ARTIFACT = 'missing_artifact'
    EVIDENCE_INCOMPLETE = 'evidence_incomplete'
    MODEL_BUDGET_EXHAUSTION = 'model_budget_exhaustion'
    HOST_WALL_BUDGET_EXHAUSTION = 'host_wall_budget_exhaustion'
    HOST_TOOL_BUDGET_EXHAUSTION = 'host_tool_budget_exhaustion'
    HOST_PIPELINE_BUDGET_EXHAUSTION = 'host_pipeline_budget_exhaustion'
    SYNTHESIS_FAILURE = 'synthesis_failure'
    INTERRUPTION = 'interruption'
    UNEXPECTED_ANALYSIS_FAILURE = 'unexpected_analysis_failure'
    COVERAGE_UNKNOWN = 'coverage_unknown'
    POLICY_VETO = 'policy_veto'
    DIRTY_SNAPSHOT = 'dirty_snapshot'


def reason_for_exception(exc: BaseException) -> ReasonCode:
    """Classify trusted exception attributes without guessing auth from prose."""
    declared = getattr(exc, 'reason_code', None)
    if isinstance(declared, str):
        try:
            return ReasonCode(declared)
        except ValueError:
            pass
    if type(exc).__name__ == 'MaxTurnsError' or getattr(exc, 'subtype', None) == 'error_max_turns':
        return ReasonCode.MODEL_BUDGET_EXHAUSTION
    decision = classify_failure(exc)
    if decision.failure_class == FailureClass.AUTH_CONFIG:
        return ReasonCode.AUTHENTICATION_FAILURE
    if decision.failure_class == FailureClass.SCHEMA:
        return ReasonCode.MALFORMED_OUTPUT
    if decision.failure_class == FailureClass.TOOL_POLICY:
        return ReasonCode.POLICY_VETO
    return ReasonCode.BACKEND_FAILURE


def reason_for_budget(reason: str) -> ReasonCode:
    """Map host budget stop vocabulary without conflating provider turn limits."""
    if reason == 'evidence_incomplete':
        return ReasonCode.EVIDENCE_INCOMPLETE
    if reason.startswith('tool_vetoed'):
        return ReasonCode.POLICY_VETO
    if reason in {'wall', 'wall_time', 'wall_budget', 'deadline', 'wall_deadline', 'wall_budget_exceeded'}:
        return ReasonCode.HOST_WALL_BUDGET_EXHAUSTION
    if reason in {'tool', 'tools', 'tool_budget', 'tool_calls', 'tool_calls_exhausted', 'tool_call_budget_exceeded'}:
        return ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION
    if reason in {'model', 'max_turns', 'error_max_turns'}:
        return ReasonCode.MODEL_BUDGET_EXHAUSTION
    return ReasonCode.HOST_PIPELINE_BUDGET_EXHAUSTION


_TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 512}
_REASONS = {'type': 'array', 'uniqueItems': True, 'maxItems': len(ReasonCode),
            'items': {'enum': [r.value for r in ReasonCode]}}
_SCOPE_PROPERTIES = {
    'scope_id': _TEXT, 'stack': _TEXT,
    'files': {'type': 'array', 'maxItems': 10000, 'uniqueItems': True, 'items': _TEXT},
    'shard': {'type': ['integer', 'null'], 'minimum': 0},
    'scope_ref': {'type': ['string', 'null'], 'maxLength': 512},
}
_SCOPE_SCHEMA = {'type': 'object', 'additionalProperties': False,
                 'required': list(_SCOPE_PROPERTIES), 'properties': _SCOPE_PROPERTIES}
_OUTCOME_PROPERTIES = {'status': {'enum': ['complete', 'incomplete', 'failed', 'uncovered']},
                       'reason_codes': _REASONS}
_STACK_SCHEMA = {'type': 'object', 'additionalProperties': False,
                 'required': [*list(_SCOPE_PROPERTIES), *list(_OUTCOME_PROPERTIES), 'partial_evidence'],
                 'properties': {**_SCOPE_PROPERTIES, **_OUTCOME_PROPERTIES,
                                'partial_evidence': {'type': 'boolean'}}}
_PHASE_SCHEMA = {'type': 'object', 'additionalProperties': False,
                 'required': ['phase', *list(_OUTCOME_PROPERTIES), 'noop', 'usable_evidence'],
                 'properties': {'phase': _TEXT, **_OUTCOME_PROPERTIES, 'noop': {'type': 'boolean'},
                                'usable_evidence': {'type': 'boolean'}}}
_REVISION_SCHEMA = {'type': 'object', 'additionalProperties': False,
                    'required': ['head_sha', 'merge_base_sha', 'diff_key'],
                    'properties': {'head_sha': _TEXT, 'merge_base_sha': _TEXT, 'diff_key': _TEXT,
                                   'pr_base_sha': _TEXT}}
TERMINAL_RESULT_SCHEMA: dict[str, Any] = {
    'type': 'object', 'additionalProperties': False,
    'required': ['schema_version', 'run_id', 'analysis_state', 'pipeline_state', 'analyzed_revision',
                 'reason_codes', 'planned_scopes', 'required_phases', 'stack_outcomes', 'phase_outcomes',
                 'completed_stacks', 'failed_stacks', 'uncovered_stacks', 'projection_valid'],
    'properties': {
        'schema_version': {'const': 1}, 'run_id': _TEXT,
        'analysis_state': {'enum': ['complete', 'incomplete', 'failed']},
        'pipeline_state': {'enum': ['completed', 'failed', 'cancelled']},
        'analyzed_revision': _REVISION_SCHEMA, 'reason_codes': _REASONS,
        'planned_scopes': {'type': 'array', 'maxItems': 10000, 'items': _SCOPE_SCHEMA},
        'required_phases': {'type': 'array', 'uniqueItems': True, 'maxItems': 64, 'items': _TEXT},
        'stack_outcomes': {'type': 'array', 'maxItems': 10000, 'items': _STACK_SCHEMA},
        'phase_outcomes': {'type': 'array', 'maxItems': 64, 'items': _PHASE_SCHEMA},
        **{key: {'type': 'array', 'uniqueItems': True, 'items': _TEXT}
           for key in ('completed_stacks', 'failed_stacks', 'uncovered_stacks')},
        'projection_valid': {'type': 'boolean'},
    },
}


@dataclass(frozen=True)
class AnalyzedRevision:
    """Immutable endpoints and provenance of the exact input analyzed."""

    head_sha: str
    merge_base_sha: str
    diff_key: str
    pr_base_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.pr_base_sha is None:
            result.pop('pr_base_sha')
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnalyzedRevision:
        jsonschema.validate(dict(data), _REVISION_SCHEMA)
        return cls(**dict(data))


@dataclass(frozen=True)
class PlannedScope:
    """An intended execution scope after omission, collapse, and sharding."""

    scope_id: str
    stack: str
    files: tuple[str, ...] = ()
    shard: int | None = None
    scope_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['files'] = sorted(self.files)
        return result


def _derive(data: Mapping[str, Any]) -> tuple[str, list[str]]:
    stacks, phases = data['stack_outcomes'], data['phase_outcomes']
    all_complete = all(o['status'] == 'complete' for o in [*stacks, *phases])
    usable = any(o['status'] == 'complete' or o['partial_evidence'] for o in stacks)
    usable = usable or any(p['usable_evidence'] for p in phases)
    # A committed no-diff has no reviewers and explicit positive host phase evidence.
    usable = usable or (not stacks and bool(phases) and all_complete)
    reasons = {reason for o in [*stacks, *phases] for reason in o['reason_codes']}
    if not stacks and not phases:
        reasons.add(ReasonCode.COVERAGE_UNKNOWN.value)
    if not data['projection_valid'] and not any(
        p['phase'] == 'findings' and set(p['reason_codes']) & {
            ReasonCode.MISSING_ARTIFACT.value, ReasonCode.MALFORMED_ARTIFACT.value}
        for p in phases
    ):
        reasons.add(ReasonCode.MALFORMED_ARTIFACT.value)
    if data['pipeline_state'] == 'cancelled':
        reasons.add(ReasonCode.INTERRUPTION.value)
    if data['pipeline_state'] == 'failed' and not reasons:
        reasons.add(ReasonCode.UNEXPECTED_ANALYSIS_FAILURE.value)
    if not data['projection_valid'] or not usable:
        state = 'failed'
    elif all_complete and data['pipeline_state'] == 'completed':
        state = 'complete'
    else:
        state = 'incomplete'
    return state, sorted(reasons)


def _validate_outcome(outcome: Mapping[str, Any], *, phase: bool = False) -> None:
    status, reasons = outcome['status'], outcome['reason_codes']
    if status == 'complete' and reasons:
        raise ValueError('complete outcomes cannot retain failure reasons')
    if status != 'complete' and not reasons:
        raise ValueError('unfinished outcomes require an explicit reason')
    if reasons != sorted(reasons):
        raise ValueError('reason codes must have deterministic ordering')
    if phase:
        if outcome['usable_evidence'] and (status not in {'complete', 'incomplete'} or outcome['noop']):
            raise ValueError('usable phase evidence requires validated review work')
        if outcome['noop'] and status != 'complete':
            raise ValueError('host no-op must be complete')
    elif outcome['partial_evidence'] and status != 'incomplete':
        raise ValueError('validated partial evidence must remain incomplete')


def validate_terminal_result(data: Any, *, expected_head_sha: str | None = None) -> None:
    """Reject schema errors and contradictory aggregate/inventory evidence."""
    try:
        jsonschema.validate(data, TERMINAL_RESULT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f'terminal result schema validation failed: {exc.message}') from exc
    planned = {s['scope_id']: s for s in data['planned_scopes']}
    outcomes = {s['scope_id']: s for s in data['stack_outcomes']}
    phases = {p['phase']: p for p in data['phase_outcomes']}
    if len(planned) != len(data['planned_scopes']) or len(outcomes) != len(data['stack_outcomes']):
        raise ValueError('duplicate scope IDs')
    if len(phases) != len(data['phase_outcomes']):
        raise ValueError('duplicate phase IDs')
    if planned.keys() != outcomes.keys() or phases.keys() != set(data['required_phases']):
        raise ValueError('outcomes must exactly match planned scope and required phase inventory')
    for scope_id, scope in planned.items():
        if any(outcomes[scope_id][key] != value for key, value in scope.items()):
            raise ValueError('scope outcome identity differs from planned inventory')
    for outcome in outcomes.values():
        _validate_outcome(outcome)
    for phase in phases.values():
        _validate_outcome(phase, phase=True)
    for field, statuses in [('completed_stacks', {'complete'}), ('failed_stacks', {'failed'}),
                            ('uncovered_stacks', {'incomplete', 'failed', 'uncovered'})]:
        expected = sorted(key for key, outcome in outcomes.items() if outcome['status'] in statuses)
        if data[field] != expected:
            raise ValueError(f'{field} disagrees with stack outcomes')
    state, reasons = _derive(data)
    if data['analysis_state'] != state or data['reason_codes'] != reasons:
        raise ValueError('aggregate state or reasons disagree with coverage evidence')
    for field, identity in [('planned_scopes', 'scope_id'), ('stack_outcomes', 'scope_id'),
                            ('phase_outcomes', 'phase')]:
        if [o[identity] for o in data[field]] != sorted(o[identity] for o in data[field]):
            raise ValueError('outcomes and inventory must have deterministic ordering')
    if expected_head_sha is not None and data['analyzed_revision']['head_sha'] != expected_head_sha:
        raise ValueError('terminal result head does not match outer artifact head')


class ReviewCoverage:
    """Mutable host evidence until a single immutable terminal snapshot is frozen."""

    def __init__(self, run_id: str, revision: AnalyzedRevision,
                 planned_scopes: Iterable[PlannedScope | Mapping[str, Any]], required_phases: Iterable[str]):
        self.run_id = run_id
        self.revision = revision
        self.planned_scopes = tuple(s if isinstance(s, PlannedScope) else PlannedScope(
            **{**dict(s), 'files': tuple(s.get('files', ()))}) for s in planned_scopes)
        if len({s.scope_id for s in self.planned_scopes}) != len(self.planned_scopes):
            raise ValueError('duplicate scope IDs')
        self.required_phases = set(required_phases)
        self.scopes: dict[str, dict[str, Any]] = {
            s.scope_id: {**s.to_dict(), 'status': 'uncovered',
                         'reason_codes': [ReasonCode.COVERAGE_UNKNOWN.value], 'partial_evidence': False}
            for s in self.planned_scopes}
        self.phases: dict[str, dict[str, Any]] = {
            phase: {'phase': phase, 'status': 'uncovered',
                    'reason_codes': [ReasonCode.COVERAGE_UNKNOWN.value], 'noop': False, 'usable_evidence': False}
            for phase in self.required_phases}
        self._terminal: dict[str, Any] | None = None

    @property
    def is_finalized(self) -> bool:
        """Whether the review boundary has frozen its terminal result."""
        return self._terminal is not None

    def _mutable(self) -> None:
        if self._terminal is not None:
            raise ValueError('review coverage is frozen')

    def require_phase(self, phase: str) -> None:
        self._mutable()
        if phase not in self.required_phases:
            self.required_phases.add(phase)
            self.phases[phase] = {'phase': phase, 'status': 'uncovered',
                                  'reason_codes': [ReasonCode.COVERAGE_UNKNOWN.value],
                                  'noop': False, 'usable_evidence': False}

    def record_scope(self, scope_id: str, status: str, *, reasons: Iterable[str] = (),
                     partial_evidence: bool = False, diagnostic: str | None = None) -> None:
        self._mutable()
        if scope_id not in self.scopes:
            raise ValueError(f'unknown planned scope: {scope_id}')
        outcome = {**self.scopes[scope_id], 'status': status,
                   'reason_codes': sorted({ReasonCode(r).value for r in reasons}),
                   'partial_evidence': partial_evidence}
        jsonschema.validate(outcome, _STACK_SCHEMA)
        _validate_outcome(outcome)
        # Diagnostics remain in the existing redacted warnings channel, never the contract.
        self.scopes[scope_id] = outcome

    def record_phase(self, phase: str, status: str, *, reasons: Iterable[str] = (), noop: bool = False,
                     usable_evidence: bool = False) -> None:
        self._mutable()
        if phase not in self.required_phases:
            raise ValueError(f'unknown required phase: {phase}')
        outcome = {'phase': phase, 'status': status,
                   'reason_codes': sorted({ReasonCode(r).value for r in reasons}), 'noop': noop,
                   'usable_evidence': usable_evidence}
        jsonschema.validate(outcome, _PHASE_SCHEMA)
        _validate_outcome(outcome, phase=True)
        self.phases[phase] = outcome

    def to_dict(self) -> dict[str, Any]:
        """Persist checked evidence, retaining the exact snapshot after finalization."""
        if self._terminal is not None:
            fields = ("schema_version", "run_id", "analyzed_revision", "planned_scopes",
                      "required_phases", "stack_outcomes", "phase_outcomes")
            return copy.deepcopy({name: self._terminal[name] for name in fields})
        return copy.deepcopy({'schema_version': 1, 'run_id': self.run_id,
            'analyzed_revision': self.revision.to_dict(),
            'planned_scopes': sorted((s.to_dict() for s in self.planned_scopes), key=lambda s: s['scope_id']),
            'required_phases': sorted(self.required_phases),
            'stack_outcomes': [self.scopes[key] for key in sorted(self.scopes)],
            'phase_outcomes': [self.phases[key] for key in sorted(self.phases)]})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewCoverage:
        """Restore coverage only after strict structural and semantic validation."""
        expected = {'schema_version', 'run_id', 'analyzed_revision', 'planned_scopes',
                    'required_phases', 'stack_outcomes', 'phase_outcomes'}
        schema = {'type': 'object', 'additionalProperties': False, 'required': sorted(expected),
                  'properties': {key: TERMINAL_RESULT_SCHEMA['properties'][key] for key in expected}}
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f'invalid coverage artifact schema: {exc.message}') from exc
        if type(data['schema_version']) is not int:
            raise ValueError('invalid coverage artifact schema version')
        c = cls(data['run_id'], AnalyzedRevision.from_dict(data['analyzed_revision']),
                data['planned_scopes'], data['required_phases'])
        candidate = {**copy.deepcopy(dict(data)), 'pipeline_state': 'completed', 'projection_valid': True,
                     'completed_stacks': sorted(s['scope_id'] for s in data['stack_outcomes']
                                                if s['status'] == 'complete'),
                     'failed_stacks': sorted(s['scope_id'] for s in data['stack_outcomes'] if s['status'] == 'failed'),
                     'uncovered_stacks': sorted(s['scope_id'] for s in data['stack_outcomes']
                                                if s['status'] != 'complete')}
        candidate['analysis_state'], candidate['reason_codes'] = _derive(candidate)
        validate_terminal_result(candidate)
        c.scopes = {s['scope_id']: copy.deepcopy(s) for s in data['stack_outcomes']}
        c.phases = {p['phase']: copy.deepcopy(p) for p in data['phase_outcomes']}
        return c

    def finalize(self, pipeline_state: str, *, projection_valid: bool = True) -> dict[str, Any]:
        """Freeze exactly once after all scope writers join and projection is validated."""
        if self._terminal is not None:
            raise ValueError('review coverage is already frozen')
        result = {**self.to_dict(), 'pipeline_state': pipeline_state, 'projection_valid': projection_valid,
                  'completed_stacks': sorted(k for k, o in self.scopes.items() if o['status'] == 'complete'),
                  'failed_stacks': sorted(k for k, o in self.scopes.items() if o['status'] == 'failed'),
                  'uncovered_stacks': sorted(k for k, o in self.scopes.items() if o['status'] != 'complete')}
        result['analysis_state'], result['reason_codes'] = _derive(result)
        validate_terminal_result(result)
        self._terminal = copy.deepcopy(result)
        return copy.deepcopy(result)
