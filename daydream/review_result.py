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

from daydream.output_schema import strict_object
from daydream.redaction import redact_text
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
    return {
        FailureClass.AUTH_CONFIG: ReasonCode.AUTHENTICATION_FAILURE,
        FailureClass.SCHEMA: ReasonCode.MALFORMED_OUTPUT,
        FailureClass.TOOL_POLICY: ReasonCode.POLICY_VETO,
    }.get(classify_failure(exc).failure_class, ReasonCode.BACKEND_FAILURE)


def reason_for_budget(reason: str) -> ReasonCode:
    """Classify the canonical host stop vocabulary at its trusted boundary."""
    if reason.startswith('tool_vetoed'):
        return ReasonCode.POLICY_VETO
    return {
        'evidence_incomplete': ReasonCode.EVIDENCE_INCOMPLETE,
        'wall_budget_exceeded': ReasonCode.HOST_WALL_BUDGET_EXHAUSTION,
        'tool_call_budget_exceeded': ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION,
        'error_max_turns': ReasonCode.MODEL_BUDGET_EXHAUSTION,
        'pipeline_budget_exceeded': ReasonCode.HOST_PIPELINE_BUDGET_EXHAUSTION,
    }[reason]


_TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 512}
_REASONS = {'type': 'array', 'uniqueItems': True, 'maxItems': len(ReasonCode),
            'items': {'enum': [r.value for r in ReasonCode]}}
_SCOPE_PROPERTIES = {
    'scope_id': _TEXT, 'stack': _TEXT,
    'files': {'type': 'array', 'maxItems': 10000, 'uniqueItems': True, 'items': _TEXT},
    'shard': {'type': ['integer', 'null'], 'minimum': 0},
    'scope_ref': {'type': ['string', 'null'], 'maxLength': 512},
}
_SCOPE_SCHEMA = strict_object(_SCOPE_PROPERTIES)
_OUTCOME_PROPERTIES = {'status': {'enum': ['complete', 'incomplete', 'failed', 'uncovered']},
                       'reason_codes': _REASONS}
_STACK_SCHEMA = strict_object({**_SCOPE_PROPERTIES, **_OUTCOME_PROPERTIES, 'partial_evidence': {'type': 'boolean'}})
_PHASE_SCHEMA = strict_object({'phase': _TEXT, **_OUTCOME_PROPERTIES, 'noop': {'type': 'boolean'},
                              'usable_evidence': {'type': 'boolean'}})
_REVISION_SCHEMA = strict_object({'head_sha': _TEXT, 'merge_base_sha': _TEXT, 'diff_key': _TEXT, 'pr_base_sha': _TEXT})
_REVISION_SCHEMA['required'].remove('pr_base_sha')
TERMINAL_RESULT_SCHEMA: dict[str, Any] = strict_object({
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
})


_COVERAGE_FIELDS = ('schema_version', 'run_id', 'analyzed_revision', 'planned_scopes',
                    'required_phases', 'stack_outcomes', 'phase_outcomes')
_COVERAGE_SCHEMA = strict_object({
    **{key: TERMINAL_RESULT_SCHEMA['properties'][key] for key in _COVERAGE_FIELDS},
    'diagnostics': strict_object({kind: {'type': 'object', 'maxProperties': 10000,
        'additionalProperties': {'type': 'string', 'minLength': 1, 'maxLength': 1024}}
        for kind in ('scopes', 'phases')}),
})


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
    usable = usable or (not stacks and any(
        p['phase'] == 'no_diff' and p['status'] == 'complete' and p['noop'] for p in phases
    ))
    reasons = {reason for o in [*stacks, *phases] for reason in o['reason_codes']}
    if not stacks and not usable:
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
    if data['required_phases'] != sorted(data['required_phases']):
        raise ValueError('required phases must have deterministic ordering')
    for scope_id, scope in planned.items():
        if scope['files'] != sorted(scope['files']):
            raise ValueError('scope files must have deterministic ordering')
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
                 planned_scopes: Iterable[PlannedScope], required_phases: Iterable[str]):
        self.run_id = run_id
        self.revision = revision
        self.planned_scopes = tuple(planned_scopes)
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
        self.diagnostics: dict[str, dict[str, str]] = {"scopes": {}, "phases": {}}
        self._terminal: dict[str, Any] | None = None
        self._frozen_evidence: dict[str, Any] | None = None

    @property
    def unfinished_scopes(self) -> dict[str, str]:
        """Render current unfinished scope evidence for downstream review context."""
        return {key: self.diagnostics['scopes'].get(key, ', '.join(outcome['reason_codes']))
                for key, outcome in self.scopes.items() if outcome['status'] != 'complete'}

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

    def _record(self, outcomes: dict[str, dict[str, Any]], key: str, status: str,
                reasons: Iterable[str], diagnostic: str | None, **fields: bool) -> None:
        self._mutable()
        if key not in outcomes:
            raise ValueError(f'unknown planned scope or required phase: {key}')
        outcome = {**outcomes[key], 'status': status,
                   'reason_codes': sorted({ReasonCode(r).value for r in reasons}), **fields}
        phase = outcomes is self.phases
        jsonschema.validate(outcome, _PHASE_SCHEMA if phase else _STACK_SCHEMA)
        _validate_outcome(outcome, phase=phase)
        if diagnostic is not None and not isinstance(diagnostic, str):
            raise TypeError('review diagnostic must be a string or None')
        diagnostics = self.diagnostics['phases' if phase else 'scopes']
        diagnostics.pop(key, None)
        if status != 'complete' and diagnostic:
            diagnostics[key] = redact_text(diagnostic)[:1024]
        outcomes[key] = outcome

    def record_scope(self, scope_id: str, status: str, *, reasons: Iterable[str] = (),
                     partial_evidence: bool = False, diagnostic: str | None = None) -> None:
        self._record(self.scopes, scope_id, status, reasons, diagnostic, partial_evidence=partial_evidence)

    def record_phase(self, phase: str, status: str, *, reasons: Iterable[str] = (), noop: bool = False,
                     usable_evidence: bool = False, diagnostic: str | None = None) -> None:
        self._record(self.phases, phase, status, reasons, diagnostic, noop=noop, usable_evidence=usable_evidence)

    def to_dict(self) -> dict[str, Any]:
        """Persist checked evidence, retaining the exact snapshot after finalization."""
        if self._terminal is not None:
            assert self._frozen_evidence is not None
            return copy.deepcopy(self._frozen_evidence)
        return copy.deepcopy({'schema_version': 1, 'run_id': self.run_id,
            'analyzed_revision': self.revision.to_dict(),
            'planned_scopes': sorted((s.to_dict() for s in self.planned_scopes), key=lambda s: s['scope_id']),
            'required_phases': sorted(self.required_phases),
            'stack_outcomes': [self.scopes[key] for key in sorted(self.scopes)],
            'phase_outcomes': [self.phases[key] for key in sorted(self.phases)],
            'diagnostics': self.diagnostics})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewCoverage:
        """Restore coverage only after strict structural and semantic validation."""
        try:
            jsonschema.validate(data, _COVERAGE_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise ValueError(f'invalid coverage artifact schema: {exc.message}') from exc
        if type(data['schema_version']) is not int:
            raise ValueError('invalid coverage artifact schema version')
        c = cls(data['run_id'], AnalyzedRevision.from_dict(data['analyzed_revision']),
                [PlannedScope(**{**s, 'files': tuple(s['files'])}) for s in data['planned_scopes']],
                data['required_phases'])
        c.scopes = {s['scope_id']: copy.deepcopy(s) for s in data['stack_outcomes']}
        c.phases = {p['phase']: copy.deepcopy(p) for p in data['phase_outcomes']}
        c.diagnostics = copy.deepcopy(data['diagnostics'])
        for kind, outcomes in [('scopes', c.scopes), ('phases', c.phases)]:
            for key, diagnostic in c.diagnostics[kind].items():
                if (key not in outcomes or outcomes[key]['status'] == 'complete'
                        or redact_text(diagnostic) != diagnostic):
                    raise ValueError('diagnostics require unfinished known outcomes and redacted text')
        # Validate the supplied inventory order too, before canonical serialization.
        validate_terminal_result(c._snapshot(dict(data), 'completed', True))
        return c

    def _snapshot(self, data: dict[str, Any], pipeline_state: str, projection_valid: bool) -> dict[str, Any]:
        data.pop('diagnostics', None)
        result = {**data, 'pipeline_state': pipeline_state, 'projection_valid': projection_valid,
                  'completed_stacks': sorted(k for k, o in self.scopes.items() if o['status'] == 'complete'),
                  'failed_stacks': sorted(k for k, o in self.scopes.items() if o['status'] == 'failed'),
                  'uncovered_stacks': sorted(k for k, o in self.scopes.items() if o['status'] != 'complete')}
        result['analysis_state'], result['reason_codes'] = _derive(result)
        return result

    def finalize(self, pipeline_state: str, *, projection_valid: bool = True) -> dict[str, Any]:
        """Freeze exactly once after all scope writers join and projection is validated."""
        if self._terminal is not None:
            raise ValueError('review coverage is already frozen')
        result = self._snapshot(self.to_dict(), pipeline_state, projection_valid)
        validate_terminal_result(result)
        self._frozen_evidence = self.to_dict()
        self._terminal = copy.deepcopy(result)
        return copy.deepcopy(result)
