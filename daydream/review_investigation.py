"""Bounded, in-memory progress for one snapshot-bound public reviewer scope."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import agent, clock

if TYPE_CHECKING:
    from daydream.review_stage_inputs import StageInputFactory
from daydream.backends import Backend, is_adapter_instance
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.host_tools import (
    _MAX_RESULT_BYTES,
    HostToolBinding,
    HostToolCallBudgetExceeded,
    HostToolFatal,
    HostToolRejected,
)
from daydream.json_utils import SchemaRejection, validates_schema
from daydream.phases.review_prompts import build_review_stage_system_instruction
from daydream.phases.schemas import (
    review_stage_admission_schema,
    review_stage_schema,
    review_stage_submission_schema,
)
from daydream.prompt_budget import (
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    truncate_utf8_to_budget,
)
from daydream.redaction import redact_structured_text, redact_text, redact_value
from daydream.repository_paths import (
    InvalidRepositoryFilePath,
    canonicalize_repository_file_path,
    valid_repository_file_path,
)
from daydream.retry_policy import FailureClass, classify_failure
from daydream.review_budget import BudgetLedger, ReviewLimits, review_limits_for_scope
from daydream.review_evidence import ReviewEvidence
from daydream.review_rationale import NOTE_ORIGIN_SCHEMA
from daydream.review_result import (
    TARGET_DECISIONS_MAX_BYTES,
    TARGET_DECISIONS_MAX_ITEMS,
    ReasonCode,
    ReviewCoverage,
    preserved_target_decisions,
    reason_for_budget,
    reason_for_exception,
)
from daydream.review_source_refs import _SubmissionSources
from daydream.run_context import resolve_run_context
from daydream.trajectory import DaydreamPhase, LifecycleReasonCode, LifecycleStatus, phase_scope
from daydream.ui.agent_stream import present_result

STAGED_REVIEW_CONTRACT = 17
HANDOFF_MAX_BYTES = 64 * 1024
HANDOFF_MAX_ITEMS = 128

def _candidate_source_files(candidates: list[dict[str, Any]]) -> set[str]:
    """Source anchors include each candidate and its actual terminal location."""
    files = {candidate['file'] for candidate in candidates if isinstance(candidate.get('file'), str)}
    for candidate in candidates:
        finding = candidate.get('finding')
        if (candidate.get('disposition') == 'confirmed' and isinstance(finding, dict)
                and isinstance(finding.get('file'), str)):
            files.add(finding['file'])
    return files


def _closed_decisions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render reviewer assertions identically for dispatch and semantic bounds."""
    decisions = []
    for candidate in candidates:
        disposition = candidate.get('disposition')
        if not isinstance(disposition, str) or disposition not in {'confirmed', 'rejected'}:
            continue
        conclusion = candidate['grounds']
        finding = candidate.get('finding')
        if isinstance(finding, dict):
            conclusion = f"{finding.get('description', '')}: {conclusion}"
        decisions.append({
            'candidate_id': candidate['candidate_id'], 'file': candidate['file'], 'line': candidate.get('line'),
            'disposition': candidate['disposition'], 'judgment_origin': 'reviewer',
            'conclusion': truncate_utf8_to_budget(conclusion, 512, '[conclusion clipped]')})
    return decisions


def _note_context(notes: list[Any]) -> list[dict[str, Any]]:
    """Add attribution without changing API 8 builders' notes:list[str] contract."""
    return [{'note_index': index, **{key: value for key, value in note.items() if key != 'text'}}
            for index, note in enumerate(notes)]


def _handoff_bytes(handoff: dict[str, list[Any]]) -> int:
    return len(json.dumps({**handoff, 'notes': [note['text'] for note in handoff['notes']],
        'note_context': _note_context(handoff['notes']), 'closed_decisions': _closed_decisions(handoff['candidates'])},
        ensure_ascii=False).encode('utf-8'))


@dataclass(frozen=True)
class _AdmissionPlan:
    """Prospective host state; constructing this never admits model output."""

    handoff: dict[str, list[Any]]
    target_progress: dict[str, dict[str, Any]]


class ReviewInvestigation:
    """Admit successful stage assertions; observed spend belongs to the host."""

    def __init__(self, stack: StackAssignment, diff_weights: Mapping[str, int], revision: dict[str, Any], *,
                 assignment_batches: list[list[dict[str, Any]]], coverage: ReviewCoverage,
                 limits: ReviewLimits | None = None) -> None:
        self.scope_id = stack.stack_name
        self.coverage = coverage
        self.files = sorted(stack.files)
        self.revision = revision
        self.diff_weights = diff_weights
        self.assignment_batches = assignment_batches
        self.assignments = {part['target_id']: part for batch in assignment_batches for part in batch}
        self.required_targets: dict[str, list[str]] = {path: [] for path in self.files}
        for batch in self.assignment_batches:
            for part in batch:
                self.required_targets[part['file']].append(part['target_id'])
        self.target_progress: dict[str, dict[str, Any]] = {}
        self.handoff: dict[str, list[Any]] = {name: [] for name in ('progress', 'notes', 'candidates')}
        self._base_limits = limits if limits is not None else ReviewLimits()
        # Structure runs one integration stage although its batches list every changed file.
        stages = 1 if self.scope_id == STRUCTURE_STACK_NAME else len(assignment_batches)
        self._scaled_limits = review_limits_for_scope(self._base_limits, stages=stages)
        self.budget = BudgetLedger.from_limits(f"stage:{self.scope_id}", self._base_limits, stages=stages)
        self._effective_allowance_s = 0.0
        self._deadline_bound = ('pipeline_budget' if self.budget.deadline == self.budget.shared_deadline
                                else 'reviewer_investigation_budget')
        self.reason: str | None = None
        self.admitted_stages = 0
        self.failed_invocation = False
        self.failure_diagnostic: str | None = None
        self.failure_class: str | None = None
        self.failure_reasons: list[ReasonCode] = []
        self._source_root: Path | None = None
        self._input_factory: StageInputFactory | None = None
        self._continuable_failure = False

    def review_context(self) -> dict[str, Any]:
        """Project existing admitted judgments for reporting, without eliciting more work.

        Source notes remain reviewer assertions. Missing or malformed attribution
        stays null; the report must never turn it into host-certified provenance.
        """
        required = ({f'integration:{self.scope_id}'} if self.scope_id == STRUCTURE_STACK_NAME else
                    {target for targets in self.required_targets.values() for target in targets})
        notes = []
        for note in self.handoff['notes'][:8]:
            origin = redact_value(note.get('origin'))
            notes.append({
                'text': truncate_utf8_to_budget(redact_structured_text(note['text']), 512, '[note clipped]'),
                **{name: [truncate_utf8_to_budget(redact_structured_text(path), 512, '[path clipped]')
                          for path in note.get(name, [])[:32]] for name in ('files', 'supporting_files')},
                'origin': origin if validates_schema(origin, NOTE_ORIGIN_SCHEMA) else None,
            })
        return {
            'admitted_stages': self.admitted_stages,
            'target_decisions': {
                'required': len(required),
                **{status: sum(self.target_progress.get(target, {}).get('status') == status for target in required)
                   for status in ('reviewed', 'not_reviewed')},
            },
            'notes': notes, 'notes_omitted': max(0, len(self.handoff['notes']) - len(notes)),
        }

    async def run(self, backend: Backend, cwd: Path, prompt_builder: Callable[[dict[str, Any]], str], *,
        stage_inputs: Callable[[dict[str, Any]], Awaitable[PreparedSanctionedInputs]],
        input_factory: StageInputFactory | None = None, **agent_kwargs: Any,
    ) -> tuple[dict[str, Any], str | None]:
        """Fresh calls share hard bounds; advisory targets cannot abort useful work."""
        self._source_root = cwd
        self._input_factory = input_factory
        # One invocation clamp covers preparation, every stage and all retries.
        # Passing only a relative wall allowance to run_agent would renew it after
        # preparation and again for each later stage; the scope ledger carries it instead.
        started = clock.monotonic()
        deadline, wall_budget_s = agent_kwargs.pop('deadline', None), agent_kwargs.pop('wall_budget_s', None)
        for bound, label in ((deadline, 'caller_deadline'),
                             (started + wall_budget_s if wall_budget_s is not None else None,
                              'invocation_wall_budget')):
            if bound is not None and bound < self._deadline:
                self.budget.deadline = bound
                self._deadline_bound = label
        self._effective_allowance_s = max(0.0, self._deadline - started)
        # Preserve the scope failure independently of each later attempt's status.
        # Full diagnostics for every attempt remain in their phase evidence.
        failure: tuple[str | None, bool, str | None, str | None] | None = None
        stopping_failure: tuple[str | None, bool, str | None, str | None] | None = None
        stopped = False
        if self.scope_id == STRUCTURE_STACK_NAME:
            if not await self._stage(backend, cwd, prompt_builder, 'integration', [f'integration:{self.scope_id}'],
                                     self.files, agent_kwargs, stage_inputs=stage_inputs):
                failure = self._failure()
                stopped = not self._isolated_failure()
        else:
            for index, batch in enumerate(self.assignment_batches):
                if failure is not None and not self._continuation_admitted(agent_kwargs):
                    stopping_failure = self._failure()
                    stopped = True
                    break
                targets = [part['target_id'] for part in batch]
                files = sorted({part['file'] for part in batch})
                if not await self._stage(backend, cwd, prompt_builder, 'first_pass', targets, files,
                                         agent_kwargs, stage_inputs=stage_inputs, assignment_parts=batch,
                                         future_batches=len(self.assignment_batches) - index - 1):
                    failure = failure or self._failure()
                    if not self._isolated_failure():
                        stopping_failure = self._failure()
                        stopped = True
                        break
        unsettled_ids = [candidate['candidate_id'] for candidate in self.handoff['candidates']
                         if candidate['disposition'] in {'open', 'unresolved'}]
        for index in range(0, len(unsettled_ids), 8):
            if stopped:
                break
            if failure is not None and not self._continuation_admitted(agent_kwargs):
                stopping_failure = self._failure()
                break
            if not await self._stage(backend, cwd, prompt_builder, 'triage', [], [], agent_kwargs,
                              stage_inputs=stage_inputs, candidate_ids=unsettled_ids[index:index + 8],
                              future_triage_stages=(len(unsettled_ids) - index - 1) // 8):
                failure = failure or self._failure()
                stopped = not self._isolated_failure()
                if stopped:
                    stopping_failure = self._failure()
        if failure is not None:
            self.reason, self.failed_invocation, self.failure_diagnostic, self.failure_class = (
                stopping_failure or failure)
            self.failed_invocation = self.failed_invocation or failure[1]
            for reason in dict.fromkeys((failure[0], self.reason)):
                if reason is not None:
                    try:
                        self.failure_reasons.append(ReasonCode(reason))
                    except ValueError:
                        self.failure_reasons.append(reason_for_budget(reason))
            self.failure_diagnostic = '\n'.join(dict.fromkeys(
                diagnostic for diagnostic in (failure[2], self.failure_diagnostic) if diagnostic)) or None
        if self.reason is None and (any(target['status'] != 'reviewed' for target in self.handoff['progress'])
                                    or any(candidate['disposition'] in {'open', 'unresolved'}
                                           for candidate in self.handoff['candidates'])):
            self.reason = 'evidence_incomplete'
            self.failure_class = 'admission_failure'
            self.failure_diagnostic = 'Stage coverage or candidate decisions remain incomplete: evidence_incomplete'
        return {'issues': [candidate['finding'] for candidate in self.handoff['candidates']
                           if candidate['disposition'] == 'confirmed']}, self.reason

    def _failure(self) -> tuple[str | None, bool, str | None, str | None]:
        return self.reason, self.failed_invocation, self.failure_diagnostic, self.failure_class

    def _isolated_failure(self) -> bool:
        """Only ordinary invocation failures permit a later independent batch."""
        return self.failed_invocation and self.reason == 'backend_failure' and self._continuable_failure

    @property
    def _deadline(self) -> float:
        """The scope ledger always has an absolute deadline."""
        assert self.budget.deadline is not None
        return self.budget.deadline

    @property
    def _remaining_calls(self) -> int:
        """The scope ledger always caps tool calls."""
        remaining = self.budget.remaining_tool_calls
        assert remaining is not None
        return remaining

    def _budget_stop_reason(self) -> str | None:
        """Use the original controlling bound when classifying an exhausted reviewer."""
        if self._remaining_calls <= 0:
            return 'tool_call_budget_exceeded'
        if clock.monotonic() >= self._deadline:
            return ('pipeline_budget_exceeded' if self.budget.deadline == self.budget.shared_deadline
                    else 'wall_budget_exceeded')
        return None

    def _refused(self, reason: str) -> str:
        """A refusal before dispatch leads with the scope ledger's stop facts, numbers first."""
        stop = self.budget.stop('tool_calls' if reason == 'tool_call_budget_exceeded' else 'time')
        return f'{stop.diagnostic()} Review limit exhausted: {reason}'

    def _continuation_admitted(self, agent_kwargs: dict[str, Any]) -> bool:
        """A new batch cannot evade shared stopping conditions after an invocation failure."""
        reason = self._budget_stop_reason()
        if reason is None:
            context = resolve_run_context(agent_kwargs.get('run_context'))
            if context.outage_circuit.admit_retry(clock.monotonic()).allowed:
                return True
            reason = 'backend_failure'
        self.reason = reason
        self.failed_invocation = False
        self.failure_class = 'quantitative_exhaustion' if reason != 'backend_failure' else 'outage_circuit'
        self.failure_diagnostic = (self._refused(reason) if reason != 'backend_failure' else
                                   'Review stopped: provider outage circuit denied a later batch.')
        return False

    def _stage_state(self, stage: str, targets: list[str], files: list[str],
                     candidate_ids: list[str], assignment_parts: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [candidate for candidate in self.handoff['candidates']
                      if candidate['candidate_id'] in candidate_ids]
        candidate_files = _candidate_source_files(candidates)
        relevant_files = candidate_files | set(files)
        notes = [note for note in self.handoff['notes']
                 if relevant_files.intersection(note['files'])]
        closed_decisions = _closed_decisions(self.handoff['candidates'])
        target = (max(4, 3 * len(candidate_ids)) if stage == 'triage' else
                  max(8, len(files)) if stage == 'integration' else max(4, 2 * len(files)))
        return {'contract_version': STAGED_REVIEW_CONTRACT, 'scope_id': self.scope_id,
            'analyzed_revision': self.revision, 'stage': stage, 'assigned_target_ids': targets,
            # Stage state offers selectors only; the host builds canonical references.
            'assignment_parts': [{key: value for key, value in part.items() if key != 'source_refs'}
                                 for part in assignment_parts],
            'source_options': self._reference_options(stage, targets),
            'completed_target_ids': [target for target, progress in self.target_progress.items()
                                     if progress['status'] == 'reviewed'],
            'assigned_files': sorted(candidate_files) if stage == 'triage' else files,
            'progress': self.handoff['progress'] if stage != 'triage' else [],
            'notes': [note['text'] for note in notes], 'note_context': _note_context(notes), 'candidates': candidates,
            'closed_decisions': closed_decisions,
            'closed_candidate_ids': [decision['candidate_id'] for decision in closed_decisions],
            'assigned_candidate_ids': candidate_ids,
            'target_diff_bytes': {path: self.diff_weights.get(path, 1) for path in files},
            'observed_tool_starts': self.budget.observed,
            'remaining_tool_calls': self._remaining_calls,
            'hard_tool_call_allowance': self._remaining_calls,
            'advisory_tool_call_target': min(target, self._remaining_calls)}

    def _reference_options(self, stage: str, targets: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Offer bounded source identities, never a claim that a source was read."""
        if stage == 'integration':
            # Whole-change judgment can cite useful assigned work without a
            # mandatory reference for every file or an exhaustive registry.
            choices = list(islice((option for part in self.assignments.values()
                                   for option in part['source_options']), 4))
            return {target: choices for target in targets}
        return {target: self.assignments.get(target, {}).get('source_options', []) for target in targets}

    def _expected_source_refs(self, stage: str, target: str) -> Iterable[dict[str, Any]]:
        """Prompt examples do not narrow the assigned whole-change identity."""
        parts = (self.assignments.values() if stage == 'integration' else
                 [self.assignments.get(target, {})])
        return (ref for part in parts for ref in part.get('source_refs', []))

    async def _stage(self, backend: Backend, cwd: Path, prompt_builder: Callable[[dict[str, Any]], str],
        stage: str, targets: list[str], files: list[str], agent_kwargs: dict[str, Any], *,
        stage_inputs: Callable[[dict[str, Any]], Awaitable[PreparedSanctionedInputs]],
        candidate_ids: list[str] | None = None, assignment_parts: list[dict[str, Any]] | None = None,
        future_batches: int = 0, future_triage_stages: int | None = None) -> bool:
        self.reason = None
        self.failure_diagnostic, self.failed_invocation, self.failure_class = None, False, None
        self._continuable_failure = False
        self.reason = self._budget_stop_reason()
        if self.reason is not None:
            self.failure_class = 'quantitative_exhaustion'
            self.failure_diagnostic = self._refused(self.reason)
            return False
        candidate_ids = candidate_ids or []
        assignment_parts = assignment_parts or []
        rejection: SchemaRejection | None = None
        for attempt in (1, 2):
            state = self._stage_state(stage, targets, files, candidate_ids, assignment_parts)
            self.failure_diagnostic, self.failed_invocation, self.failure_class = None, False, None
            state.update(attempt=attempt, max_attempts=2, schema_rejection=rejection.to_dict() if rejection else None)
            stage_schema = review_stage_schema(targets, candidate_ids, triage=stage == "triage")
            # Every backend receives the strict selector transport; host admission
            # tolerates absent supporting metadata and checks canonical output separately.
            transport_schema = review_stage_submission_schema(stage_schema)
            submission_schema = review_stage_admission_schema(transport_schema)
            canonical_schema = review_stage_admission_schema(stage_schema)
            native_admission = is_adapter_instance(backend, "PiBackend")
            # Only a host-tool submission can be corrected within the invocation.
            state['submission_transport'] = 'host_tool' if native_admission else 'final_output'
            state["response_contract"] = {"schema": transport_schema, "skeleton": {
                "targets": [{"target_id": target, "status": "reviewed", "reason": "", "evidence_refs": []}
                            for target in targets],
                "notes": "", "supporting_files": [], "candidates": [{
                    "candidate_id": candidate["candidate_id"], "file": candidate["file"], "line": candidate["line"],
                    "trigger": "", "consequence": "", "grounds": "", "disposition": "unresolved", "finding": None,
                } for candidate in state["candidates"]] if stage == "triage" else [], "contradictions": []}}
            # Pi returns the host-built canonical result; other backends return selectors for host admission.
            evidence = ReviewEvidence(canonical_schema if native_admission else submission_schema)
            self.assertion_failure_class: str | None = None
            self.assertion_failure_diagnostic: str | None = None
            syntax_error: dict[str, int] | None = None
            retry_rejection: SchemaRejection | None = None
            actual_rejection: SchemaRejection | None = None
            input_failure = ''
            accepted = False
            binding: HostToolBinding | None = None
            stage_budget: BudgetLedger | None = None
            starts_before = self.budget.observed
            candidate_count = 0
            async with phase_scope(DaydreamPhase.DEEP, stage=f'review-{stage.replace('_', '-')}') as transition:
                prior_admitted = self.admitted_stages
                try:
                    prepared = await stage_inputs(state)
                    kwargs = {**agent_kwargs, 'sanctioned_inputs': prepared}
                    prepared.revalidate(backend, cwd, kwargs.get('read_only', False))
                    if self._input_factory is None:
                        raise SanctionedInputUnavailable('Staged review requires frozen source admission')
                    assigned_files = files or state['assigned_files']

                    def admit(submitted: dict[str, Any], binding: HostToolBinding | None = None) -> dict[str, Any]:
                        return self._admit_submission(
                            submitted, stage, targets, candidate_ids, files=assigned_files,
                            canonical_schema=canonical_schema, binding=binding,
                            revalidate=lambda: prepared.revalidate(backend, cwd, kwargs.get('read_only', False)))

                    # This invocation's ledger charges the scope ledger; native admission narrows it below.
                    allowance: dict[str, int] = {}
                    # Submission transport is a prospective Pi planning hint; RequestEvent owns output authority.
                    if native_admission and not (
                        kwargs.get("tools_disabled") or kwargs.get("finalization")
                        or kwargs.get("validate_structured_output") is False):
                        state["max_attempts"] = 1
                        remaining = state.setdefault("remaining_work", {})
                        remaining["current_stage_submission_start_floor"] = 1
                        triage_stages = future_triage_stages if future_triage_stages is not None else (
                            sum(candidate['disposition'] in {'open', 'unresolved'}
                                for candidate in self.handoff['candidates']) + 7) // 8
                        exact = prepared.transport is SanctionedInputTransport.EXACT_PATHS
                        read_floor = (1 if state.get('supporting_bundle') else sum(
                            item.label in {'diff', 'hunk-index'} for item in prepared.inputs)) if exact else 0
                        floor = 1 + read_floor
                        reserve = future_batches * floor + triage_stages
                        remaining.update(remaining_submission_start_floor=1 + future_batches + triage_stages,
                                         current_stage_total_start_floor=floor,
                                         reserved_future_tool_starts=reserve)
                        state['hard_tool_call_allowance'] = max(0, self._remaining_calls - reserve)
                        allowance = {'limit': state['hard_tool_call_allowance'], 'reserved': reserve,
                                     'start_floor': floor}
                        state['advisory_tool_call_target'] = min(state['advisory_tool_call_target'],
                                                               state['hard_tool_call_allowance'])
                        if state['hard_tool_call_allowance'] < floor:
                            self.reason = 'tool_call_budget_exceeded'
                    stage_budget = self.budget.carve(f'stage:{self.scope_id}/{stage}', **allowance)
                    if native_admission:
                        def accept_submission(submitted: Any) -> dict[str, Any]:
                            """Prepare and install one whole stage without yielding authority."""
                            assert binding is not None
                            binding.ensure_open()
                            if not validates_schema(submitted, submission_schema):
                                raise HostToolRejected(
                                    'Review stage fields do not match response_contract.schema; '
                                    'supply every required judgment and remove unknown fields.')
                            return admit(submitted, binding)

                        binding = HostToolBinding(budget=stage_budget, accept=accept_submission)
                        kwargs['host_tools'] = binding
                    if self.reason is None:
                        stage_prompt = prompt_builder(state)
                        output, _, self.reason = await agent.run_agent(
                            backend, cwd, stage_prompt, phase=DaydreamPhase.DEEP, output_schema=transport_schema,
                            review_stage_scope_id=transition.scope_id,
                            review_system_instructions=build_review_stage_system_instruction(state),
                            require_full_schema=True, budget=stage_budget, review_evidence=evidence,
                            schema_rejection_guard=lambda value: self._retryable(
                                value, stage, targets, candidate_ids, files=assigned_files),
                            advisory_tool_call_target=state['advisory_tool_call_target'], **kwargs)
                    else:
                        output = None
                        stage_budget.stop('tool_calls')  # The native start floor refused this dispatch.
                    prepared.revalidate(backend, cwd, kwargs.get('read_only', False))
                except SanctionedInputUnavailable as exc:
                    self.reason = 'evidence_incomplete'
                    input_failure = f' ({redact_text(str(exc))[:240]})'
                except (asyncio.CancelledError, KeyboardInterrupt):
                    self.reason = 'interruption'
                    self.failure_class = 'cancellation'
                    raise
                except Exception as exc:  # noqa: BLE001 -- retain successful-stage progress after ordinary failures
                    self.reason = reason_for_exception(exc).value
                    self.failed_invocation = True
                    self._continuable_failure = classify_failure(exc).failure_class not in {
                        FailureClass.AUTH_CONFIG, FailureClass.SCHEMA, FailureClass.TOOL_POLICY, FailureClass.PERMANENT}
                    codex_diagnostic = None
                    if is_adapter_instance(exc, "CodexLifecycleError"):
                        from daydream.backends.codex import codex_lifecycle_diagnostic

                        codex_diagnostic = codex_lifecycle_diagnostic(exc)
                    self.failure_diagnostic = (codex_diagnostic
                        or f"{type(exc).__name__}: review invocation failed ({self.reason}).")
                    fatal = exc if isinstance(exc, HostToolFatal) else binding.fatal_error if binding else None
                    if fatal is not None:
                        self._record_fatal(fatal)
                else:
                    actual_rejection = output.rejection if isinstance(output, agent.StructuredOutputFailure) else None
                    syntax_error = output.syntax_error if isinstance(output, agent.StructuredOutputFailure) else None
                    candidate_count = len(output.get('candidates', [])) if isinstance(output, dict) else 0
                    if self.reason is None:
                        if (actual_rejection is not None and isinstance(output, agent.StructuredOutputFailure)
                                and output.schema_retry_eligible and not evidence.native_output):
                            retry_rejection = actual_rejection
                        if binding is not None:
                            if not binding.accepted or output != binding.accepted_result:
                                self.reason = 'evidence_incomplete'
                        else:
                            self.reason = self._admit(output, admit)
                    accepted = self.reason is None
                    if accepted:
                        assert isinstance(output, dict)
                        context = resolve_run_context(agent_kwargs.get("run_context"))
                        # Frozen selectors stay in raw evidence and typed coverage;
                        # their hashes must not crowd judgments out of bounded logs.
                        display = {**output, 'targets': [
                            {key: value for key, value in target.items() if key != 'evidence_refs'}
                            for target in output['targets']]}
                        present_result(agent.console, context.policy, agent_kwargs.get("progress_callback"), display)
                finally:
                    if self.reason in {'tool_call_budget_exceeded', 'wall_budget_exceeded', 'pipeline_budget_exceeded'}:
                        self.failure_class = 'quantitative_exhaustion'
                    elif self.assertion_failure_class is not None:
                        self.failure_class = self.assertion_failure_class
                    elif syntax_error is not None:
                        self.failure_class = "syntax_failure"
                    elif actual_rejection is not None:
                        self.failure_class = 'schema_rejection'
                    elif self.reason == 'evidence_incomplete':
                        self.failure_class = 'admission_failure'
                    elif self.reason == 'interruption':
                        self.failure_class = 'cancellation'
                    else:
                        self.failure_class = self.reason
                    if self.reason is not None and not self.failed_invocation:
                        labels = {'schema_rejection': 'Strict stage schema rejected',
                                  'syntax_failure': 'Invalid JSON syntax',
                                  'assignment_mismatch': 'Stage assignment identities rejected',
                                  'source_attribution': 'Frozen source attribution rejected',
                                  'admission_failure': 'Stage evidence admission failed',
                                  'quantitative_exhaustion': 'Review limit exhausted'}
                        label = labels.get(self.failure_class or '', 'Review stopped')
                        self.failure_diagnostic = f"{label}: {self.reason}{input_failure}"
                        if stage_budget is not None and stage_budget.stopped is not None:
                            # The invocation ledger's stop facts lead, so bounded coverage keeps them.
                            self.failure_diagnostic = f"{stage_budget.stopped.diagnostic()} {self.failure_diagnostic}"
                    transition.extra.update(
                        review_scope_id=self.scope_id, observed_tool_starts=self.budget.observed,
                        investigation_base_budget_s=self._base_limits.investigation_s,
                        investigation_scaled_budget_s=self._scaled_limits.investigation_s,
                        investigation_effective_allowance_s=self._effective_allowance_s,
                        investigation_limit=self._deadline_bound,
                        remaining_tool_calls=self.budget.remaining_tool_calls,
                        hard_tool_call_allowance=state['hard_tool_call_allowance'],
                        advisory_tool_call_target=state['advisory_tool_call_target'],
                        assigned_target_ids=targets, assigned_candidate_ids=candidate_ids or [],
                        admitted=self.admitted_stages > prior_admitted, stop_reason=self.reason,
                        logical_stage=stage, attempt=attempt,
                        attempt_tool_starts=self.budget.observed - starts_before,
                        schema_rejection=actual_rejection.to_dict() if actual_rejection else None,
                        schema_retry_eligible=retry_rejection is not None,
                        retry_feedback=rejection.to_dict() if rejection else None, native_output=evidence.native_output,
                        submission_starts=evidence.output_starts, successful_submissions=evidence.output_successes,
                        failed_submissions=evidence.output_failures,
                        replaced_submissions=max(0, evidence.output_successes - 1), failure_class=self.failure_class,
                        syntax_error=syntax_error, stage_candidate_count=candidate_count,
                        admitted_candidate_count=len(self.handoff['candidates']),
                        retained_findings=sum(item['disposition'] == 'confirmed'
                                              for item in self.handoff['candidates']))
                if self.reason in {'wall_budget_exceeded', 'pipeline_budget_exceeded'}:
                    transition.finish(LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT)
                elif not accepted:
                    transition.finish(LifecycleStatus.FAILED, LifecycleReasonCode.DOMAIN_FAILURE)
                else:
                    transition.finish(LifecycleStatus.SUCCEEDED)
            if accepted or attempt == 2 or retry_rejection is None:
                return accepted
            rejection = retry_rejection
            self.reason = None
        return False

    def _supporting_files(self, output: dict[str, Any]) -> list[str]:
        """Optional relevance references grant no access or source-read authority."""
        if self._source_root is None:
            return []
        files: set[str] = set()
        references = output.get('supporting_files')
        if not isinstance(references, list):
            return []
        for raw in references:
            try:
                path = canonicalize_repository_file_path(self._source_root, raw)
                # Generated artifacts and Git metadata are not supporting source.
                if path.split('/')[0] in {'.git', '.daydream', '.review-output.md'}:
                    continue
                if (self._source_root / path).is_file():
                    files.add(path)
            except (InvalidRepositoryFilePath, OSError):
                continue
        return sorted(files)

    def _assessment(self, output: dict[str, Any], stage: str, targets: list[str],
                    candidate_ids: list[str], *, files: list[str],
                    admitting: bool = False, native_sources: bool = False,
                    ) -> tuple[_AdmissionPlan | None, str | None]:
        """Assess available domain failures and bounds without admitting output or repairing unassessable fields."""
        raw_targets = output.get('targets')
        declared_targets = [item for item in raw_targets if isinstance(item, dict)] if (
            isinstance(raw_targets, list)) else []
        target_ids = [item.get('target_id') for item in declared_targets]
        known_target_ids = [item for item in target_ids if isinstance(item, str)]
        raw_decisions = output.get('candidates')
        decisions = [item for item in raw_decisions if isinstance(item, dict)] if (
            isinstance(raw_decisions, list)) else []
        prior_candidates = self.handoff['candidates']
        raw_contradictions = output.get('contradictions')
        contradictions = [item for item in raw_contradictions if isinstance(item, str)] if (
            isinstance(raw_contradictions, list)) else []
        closed_ids = {candidate['candidate_id'] for candidate in prior_candidates
                      if candidate['disposition'] in {'confirmed', 'rejected'}}
        decision_ids = [candidate.get('candidate_id') for candidate in decisions]
        known_decision_ids = [item for item in decision_ids if isinstance(item, str)]
        def reject(message: str, failure: str = 'assignment_mismatch',
                   reason: str = 'malformed_output') -> tuple[None, str]:
            self.assertion_failure_class = failure
            self.assertion_failure_diagnostic = message
            return None, reason

        for index, target in enumerate(declared_targets):
            identifier = target.get('target_id')
            if isinstance(identifier, str) and identifier not in targets:
                return reject(f'targets[{index}].target_id {redact_text(identifier)[:160]!r}: '
                              'outside this stage assignment; use each assigned_target_id exactly once.')
            if target.get('status') == 'not_reviewed' and isinstance(target.get('reason'), str) and not (
                    target['reason'].strip()):
                return reject(f'targets[{index}].reason: a not_reviewed judgment needs a nonblank explanation; '
                              'supply the concrete reason this target remains unreviewed.')
        if len(set(known_target_ids)) != len(known_target_ids):
            return reject('targets[].target_id: duplicate assigned target; supply each assigned_target_id once.')
        if (isinstance(raw_targets, list) and len(known_target_ids) == len(raw_targets)
                and sorted(known_target_ids) != sorted(targets)):
            missing = next((identifier for identifier in targets if identifier not in known_target_ids), '')
            return reject(f'targets: missing assigned target_id {redact_text(missing)[:160]!r}; '
                          'supply one reviewed or not_reviewed judgment for every assigned_target_id.')
        if len(set(contradictions)) != len(contradictions) or not set(contradictions) <= closed_ids:
            return reject('contradictions: IDs must be unique previously closed_candidate_ids; '
                          'remove duplicate or foreign IDs.')
        if stage == 'triage':
            for index, item in enumerate(decisions):
                if isinstance(item.get('candidate_id'), str) and item['candidate_id'] not in candidate_ids:
                    return reject(f'candidates[{index}].candidate_id: outside assigned_candidate_ids; '
                                  'use every assigned candidate exactly once.')
                if item.get('disposition') == 'open':
                    return reject(f'candidates[{index}].disposition: triage cannot reopen a candidate; '
                                  'choose confirmed, rejected or unresolved.')
            if (isinstance(raw_decisions, list) and len(known_decision_ids) == len(raw_decisions)
                    and sorted(known_decision_ids) != sorted(candidate_ids)):
                return reject('candidates: missing or duplicate assigned_candidate_ids; '
                              'supply exactly one judgment for every assigned candidate.')
        elif any(known_decision_ids):
            return reject('candidates[].candidate_id: new candidates have no host identity yet; '
                          'use an empty string for candidate_id outside triage.')
        for index, item in enumerate(decisions):
            if item.get('disposition') == 'confirmed' and 'finding' in item and item['finding'] is None:
                return reject(f'candidates[{index}].finding: confirmed requires a complete finding; '
                              'supply it or choose an unconfirmed disposition.')
            for field in ('grounds', 'trigger', 'consequence'):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    return reject(f'candidates[{index}].{field}: a nonblank judgment is required; '
                                  f'supply the concrete {field} for this candidate.',
                                  'admission_failure', 'evidence_incomplete')
        if contradictions:
            if admitting:
                self.assertion_failure_class = 'admission_failure'
                self.assertion_failure_diagnostic = (
                    'contradictions: source evidence contradicts a previously closed judgment; '
                    'retained findings stay admitted and affected work remains incomplete.')
                if native_sources:
                    raise HostToolFatal(self.assertion_failure_diagnostic, code='evidence_incomplete')
            return None, 'evidence_incomplete'
        can_prepare = (
            'notes' in output and isinstance(raw_targets, list) and isinstance(raw_decisions, list)
            and all(isinstance(item, dict) and isinstance(item.get('target_id'), str)
                    and isinstance(item.get('status'), str) for item in raw_targets)
            and all(isinstance(item, dict) and isinstance(item.get('file'), str)
                    and (stage != 'triage' or isinstance(item.get('candidate_id'), str)) for item in raw_decisions)
        )
        if not can_prepare:
            # Shape errors allow only independently proven bounds failures; never repair rejected output to make a plan.
            if 'notes' in output and len(json.dumps(output['notes'], ensure_ascii=False).encode()) > HANDOFF_MAX_BYTES:
                return None, 'evidence_incomplete'
            if isinstance(raw_decisions, list):
                prior = len(prior_candidates) if stage != 'triage' else 0
                if len(raw_decisions) + prior > HANDOFF_MAX_ITEMS:
                    return None, 'evidence_incomplete'
            return None, None
        if stage == 'triage':
            by_id = {candidate['candidate_id']: candidate for candidate in decisions}
            candidates = [by_id.get(candidate['candidate_id'], candidate) for candidate in prior_candidates]
        else:
            candidates = prior_candidates + [
                {**candidate, 'candidate_id': f'{self.scope_id}:candidate:{len(prior_candidates) + index + 1}'}
                for index, candidate in enumerate(decisions)]
        target_progress = {**self.target_progress, **{target['target_id']: target for target in output['targets']}}
        if native_sources:
            # Retained judgments cannot be rewritten to make room. Even marking
            # every new required target unreviewed still needs an ID and reason.
            minimum = {**self.target_progress, **{target['target_id']: {
                'target_id': target['target_id'], 'status': 'not_reviewed', 'reason': 'x', 'evidence_refs': [],
            } for target in output['targets']}}
            try:
                if (len(minimum) > TARGET_DECISIONS_MAX_ITEMS
                        or len(json.dumps(list(minimum.values()), ensure_ascii=False).encode())
                        > TARGET_DECISIONS_MAX_BYTES):
                    raise ValueError('required target identities cannot fit')
                self.coverage.check_target_decisions_budget(self.scope_id, list(minimum.values()))
            except ValueError as exc:
                raise HostToolFatal(
                    'targets: cumulative retained capacity cannot fit even the required target IDs with '
                    'minimal unreviewed judgments; earlier accepted judgments must remain intact. '
                    'This review invocation must stop.', code='capacity_exhausted') from exc
        try:
            if (len(target_progress) > TARGET_DECISIONS_MAX_ITEMS
                    or len(json.dumps(list(target_progress.values()), ensure_ascii=False).encode())
                    > TARGET_DECISIONS_MAX_BYTES):
                raise ValueError('target decisions exceed retained limits')
            if admitting:
                canonical = preserved_target_decisions(list(target_progress.values()))
                self.coverage.check_target_decisions_budget(self.scope_id, canonical)
                target_progress = {decision['target_id']: decision for decision in canonical}
        except ValueError:
            self.assertion_failure_class = 'source_attribution'
            self.assertion_failure_diagnostic = (
                f'targets: retained decisions exceed {TARGET_DECISIONS_MAX_ITEMS} items, '
                f'{TARGET_DECISIONS_MAX_BYTES} UTF-8 bytes per scope, or shared run capacity; '
                'shorten reasons and remove optional supporting evidence. Earlier accepted judgments are retained.')
            return None, 'evidence_incomplete'
        if stage == 'triage':
            progress = self.handoff['progress']
        elif stage == 'integration':
            progress = [target_progress[target['target_id']] for target in output['targets']]
        else:
            progress = []
            for path, required in self.required_targets.items():
                if not any(target in target_progress for target in required):
                    continue
                complete = all(target in target_progress and target_progress[target]['status'] == 'reviewed'
                               for target in required)
                progress.append({'target_id': path, 'status': 'reviewed' if complete else 'not_reviewed',
                                 'reason': '' if complete else 'Required assignment parts remain incomplete.'})
        supporting = self._supporting_files(output)
        if native_sources and (len(self.handoff['notes']) + len(prior_candidates) + 1
                + sum(len(note['supporting_files']) for note in self.handoff['notes']) > HANDOFF_MAX_ITEMS):
            raise HostToolFatal(
                f'The cumulative retained handoff has no capacity for the required stage note within '
                f'{HANDOFF_MAX_ITEMS} items; earlier accepted notes and candidates must remain intact. '
                'This review invocation must stop.', code='capacity_exhausted')
        handoff = {
            'progress': progress,
            'notes': self.handoff['notes'] + [{
                'files': sorted(set(files) | _candidate_source_files(decisions) | set(supporting)),
                'supporting_files': supporting, 'text': output['notes'],
                'origin': {'scope_id': self.scope_id, 'stage': stage, 'sequence': self.admitted_stages + 1,
                           'judgment_origin': 'reviewer', 'analyzed_revision': self.revision},
            }],
            'candidates': candidates,
        }
        if native_sources:
            # A lower bound drops current optional text and supporting paths,
            # retaining every required judgment. Failure even here is irreparable;
            # prior accepted assertions still cannot be discarded to free bytes.
            if stage == 'triage':
                minimum_progress = self.handoff['progress']
            elif stage == 'integration':
                minimum_progress = [{
                    'target_id': target['target_id'], 'status': 'not_reviewed', 'reason': 'x', 'evidence_refs': [],
                } for target in output['targets']]
            else:
                current_targets = set(targets)
                minimum_progress = []
                for path, required in self.required_targets.items():
                    if not any(target in target_progress for target in required):
                        continue
                    # Current judgments may be shortened or marked reviewed;
                    # missing parts and earlier unreviewed decisions stay so.
                    complete = all(target in target_progress and (target in current_targets
                                   or target_progress[target]['status'] == 'reviewed') for target in required)
                    minimum_progress.append({
                        'target_id': path, 'status': 'reviewed' if complete else 'not_reviewed',
                        'reason': '' if complete else 'Required assignment parts remain incomplete.',
                    })
            minimum_handoff = {**handoff, 'progress': minimum_progress, 'notes': self.handoff['notes'] + [{
                **handoff['notes'][-1], 'files': files, 'supporting_files': [], 'text': '',
            }], 'candidates': prior_candidates}
            if stage == 'triage' and candidate_ids:
                # Rejected adds a closed-decision projection; confirmed also
                # needs a finding. Unresolved is the smallest full projection.
                # Sharing one path minimizes row bytes plus the new note's file
                # set; an existing required path may beat a new one-character path.
                minimum_bytes = min(_handoff_bytes({
                    **minimum_handoff, 'notes': self.handoff['notes'] + [{
                        **minimum_handoff['notes'][-1], 'files': sorted(set(files) | {path}),
                    }], 'candidates': [{
                        'candidate_id': item['candidate_id'], 'file': path, 'line': 1,
                        'trigger': 'x', 'consequence': 'x', 'grounds': 'x',
                        'disposition': 'unresolved', 'finding': None,
                    } if item['candidate_id'] in candidate_ids else item for item in prior_candidates],
                }) for path in {'a', *(path for path in files if valid_repository_file_path(path))})
            else:
                minimum_bytes = _handoff_bytes(minimum_handoff)
            if minimum_bytes > HANDOFF_MAX_BYTES:
                raise HostToolFatal(
                    f'The cumulative retained handoff cannot fit the required stage context within '
                    f'{HANDOFF_MAX_BYTES} UTF-8 bytes even without optional new text; earlier accepted work '
                    'must remain intact. This review invocation must stop.', code='capacity_exhausted')
        if (len(handoff['notes']) + len(handoff['candidates'])
                + sum(len(note['supporting_files']) for note in handoff['notes']) > HANDOFF_MAX_ITEMS
                or _handoff_bytes(handoff) > HANDOFF_MAX_BYTES):
            self.assertion_failure_class = 'admission_failure'
            self.assertion_failure_diagnostic = (
                f'The cumulative retained handoff exceeds {HANDOFF_MAX_ITEMS} items or '
                f'{HANDOFF_MAX_BYTES} UTF-8 bytes across notes, candidates and supporting-file references; '
                'shorten this submission or remove optional candidates/context. Earlier accepted work is retained.')
            return None, 'evidence_incomplete'
        return _AdmissionPlan(handoff, target_progress), None

    def _retryable(self, value: Any, stage: str, targets: list[str], candidate_ids: list[str], *,
                   files: list[str]) -> bool:
        """A fresh schema retry must not disguise an independently invalid assignment or source selection."""
        if not isinstance(value, dict) or self._assessment(
                value, stage, targets, candidate_ids, files=files)[1] is not None:
            return False
        assert self._input_factory is not None
        # Selections are checked even when other fields leave the payload unpreparable.
        raw_targets = value.get('targets')
        for target in raw_targets if isinstance(raw_targets, list) else []:
            if not isinstance(target, dict) or not isinstance(target.get('target_id'), str) or (
                    'evidence_refs' not in target):
                continue
            try:
                self._input_factory.canonicalize_source_selectors(
                    target['evidence_refs'], self._expected_source_refs(stage, target['target_id']),
                    require_assigned=target.get('status') == 'reviewed')
            except HostToolRejected:
                self.assertion_failure_class = 'source_attribution'
                return False
        return True

    def _admit_submission(self, submitted: dict[str, Any], stage: str, targets: list[str],
                          candidate_ids: list[str], *, files: list[str], canonical_schema: dict[str, Any],
                          binding: HostToolBinding | None, revalidate: Callable[[], None]) -> dict[str, Any]:
        """Build canonical references from selectors, then install the whole stage atomically.

        Every backend shares this host path. Rejections raise HostToolRejected
        before any domain mutation; unavailable frozen input raises HostToolFatal.
        A host-tool binding makes irreparable capacity and contradiction failures
        fatal, since correction cannot fix them; without one, any rejection ends the stage.
        """
        assert self._input_factory is not None

        def prepare(sources: _SubmissionSources) -> Callable[[], dict[str, Any]]:
            canonical: dict[str, Any] = deepcopy(submitted)
            canonical['targets'] = []
            for index, target in enumerate(submitted['targets']):
                try:
                    references = sources.canonicalize(
                        target['evidence_refs'], self._expected_source_refs(stage, target['target_id']),
                        require_assigned=target['status'] == 'reviewed')
                except HostToolRejected as exc:
                    label = redact_text(target['target_id'])[:160]
                    raise HostToolRejected(f'targets[{index}] target_id {label!r}: {exc}') from exc
                canonical['targets'].append({**target, 'evidence_refs': references})
            canonical['supporting_files'] = self._supporting_files(submitted)
            serialized = json.dumps(canonical, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')
            if len(serialized) > _MAX_RESULT_BYTES:
                self.assertion_failure_class = 'admission_failure'
                raise HostToolRejected(
                    f'The whole canonical review result exceeds {_MAX_RESULT_BYTES} UTF-8 bytes; '
                    'shorten notes, reasons or candidate text and remove optional supporting '
                    'selectors. Assigned canonical hashes also count toward this bound.')
            if not validates_schema(canonical, canonical_schema):
                self.assertion_failure_class = 'admission_failure'
                raise HostToolRejected('Canonical review result does not match the required schema.')
            plan, _ = self._assessment(canonical, stage, targets, candidate_ids, files=files,
                                       admitting=True, native_sources=binding is not None)
            if plan is None:
                raise HostToolRejected(self.assertion_failure_diagnostic or (
                    'Review-stage fields cannot be retained; shorten optional text and '
                    'supply the assigned judgments from response_contract.'))
            revalidate()

            def install() -> dict[str, Any]:
                if binding is not None:
                    binding.ensure_open()
                self._install(plan)
                self.assertion_failure_class = None
                return canonical

            return install

        try:
            return self._input_factory.admit_submission(prepare)
        except SanctionedInputUnavailable as exc:
            raise HostToolFatal(
                f'Frozen review input validation failed: {redact_text(str(exc))}. '
                'this review invocation must stop.', code='input_unavailable') from exc

    def _admit(self, output: Any, admit: Callable[[dict[str, Any]], dict[str, Any]]) -> str | None:
        """Admit final output once; without host tools an invalid selection rejects the stage."""
        if not isinstance(output, dict):
            return (output.reason if isinstance(output, agent.StructuredOutputFailure)
                    else 'malformed_output' if output else 'missing_output')
        self.assertion_failure_class = None
        try:
            admit(output)
        except HostToolRejected:
            # Domain assessment classifies its own failures; anything earlier is source construction.
            self.assertion_failure_class = self.assertion_failure_class or 'source_attribution'
            return ('malformed_output' if self.assertion_failure_class == 'assignment_mismatch'
                    else 'evidence_incomplete')
        except HostToolFatal as exc:
            self._record_fatal(exc)
            return self.reason or 'evidence_incomplete'
        return None

    def _record_fatal(self, fatal: HostToolFatal) -> None:
        """An irreparable admission failure stops this invocation; admitted work stays retained."""
        self.failed_invocation = True
        self._continuable_failure = False
        self.failure_diagnostic = str(fatal)
        if fatal.code in {'input_unavailable', 'evidence_incomplete'}:
            self.reason = 'evidence_incomplete'
        elif isinstance(fatal, HostToolCallBudgetExceeded):
            self.reason = 'tool_call_budget_exceeded'

    def _install(self, plan: _AdmissionPlan) -> None:
        self.coverage.admit_target_decisions(self.scope_id, list(plan.target_progress.values()))
        self.admitted_stages += 1
        self.handoff = plan.handoff
        self.target_progress = plan.target_progress
