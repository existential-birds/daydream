"""Bounded, in-memory progress for one snapshot-bound public reviewer scope."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream import agent
from daydream.backends import Backend
from daydream.backends.pi import PiBackend
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.deep.sharding import file_change_bytes, pack_file_batches
from daydream.json_utils import SchemaRejection, validates_schema
from daydream.phases.review_prompts import build_review_stage_system_instruction
from daydream.phases.schemas import review_stage_schema
from daydream.prompt_budget import PreparedSanctionedInputs, SanctionedInputUnavailable, SourceAccessUnavailable
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.review_evidence import (
    FULL_REVIEW_MAX_BYTES,
    EvidenceReceipt,
    ReviewEvidence,
    source_window_covered,
)
from daydream.review_result import reason_for_exception
from daydream.trajectory import DaydreamPhase, LifecycleReasonCode, LifecycleStatus, phase_scope

STAGED_REVIEW_CONTRACT = 7
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


@dataclass(frozen=True)
class _AdmissionPlan:
    """Prospective host state; constructing this never admits model output."""

    handoff: dict[str, list[Any]]
    target_progress: dict[str, dict[str, Any]]
    receipts: list[EvidenceReceipt]
    compact_clipped: bool


class ReviewInvestigation:
    """Admit successful stage assertions; observed spend belongs to the host."""

    def __init__(self, stack: StackAssignment, full_diff: str, revision: dict[str, Any], *,
                 assignment_batches: list[list[dict[str, Any]]] | None = None) -> None:
        self.scope_id = stack.stack_name
        self.files = sorted(stack.files)
        self.revision = revision
        self.diff_weights = file_change_bytes(full_diff)
        # Keep nearby changed paths together without another dependency-planning
        # pass. Smaller hunks leave evidence room for their enclosing symbols.
        directories: dict[str, list[str]] = {}
        for path in self.files:
            directories.setdefault(str(Path(path).parent), []).append(path)
        self.batches = ([] if self.scope_id == STRUCTURE_STACK_NAME else
                        pack_file_batches(stack, self.diff_weights, 4, 16 * 1024, list(directories.values())))
        self.assignment_batches: list[list[dict[str, Any]]] = (
            assignment_batches if assignment_batches is not None else
            [[{'target_id': path, 'file': path, 'part_index': 1, 'part_count': 1, 'kind': 'file'}
              for path in batch.files] for batch in self.batches]
        )
        self.required_targets: dict[str, list[str]] = {path: [] for path in self.files}
        for batch in self.assignment_batches:
            for part in batch:
                self.required_targets[part['file']].append(part['target_id'])
        self.target_progress: dict[str, dict[str, Any]] = {}
        self.admitted_receipts: list[EvidenceReceipt] = []
        self.admitted_retained_bytes = 0
        self.handoff: dict[str, list[Any]] = {name: [] for name in ('progress', 'notes', 'candidates', 'evidence')}
        self.budget = ReviewInvestigationBudget.from_limits(ReviewLimits())
        self.reason: str | None = None
        self.admitted_stages = 0
        self.failed_invocation = False
        self.failure_diagnostic: str | None = None
        self.failure_class: str | None = None

    async def run(
        self, backend: Backend, cwd: Path, prompt_builder: Callable[[dict[str, Any]], str], *,
        stage_inputs: Callable[[dict[str, Any]], PreparedSanctionedInputs | None] | None = None,
        **agent_kwargs: Any,
    ) -> tuple[dict[str, Any], str | None]:
        """Fresh calls share hard bounds; advisory targets cannot abort useful work."""
        if self.scope_id == STRUCTURE_STACK_NAME:
            await self._stage(backend, cwd, prompt_builder, 'integration', [f'integration:{self.scope_id}'],
                              self.files, agent_kwargs, stage_inputs=stage_inputs)
        else:
            for batch in self.assignment_batches:
                targets = [part['target_id'] for part in batch]
                files = sorted({part['file'] for part in batch})
                if not await self._stage(backend, cwd, prompt_builder, 'first_pass', targets, files,
                                         agent_kwargs, stage_inputs=stage_inputs, assignment_parts=batch):
                    break
        open_ids = [candidate['candidate_id'] for candidate in self.handoff['candidates']
                    if candidate['disposition'] == 'open']
        for index in range(0, len(open_ids), 8):
            if self.reason is not None:
                break
            await self._stage(backend, cwd, prompt_builder, 'triage', [], [], agent_kwargs,
                              stage_inputs=stage_inputs, candidate_ids=open_ids[index:index + 8])
        if self.reason is None and (any(target['status'] != 'reviewed' for target in self.handoff['progress'])
                                    or any(candidate['disposition'] in {'open', 'unresolved'}
                                           for candidate in self.handoff['candidates'])):
            self.reason = 'evidence_incomplete'
            self.failure_class = 'admission_failure'
            self.failure_diagnostic = 'Stage coverage or candidate decisions remain incomplete: evidence_incomplete'
        return {'issues': [candidate['finding'] for candidate in self.handoff['candidates']
                           if candidate['disposition'] == 'confirmed']}, self.reason

    def _stage_state(self, stage: str, targets: list[str], files: list[str],
                     candidate_ids: list[str], assignment_parts: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [candidate for candidate in self.handoff['candidates']
                      if candidate['candidate_id'] in candidate_ids]
        candidate_files = _candidate_source_files(candidates)
        relevant_files = candidate_files | set(files)
        # Only admitted context relevant to this triage assignment travels into
        # its fresh request. Closed decisions expose handles, never fresh work.
        notes = [note['text'] for note in self.handoff['notes']
                 if relevant_files.intersection(note['files'])]
        evidence = ReviewEvidence(None).compact_for_files(relevant_files, receipts=self.admitted_receipts)
        target = (max(4, 3 * len(candidate_ids)) if stage == 'triage' else
                  max(8, len(files)) if stage == 'integration' else max(4, 2 * len(files)))
        return {
            'contract_version': STAGED_REVIEW_CONTRACT, 'scope_id': self.scope_id,
            'analyzed_revision': self.revision, 'stage': stage, 'assigned_target_ids': targets,
            'assignment_parts': assignment_parts,
            'completed_target_ids': [target for target, progress in self.target_progress.items()
                                     if progress['status'] == 'reviewed'],
            'admitted_source_windows': [receipt.source.metadata() for receipt in self.admitted_receipts
                                        if receipt.complete and receipt.source is not None
                                        and receipt.source.file in relevant_files],
            'assigned_files': sorted(candidate_files) if stage == 'triage' else files,
            'progress': self.handoff['progress'] if stage != 'triage' else [],
            'notes': notes, 'candidates': candidates, 'evidence': evidence,
            'closed_candidate_ids': [candidate['candidate_id'] for candidate in self.handoff['candidates']
                                     if candidate['disposition'] in {'confirmed', 'rejected'}],
            'assigned_candidate_ids': candidate_ids,
            'target_diff_bytes': {path: self.diff_weights.get(path, 1) for path in files},
            'observed_tool_starts': self.budget.observed_tool_starts,
            'remaining_tool_calls': self.budget.remaining_tool_calls,
            'advisory_tool_call_target': min(target, self.budget.remaining_tool_calls),
        }

    async def _stage(
        self, backend: Backend, cwd: Path, prompt_builder: Callable[[dict[str, Any]], str],
        stage: str, targets: list[str], files: list[str], agent_kwargs: dict[str, Any], *,
        stage_inputs: Callable[[dict[str, Any]], PreparedSanctionedInputs | None] | None,
        candidate_ids: list[str] | None = None,
        assignment_parts: list[dict[str, Any]] | None = None,
    ) -> bool:
        if self.budget.remaining_tool_calls <= 0:
            self.reason = 'tool_call_budget_exceeded'
            self.failure_class = 'quantitative_exhaustion'
            self.failure_diagnostic = 'Review limit exhausted: tool_call_budget_exceeded'
            return False
        rejection: SchemaRejection | None = None
        for attempt in (1, 2):
            accepted, retry_rejection = await self._attempt(
                backend, cwd, prompt_builder, stage, targets, files, agent_kwargs,
                stage_inputs=stage_inputs, candidate_ids=candidate_ids or [],
                assignment_parts=assignment_parts or [], attempt=attempt, rejection=rejection,
            )
            if accepted or attempt == 2 or retry_rejection is None:
                return accepted
            rejection = retry_rejection
            self.reason = None
        return False

    async def _attempt(
        self, backend: Backend, cwd: Path, prompt_builder: Callable[[dict[str, Any]], str],
        stage: str, targets: list[str], files: list[str], agent_kwargs: dict[str, Any], *,
        stage_inputs: Callable[[dict[str, Any]], PreparedSanctionedInputs | None] | None,
        candidate_ids: list[str], assignment_parts: list[dict[str, Any]],
        attempt: int, rejection: SchemaRejection | None,
    ) -> tuple[bool, SchemaRejection | None]:
        state = self._stage_state(stage, targets, files, candidate_ids, assignment_parts)
        self.failure_diagnostic = None
        self.failed_invocation = False
        self.failure_class = None
        if attempt > 1:
            state["admitted_source_windows"] = []
        state.update(attempt=attempt, max_attempts=2,
                     schema_rejection=rejection.to_dict() if rejection else None)
        schema = review_stage_schema(targets, candidate_ids, triage=stage == "triage")
        state["response_contract"] = {"schema": schema, "skeleton": {
            "targets": [{"target_id": target, "status": "reviewed", "reason": ""} for target in targets],
            "notes": "", "candidates": [{
                "candidate_id": candidate["candidate_id"], "file": candidate["file"], "line": candidate["line"],
                "trigger": "", "consequence": "", "grounds": "", "disposition": "unresolved", "finding": None,
            } for candidate in state["candidates"]] if stage == "triage" else [], "contradictions": [],
        }}
        evidence = ReviewEvidence(schema)
        self.assertion_failure_class: str | None = None
        syntax_error: dict[str, int] | None = None
        retry_rejection: SchemaRejection | None = None
        actual_rejection: SchemaRejection | None = None
        accepted = False
        starts_before = self.budget.observed_tool_starts
        candidate_count = 0
        identity_failure = False
        source_access_failure = False
        prepared: PreparedSanctionedInputs | None = None
        async with phase_scope(DaydreamPhase.DEEP, stage=f'review-{stage.replace('_', '-')}') as transition:
            prior_admitted = self.admitted_stages
            try:
                kwargs = dict(agent_kwargs)
                if stage_inputs is not None:
                    kwargs['sanctioned_inputs'] = stage_inputs(state)
                prepared = kwargs.get('sanctioned_inputs')
                if prepared is not None:
                    prepared.revalidate(backend, cwd, kwargs.get('read_only', False))
                reused: list[EvidenceReceipt] = []
                if prepared is not None:
                    kwargs["source_recipe"] = prepared.source_recipe
                    recipe = prepared.source_recipe
                    if recipe is not None and attempt == 1:
                        reused = [receipt for receipt in self.admitted_receipts
                                  if receipt.complete and receipt.source is not None
                                  and receipt.source.file in state["assigned_files"] and any(
                                      receipt.source.file == window.file and receipt.source.side == window.side
                                      and receipt.source.revision == window.revision
                                      and receipt.source.content_sha256 == window.content_sha256
                                      and receipt.source.blob_oid == window.blob_oid
                                      and receipt.source.source_path == window.source_path
                                      for window in recipe.windows)]
                        for access in state.get("source_access", []):
                            window = next((window for window in recipe.windows
                                           if window.file == access["file"] and window.side == access["side"]
                                           and window.start_byte == access["start_byte"]
                                           and window.end_byte == access["end_byte"]
                                           and set(access["target_ids"]) <= set(window.target_ids)), None)
                            if (window is not None and recipe.verify_window(window)
                                    and source_window_covered(window, reused)):
                                access["read_required"] = False
                                access["reuse"] = "admitted complete frozen source coverage"
                        state["admitted_source_windows"] = [receipt.source.metadata() for receipt in reused
                                                            if receipt.source is not None]
                    if recipe is not None:
                        remaining = state.setdefault("remaining_work", {})
                        required = [window for window in recipe.windows
                                    if window.read_required and window.file in state["assigned_files"]]
                        remaining["current_stage_source_windows"] = 1 if stage == "integration" else len(required)
                        remaining["current_stage_fresh_source_windows"] = (
                            int(not any(source_window_covered(window, reused) for window in recipe.windows))
                            if stage == "integration" else
                            sum(not source_window_covered(window, reused) for window in required))
                        supporting_reads = (0 if stage == "triage" or prepared.transport.value == "inline" else
                                            1 if state.get("supporting_bundle") else 3)
                        remaining["current_stage_mandatory_read_estimate"] = (
                            supporting_reads + remaining["current_stage_fresh_source_windows"])
                        remaining["read_estimate_scope"] = (
                            "Transport lower bound only; candidate and dependency investigation are additional.")
                # Planning floor only; actual output authority comes from RequestEvent.
                if isinstance(backend, PiBackend) and not (
                    kwargs.get("tools_disabled") or kwargs.get("finalization")
                    or kwargs.get("validate_structured_output") is False
                ):
                    state["max_attempts"] = 1
                    remaining = state.setdefault("remaining_work", {})
                    remaining["current_stage_submission_start_floor"] = 1
                    triage_stages = (sum(candidate['disposition'] == 'open'
                                         for candidate in self.handoff['candidates']) + 7) // 8
                    remaining["remaining_submission_start_floor"] = max(
                        1, remaining.get("stages", 1) + triage_stages)
                    remaining["current_stage_total_start_floor"] = (
                        remaining.get("current_stage_mandatory_read_estimate", 0) + 1)
                evidence.configure_capture(cwd, allowance=FULL_REVIEW_MAX_BYTES - self.admitted_retained_bytes,
                                           sanctioned_inputs=prepared, reused_receipts=reused,
                                           snapshot_revisions=tuple(self.revision[key] for key in
                                                                    ('head_sha', 'merge_base_sha')))
                stage_prompt = prompt_builder(state)
                output, _, self.reason = await agent.run_agent(
                    backend, cwd, stage_prompt, phase=DaydreamPhase.DEEP, output_schema=schema,
                    review_system_instructions=build_review_stage_system_instruction(state),
                    require_full_schema=True, investigation_budget=self.budget, review_evidence=evidence,
                    schema_rejection_guard=lambda value: self._schema_retry_eligible(
                        value, stage, targets, candidate_ids, evidence, files=files or state['assigned_files']),
                    advisory_tool_call_target=state['advisory_tool_call_target'], **kwargs,
                )
                if prepared is not None:
                    prepared.revalidate(backend, cwd, kwargs.get('read_only', False))
            except SourceAccessUnavailable:
                self.reason = "evidence_incomplete"
                source_access_failure = True
                accepted = False
            except SanctionedInputUnavailable:
                self.reason = 'evidence_incomplete'
                identity_failure = True
                accepted = False
            except (asyncio.CancelledError, KeyboardInterrupt):
                self.reason = 'interruption'
                self.failure_class = 'cancellation'
                raise
            except Exception as exc:  # noqa: BLE001 -- retain successful-stage progress after ordinary failures
                self.reason = reason_for_exception(exc).value
                self.failed_invocation = True
                self.failure_diagnostic = f"{type(exc).__name__}: review invocation failed ({self.reason})."
                accepted = False
            else:
                actual_rejection = output.rejection if isinstance(output, agent.StructuredOutputFailure) else None
                syntax_error = output.syntax_error if isinstance(output, agent.StructuredOutputFailure) else None
                candidate_count = len(output.get('candidates', [])) if isinstance(output, dict) else 0
                if self.reason is None:
                    if (actual_rejection is not None and isinstance(output, agent.StructuredOutputFailure)
                            and output.schema_retry_eligible and not evidence.native_output
                            and not evidence.capture_failure(files or state['assigned_files'], require_source=False,
                                                             interaction=stage == 'integration')):
                        retry_rejection = actual_rejection
                    self.reason = self._admit(output, stage, targets, candidate_ids or [], evidence,
                                              files=files or state['assigned_files'])
                accepted = self.reason is None
            finally:
                if self.reason in {'tool_call_budget_exceeded', 'wall_budget_exceeded', 'pipeline_budget_exceeded',
                                   'model_budget_exhaustion'}:
                    self.failure_class = 'quantitative_exhaustion'
                elif source_access_failure or evidence.source_access_failures:
                    self.failure_class = "source_access_failure"
                elif self.assertion_failure_class is not None:
                    self.failure_class = self.assertion_failure_class
                elif syntax_error is not None:
                    self.failure_class = "syntax_failure"
                elif actual_rejection is not None:
                    self.failure_class = 'schema_rejection'
                elif self.reason == 'evidence_incomplete':
                    self.failure_class = ('capture_loss' if identity_failure or evidence.capture_failure(
                        files or state['assigned_files'], require_source=False,
                        interaction=stage == 'integration') else 'missing_source_receipt')
                elif self.reason == 'interruption':
                    self.failure_class = 'cancellation'
                else:
                    self.failure_class = self.reason
                if self.reason is not None and not self.failed_invocation:
                    labels = {'schema_rejection': 'Strict stage schema rejected',
                              'syntax_failure': 'Invalid JSON syntax',
                              'assignment_mismatch': 'Stage assignment identities rejected',
                              'missing_source_receipt': 'Required source receipt missing',
                              'source_access_failure': 'Frozen source access unavailable or rejected',
                              'capture_loss': 'Review evidence capture incomplete',
                              'admission_failure': 'Stage evidence admission failed',
                              'quantitative_exhaustion': 'Review limit exhausted'}
                    self.failure_diagnostic = f"{labels.get(self.failure_class or '', 'Review stopped')}: {self.reason}"
                transition.extra.update(
                    review_scope_id=self.scope_id, observed_tool_starts=self.budget.observed_tool_starts,
                    remaining_tool_calls=self.budget.remaining_tool_calls,
                    hard_tool_call_allowance=state['remaining_tool_calls'],
                    advisory_tool_call_target=state['advisory_tool_call_target'],
                    assigned_target_ids=targets, assigned_candidate_ids=candidate_ids or [],
                    admitted=self.admitted_stages > prior_admitted, stop_reason=self.reason,
                    logical_stage=stage, attempt=attempt,
                    attempt_tool_starts=self.budget.observed_tool_starts - starts_before,
                    schema_rejection=actual_rejection.to_dict() if actual_rejection else None,
                    schema_retry_eligible=retry_rejection is not None,
                    retry_feedback=rejection.to_dict() if rejection else None,
                    full_retained_bytes=evidence.full_retained_bytes,
                    sanctioned_input_count=len(prepared.inputs) if prepared else 0,
                    sanctioned_input_bytes=sum(item.size for item in prepared.inputs if not item.pointer_only)
                    if prepared else 0,
                    compact_view_clipped=evidence.clipped,
                    native_truncated_results=evidence.native_truncated_results,
                    source_access_failures=evidence.source_access_failures + int(source_access_failure),
                    reused_source_windows=sum(not access.get("read_required", True) and "reuse" in access
                                              for access in state.get("source_access", [])),
                    native_output=evidence.native_output,
                    submission_starts=evidence.output_starts,
                    successful_submissions=evidence.output_successes,
                    failed_submissions=evidence.output_failures,
                    replaced_submissions=max(0, evidence.output_successes - 1),
                    fresh_source_reads=sum(receipt.complete and receipt.source is not None
                                           for receipt in evidence.receipts),
                    before_source_reads=sum(receipt.complete and receipt.source is not None
                                            and receipt.source.side == "before" for receipt in evidence.receipts),
                    base_only_source_reads=sum(
                        receipt.complete and receipt.source is not None and receipt.source.side == "before"
                        and evidence.source_recipe is not None
                        and not any(window.side == "after" and window.source_path == receipt.source.source_path
                                    for window in evidence.source_recipe.windows) for receipt in evidence.receipts),
                    source_body_bytes=sum(len(receipt.source.body.encode()) for receipt in evidence.receipts
                                          if receipt.complete and receipt.source is not None),
                    after_source_reads=sum(receipt.complete and receipt.source is not None
                                           and receipt.source.side == "after" for receipt in evidence.receipts),
                    retention_overflow_results=evidence.retention_overflow_results,
                    unmatched_results=evidence.unmatched_results,
                    nonblocking_unavailable_reads=evidence.nonblocking_unavailable_reads,
                    blocking_opaque_receipts=evidence.blocking_opaque_receipts,
                    blocking_pending_receipts=evidence.blocking_pending_receipts,
                    failure_class=self.failure_class,
                    syntax_error=syntax_error,
                    stage_candidate_count=candidate_count,
                    admitted_candidate_count=len(self.handoff['candidates']),
                    retained_findings=sum(item['disposition'] == 'confirmed' for item in self.handoff['candidates']),
                )
            if self.reason in {'wall_budget_exceeded', 'pipeline_budget_exceeded'}:
                transition.finish(LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT)
            elif not accepted:
                transition.finish(LifecycleStatus.FAILED, LifecycleReasonCode.DOMAIN_FAILURE)
            else:
                transition.finish(LifecycleStatus.SUCCEEDED)
            return accepted, retry_rejection

    def _assertion_failure(self, output: dict[str, Any], stage: str, targets: list[str], candidate_ids: list[str],
                           evidence: ReviewEvidence, *, files: list[str]) -> str | None:
        """Check proven domain failures without widening the independent strict schema gate.

        Schema-rejected objects may lack a known field or carry the wrong type.
        Unassessable fields alone are schema errors; available assertions still
        cannot override assignment identity, grounded evidence, or closed decisions.
        """
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
        if (
            any(identifier not in targets for identifier in known_target_ids)
            or len(set(known_target_ids)) != len(known_target_ids)
            or (isinstance(raw_targets, list) and len(known_target_ids) == len(raw_targets)
                and sorted(known_target_ids) != sorted(targets))
            or any(target.get('status') == 'not_reviewed' and isinstance(target.get('reason'), str)
                   and not target['reason'].strip() for target in declared_targets)
            or len(set(contradictions)) != len(contradictions)
            or not set(contradictions) <= closed_ids
            or (stage == 'triage' and (any(identifier not in candidate_ids for identifier in known_decision_ids)
                                      or (isinstance(raw_decisions, list)
                                          and len(known_decision_ids) == len(raw_decisions)
                                          and sorted(known_decision_ids) != sorted(candidate_ids))
                                      or any(candidate.get('disposition') == 'open' for candidate in decisions)))
            or (stage != 'triage' and any(known_decision_ids))
            or any(candidate.get('disposition') == 'confirmed' and 'finding' in candidate
                   and candidate['finding'] is None
                   for candidate in decisions)
        ):
            self.assertion_failure_class = 'assignment_mismatch'
            return 'malformed_output'
        if decisions and any(
            not isinstance(candidate.get(field), str) or not candidate[field].strip()
            for candidate in decisions for field in ('grounds', 'trigger', 'consequence')
        ):
            self.assertion_failure_class = 'admission_failure'
            return 'evidence_incomplete'
        reviewed_ids = {target['target_id'] for target in declared_targets
                        if target.get('status') == 'reviewed' and isinstance(target.get('target_id'), str)}
        reviewed_files = ([] if not reviewed_ids else files if stage == 'integration' else
                          [path for path in files if any(target in reviewed_ids
                                                        for target in self.required_targets[path])])
        candidate_files = _candidate_source_files(decisions)
        grounding_files = sorted(set(reviewed_files) | candidate_files)
        if grounding_files:
            if evidence.capture_failure(grounding_files, require_source=False):
                return 'evidence_incomplete'
        if reviewed_files and evidence.capture_failure(
            grounding_files if stage == 'integration' else reviewed_files,
            interaction=stage == 'integration',
        ):
            return 'evidence_incomplete'
        # Whole-change interaction evidence is deliberately broader than a
        # language/file audit. Each defect claim still needs its own associated
        # source, rather than borrowing an unrelated receipt from that union.
        receipts = [*evidence.reused_receipts, *evidence.receipts]
        grounded = {path for receipt in receipts if receipt.complete and not receipt.supporting
                    for path in receipt.paths}
        if not candidate_files <= grounded:
            return 'evidence_incomplete'
        return None

    def _schema_retry_eligible(self, output: Any, stage: str, targets: list[str], candidate_ids: list[str],
                               evidence: ReviewEvidence, *, files: list[str]) -> bool:
        """Inspect the original rejected object; never repair, publish, or retain it."""
        if not isinstance(output, dict) or self._assertion_failure(
            output, stage, targets, candidate_ids, evidence, files=files
        ) is not None:
            return False
        if evidence.capture_failure(files, interaction=stage == 'integration'):
            self.assertion_failure_class = ('capture_loss' if evidence.capture_failure(
                files, require_source=False, interaction=stage == 'integration') else 'missing_source_receipt')
            return False
        contradictions = output.get('contradictions')
        if isinstance(contradictions, list) and any(isinstance(item, str) for item in contradictions):
            return False
        raw_targets = output.get('targets')
        raw_candidates = output.get('candidates')
        can_prepare = (
            'notes' in output and isinstance(raw_targets, list) and isinstance(raw_candidates, list)
            and all(isinstance(item, dict) and isinstance(item.get('target_id'), str)
                    and isinstance(item.get('status'), str) for item in raw_targets)
            and all(isinstance(item, dict) and isinstance(item.get('file'), str)
                    and (stage != 'triage' or isinstance(item.get('candidate_id'), str)) for item in raw_candidates)
        )
        if can_prepare:
            _, reason = self._prepare_handoff(output, stage, targets, evidence, files=files)
            return reason is None
        # When schema-only shape errors prevent a prospective plan, block only
        # independently proven bounds failures, rather than inventing another
        # shape validator or repairing the rejected payload into a usable result.
        if 'notes' in output and len(json.dumps(output['notes'], ensure_ascii=False).encode()) > HANDOFF_MAX_BYTES:
            return False
        if isinstance(raw_candidates, list):
            prior = len(self.handoff['candidates']) if stage != 'triage' else 0
            if len(raw_candidates) + prior > HANDOFF_MAX_ITEMS:
                return False
        return True

    def _prepare_handoff(self, output: dict[str, Any], stage: str, targets: list[str],
                         evidence: ReviewEvidence, *, files: list[str]) -> tuple[_AdmissionPlan | None, str | None]:
        """Construct prospective state for bounds checks without changing admitted state."""
        decisions = output['candidates']
        prior_candidates = self.handoff['candidates']
        if stage == 'triage':
            by_id = {candidate['candidate_id']: candidate for candidate in decisions}
            candidates = [by_id.get(candidate['candidate_id'], candidate) for candidate in prior_candidates]
        else:
            candidates = prior_candidates + [
                {**candidate, 'candidate_id': f'{self.scope_id}:candidate:{len(prior_candidates) + index + 1}'}
                for index, candidate in enumerate(decisions)
            ]
        target_progress = {**self.target_progress,
                           **{target['target_id']: target for target in output['targets']}}
        if stage == 'triage':
            progress = self.handoff['progress']
        elif stage == 'integration':
            progress = output['targets']
        else:
            progress = []
            for path, required in self.required_targets.items():
                if not any(target in target_progress for target in required):
                    continue
                complete = all(target in target_progress and target_progress[target]['status'] == 'reviewed'
                               for target in required)
                progress.append({'target_id': path, 'status': 'reviewed' if complete else 'not_reviewed',
                                 'reason': '' if complete else 'Required assignment parts remain incomplete.'})
        # Full associated receipts remain the admission authority. Compact views
        # may omit late blocks without turning otherwise complete reads into loss.
        compact = list(self.handoff['evidence'])
        compact_bytes = len(json.dumps(compact, ensure_ascii=False).encode())
        compact_clipped = False
        relevant_files = _candidate_source_files(decisions) | set(files)
        relevant_blocks = evidence.compact_for_files(relevant_files)
        for index, block in enumerate(relevant_blocks):
            entry = {'stage': stage, 'assigned_target_ids': targets, **block}
            size = len(json.dumps(entry, ensure_ascii=False).encode())
            if compact_bytes + size > 48000 or len(compact) >= 64:
                compact_clipped = True
                if compact:
                    compact[-1] = {**compact[-1], 'partial': True,
                                   'omitted_receipts': len(relevant_blocks) - index}
                else:
                    compact.append({'stage': stage, 'assigned_target_ids': targets,
                                    'files': sorted(relevant_files), 'partial': True,
                                    'omitted_receipts': len(relevant_blocks), 'excerpt': '[partial evidence view] '})
                break
            compact.append(entry)
            compact_bytes += size
        handoff = {
            'progress': progress,
            'notes': self.handoff['notes'] + [{
                'files': sorted(set(files) | relevant_files | {
                    receipt.source.file for receipt in evidence.receipts
                    if receipt.complete and receipt.source is not None
                }), 'text': output['notes'],
            }],
            'candidates': candidates, 'evidence': compact,
        }
        if (sum(map(len, handoff.values())) > HANDOFF_MAX_ITEMS
                or len(json.dumps(handoff, ensure_ascii=False).encode('utf-8')) > HANDOFF_MAX_BYTES):
            self.assertion_failure_class = 'admission_failure'
            return None, 'evidence_incomplete'
        receipts = [receipt for receipt in evidence.receipts
                    if receipt.complete and (receipt.source is not None
                                             or relevant_files.intersection(receipt.paths))]
        return _AdmissionPlan(handoff, target_progress, receipts, compact_clipped), None

    def _admit(self, output: Any, stage: str, targets: list[str], candidate_ids: list[str],
               evidence: ReviewEvidence, *, files: list[str]) -> str | None:
        """Commit a bounded plan only after the full strict schema and domain gates."""
        if not isinstance(output, dict) or not validates_schema(output, evidence.schema or {}):
            return (output.reason if isinstance(output, agent.StructuredOutputFailure)
                    else 'malformed_output' if output else 'missing_output')
        reason = self._assertion_failure(output, stage, targets, candidate_ids, evidence, files=files)
        if reason is not None:
            return reason
        plan, reason = self._prepare_handoff(output, stage, targets, evidence, files=files)
        if plan is None:
            return reason
        if output['contradictions']:
            self.assertion_failure_class = 'admission_failure'
            return 'evidence_incomplete'
        self.admitted_stages += 1
        self.handoff = plan.handoff
        self.target_progress = plan.target_progress
        evidence.clipped |= plan.compact_clipped
        self.admitted_receipts.extend(plan.receipts)
        self.admitted_retained_bytes += sum(receipt.retained_bytes for receipt in plan.receipts)
        return None
