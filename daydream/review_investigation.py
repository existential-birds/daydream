"""Bounded, in-memory progress for one snapshot-bound public reviewer scope."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from daydream import agent
from daydream.backends import Backend
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.deep.sharding import file_change_bytes, pack_file_batches
from daydream.json_utils import SchemaRejection, validates_schema
from daydream.phases.schemas import REVIEW_STAGE_SCHEMA
from daydream.prompt_budget import PreparedSanctionedInputs, SanctionedInputUnavailable
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.review_evidence import FULL_REVIEW_MAX_BYTES, EvidenceReceipt, ReviewEvidence
from daydream.review_result import reason_for_exception
from daydream.trajectory import DaydreamPhase, LifecycleReasonCode, LifecycleStatus, phase_scope

STAGED_REVIEW_CONTRACT = 3
HANDOFF_MAX_BYTES = 64 * 1024
HANDOFF_MAX_ITEMS = 128


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
        candidate_files = {candidate['file'] for candidate in candidates}
        # Only admitted context relevant to this triage assignment travels into
        # its fresh request. Closed decisions expose handles, never fresh work.
        notes = [note['text'] for note in self.handoff['notes']
                 if candidate_files.intersection(note['files'])]
        evidence = [block for block in self.handoff['evidence']
                    if candidate_files.intersection(block['files'])]
        target = (max(4, 3 * len(candidate_ids)) if stage == 'triage' else
                  max(8, len(files)) if stage == 'integration' else max(4, 2 * len(files)))
        return {
            'contract_version': STAGED_REVIEW_CONTRACT, 'scope_id': self.scope_id,
            'analyzed_revision': self.revision, 'stage': stage, 'assigned_target_ids': targets,
            'assignment_parts': assignment_parts,
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
        state.update(attempt=attempt, max_attempts=2,
                     schema_rejection=rejection.to_dict() if rejection else None)
        evidence = ReviewEvidence(REVIEW_STAGE_SCHEMA)
        retry_rejection: SchemaRejection | None = None
        actual_rejection: SchemaRejection | None = None
        accepted = False
        starts_before = self.budget.observed_tool_starts
        candidate_count = 0
        identity_failure = False
        async with phase_scope(DaydreamPhase.DEEP, stage=f'review-{stage.replace('_', '-')}') as transition:
            prior_admitted = self.admitted_stages
            try:
                kwargs = dict(agent_kwargs)
                if stage_inputs is not None:
                    kwargs['sanctioned_inputs'] = stage_inputs(state)
                prepared = kwargs.get('sanctioned_inputs')
                if prepared is not None:
                    prepared.revalidate(backend, cwd, kwargs.get('read_only', False))
                evidence.configure_capture(cwd, allowance=FULL_REVIEW_MAX_BYTES - self.admitted_retained_bytes,
                                           sanctioned_inputs=prepared,
                                           snapshot_revisions=tuple(self.revision[key] for key in
                                                                    ('head_sha', 'merge_base_sha')))
                stage_prompt = prompt_builder(state)
                output, _, self.reason = await agent.run_agent(
                    backend, cwd, stage_prompt, phase=DaydreamPhase.DEEP, output_schema=REVIEW_STAGE_SCHEMA,
                    require_full_schema=True, investigation_budget=self.budget, review_evidence=evidence,
                    advisory_tool_call_target=state['advisory_tool_call_target'], **kwargs,
                )
                if prepared is not None:
                    prepared.revalidate(backend, cwd, kwargs.get('read_only', False))
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
                candidate_count = len(output.get('candidates', [])) if isinstance(output, dict) else 0
                if self.reason is None:
                    if (actual_rejection is not None
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
                elif actual_rejection is not None:
                    self.failure_class = 'schema_rejection'
                elif self.reason == 'evidence_incomplete':
                    self.failure_class = ('capture_loss' if identity_failure or evidence.capture_failure(
                        files or state['assigned_files'], require_source=False,
                        interaction=stage == 'integration') else 'admission_failure')
                elif self.reason == 'interruption':
                    self.failure_class = 'cancellation'
                else:
                    self.failure_class = self.reason
                if self.reason is not None and not self.failed_invocation:
                    labels = {'schema_rejection': 'Strict stage schema rejected',
                              'capture_loss': 'Required source evidence capture incomplete',
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
                    retry_feedback=rejection.to_dict() if rejection else None,
                    full_retained_bytes=evidence.full_retained_bytes,
                    compact_view_clipped=evidence.clipped,
                    native_truncated_results=evidence.native_truncated_results,
                    retention_overflow_results=evidence.retention_overflow_results,
                    unmatched_results=evidence.unmatched_results,
                    failure_class=self.failure_class,
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

    def _admit(self, output: Any, stage: str, targets: list[str], candidate_ids: list[str],
               evidence: ReviewEvidence, *, files: list[str]) -> str | None:
        """Atomically validate and admit successful invocation assertions within the handoff bounds."""
        if not isinstance(output, dict) or not validates_schema(output, REVIEW_STAGE_SCHEMA):
            return (output.reason if isinstance(output, agent.StructuredOutputFailure)
                    else 'malformed_output' if output else 'missing_output')
        decisions = output['candidates']
        prior_candidates = self.handoff['candidates']
        contradictions = output['contradictions']
        closed_ids = {candidate['candidate_id'] for candidate in prior_candidates
                      if candidate['disposition'] in {'confirmed', 'rejected'}}
        decision_ids = [candidate['candidate_id'] for candidate in decisions]
        if (
            sorted(target['target_id'] for target in output['targets']) != sorted(targets)
            or any(target['status'] == 'not_reviewed' and not target['reason'].strip()
                   for target in output['targets'])
            or len(set(contradictions)) != len(contradictions)
            or not set(contradictions) <= closed_ids
            or (stage == 'triage' and (sorted(decision_ids) != sorted(candidate_ids)
                                      or any(candidate['disposition'] == 'open' for candidate in decisions)))
            or (stage != 'triage' and any(decision_ids))
            or any(candidate['disposition'] == 'confirmed' and candidate['finding'] is None
                   for candidate in decisions)
        ):
            return 'malformed_output'
        if decisions and any(
            not candidate[field].strip() for candidate in decisions for field in ('grounds', 'trigger', 'consequence')
        ):
            return 'evidence_incomplete'
        reviewed_ids = {target['target_id'] for target in output['targets'] if target['status'] == 'reviewed'}
        reviewed_files = (files if stage == 'integration' and reviewed_ids else
                          [path for path in files if any(target in reviewed_ids
                                                        for target in self.required_targets[path])])
        grounding_files = sorted(set(reviewed_files) | {candidate['file'] for candidate in decisions})
        if grounding_files:
            if evidence.capture_failure(grounding_files, require_source=stage != 'triage',
                                        interaction=stage == 'integration'):
                return 'evidence_incomplete'
            if stage == 'triage':
                grounded = {path for receipt in [*self.admitted_receipts, *evidence.receipts]
                            if receipt.complete and not receipt.supporting for path in receipt.paths}
                if not set(grounding_files) <= grounded:
                    return 'evidence_incomplete'
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
        relevant_files = {candidate['file'] for candidate in decisions}
        relevant_blocks = evidence.compact_for_files(relevant_files)
        for index, block in enumerate(relevant_blocks):
            entry = {'stage': stage, 'assigned_target_ids': targets, **block}
            size = len(json.dumps(entry, ensure_ascii=False).encode())
            if compact_bytes + size > 48000 or len(compact) >= 64:
                evidence.clipped = True
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
                'files': sorted(set(files) | {candidate['file'] for candidate in decisions}), 'text': output['notes'],
            }],
            'candidates': candidates, 'evidence': compact,
        }
        if (sum(map(len, handoff.values())) > HANDOFF_MAX_ITEMS
                or len(json.dumps(handoff, ensure_ascii=False).encode('utf-8')) > HANDOFF_MAX_BYTES):
            return 'evidence_incomplete'
        self.admitted_stages += 1
        self.handoff = handoff
        self.target_progress = target_progress
        receipts = [receipt for receipt in evidence.receipts
                    if receipt.complete and relevant_files.intersection(receipt.paths)]
        self.admitted_receipts.extend(receipts)
        self.admitted_retained_bytes += sum(receipt.retained_bytes for receipt in receipts)
        return 'evidence_incomplete' if output['contradictions'] else None
