"""Bounded, in-memory progress for one snapshot-bound public reviewer scope."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from daydream import agent
from daydream.backends import Backend
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.deep.sharding import file_change_bytes, pack_file_batches
from daydream.diagnostics import exception_text
from daydream.json_utils import validates_schema
from daydream.phases.schemas import REVIEW_STAGE_SCHEMA
from daydream.prompt_budget import PreparedSanctionedInputs
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.review_evidence import ReviewEvidence
from daydream.review_result import reason_for_exception
from daydream.trajectory import DaydreamPhase, LifecycleReasonCode, LifecycleStatus, phase_scope

STAGED_REVIEW_CONTRACT = 2
HANDOFF_MAX_BYTES = 64 * 1024
HANDOFF_MAX_ITEMS = 128


class ReviewInvestigation:
    """Admit successful stage assertions; observed spend belongs to the host."""

    def __init__(self, stack: StackAssignment, full_diff: str, revision: dict[str, Any]) -> None:
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
        self.handoff: dict[str, list[Any]] = {name: [] for name in ('progress', 'notes', 'candidates', 'evidence')}
        self.budget = ReviewInvestigationBudget.from_limits(ReviewLimits())
        self.reason: str | None = None
        self.admitted_stages = 0
        self.failed_invocation = False
        self.failure_diagnostic: str | None = None

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
            for batch in self.batches:
                if not await self._stage(backend, cwd, prompt_builder, 'first_pass', batch.files, batch.files,
                                         agent_kwargs, stage_inputs=stage_inputs):
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
        return {'issues': [candidate['finding'] for candidate in self.handoff['candidates']
                           if candidate['disposition'] == 'confirmed']}, self.reason

    def _stage_state(self, stage: str, targets: list[str], files: list[str],
                     candidate_ids: list[str]) -> dict[str, Any]:
        candidates = [candidate for candidate in self.handoff['candidates']
                      if candidate['candidate_id'] in candidate_ids]
        candidate_files = {candidate['file'] for candidate in candidates}
        # Only admitted context relevant to this triage assignment travels into
        # its fresh request. Closed decisions expose handles, never fresh work.
        notes = [note['text'] for note in self.handoff['notes']
                 if candidate_files.intersection(note['files'])]
        evidence = [block for block in self.handoff['evidence']
                    if any(path in block['excerpt'] for path in candidate_files)]
        target = (max(4, 3 * len(candidate_ids)) if stage == 'triage' else
                  max(8, len(files)) if stage == 'integration' else max(4, 2 * len(files)))
        return {
            'contract_version': STAGED_REVIEW_CONTRACT, 'scope_id': self.scope_id,
            'analyzed_revision': self.revision, 'stage': stage, 'assigned_target_ids': targets,
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
    ) -> bool:
        if self.budget.remaining_tool_calls <= 0:
            self.reason = 'tool_call_budget_exceeded'
            return False
        state = self._stage_state(stage, targets, files, candidate_ids or [])
        evidence = ReviewEvidence(REVIEW_STAGE_SCHEMA)
        async with phase_scope(DaydreamPhase.DEEP, stage=f'review-{stage.replace('_', '-')}') as transition:
            prior_admitted = self.admitted_stages
            try:
                stage_prompt = prompt_builder(state)
                kwargs = dict(agent_kwargs)
                if stage_inputs is not None:
                    kwargs['sanctioned_inputs'] = stage_inputs(state)
                output, _, self.reason = await agent.run_agent(
                    backend, cwd, stage_prompt, phase=DaydreamPhase.DEEP, output_schema=REVIEW_STAGE_SCHEMA,
                    require_full_schema=True, investigation_budget=self.budget, review_evidence=evidence,
                    advisory_tool_call_target=state['advisory_tool_call_target'], **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 -- retain successful-stage progress after ordinary failures
                self.reason = reason_for_exception(exc).value
                self.failed_invocation = True
                self.failure_diagnostic = f"{type(exc).__name__}: {exception_text(exc) or '(unavailable)'}"
                accepted = False
            else:
                if self.reason is None:
                    self.reason = self._admit(output, stage, targets, candidate_ids or [], evidence,
                                              files=files or state['assigned_files'])
                accepted = self.reason is None
            finally:
                transition.extra.update(
                    review_scope_id=self.scope_id, observed_tool_starts=self.budget.observed_tool_starts,
                    remaining_tool_calls=self.budget.remaining_tool_calls,
                    hard_tool_call_allowance=state['remaining_tool_calls'],
                    advisory_tool_call_target=state['advisory_tool_call_target'],
                    assigned_target_ids=targets, assigned_candidate_ids=candidate_ids or [],
                    admitted=self.admitted_stages > prior_admitted, stop_reason=self.reason,
                )
            if self.reason in {'wall_budget_exceeded', 'pipeline_budget_exceeded'}:
                transition.finish(LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT)
            elif not accepted:
                transition.finish(LifecycleStatus.FAILED, LifecycleReasonCode.DOMAIN_FAILURE)
            else:
                transition.finish(LifecycleStatus.SUCCEEDED)
            return accepted

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
        if decisions and (evidence.clipped or evidence.omitted or any(
            not candidate[field].strip() for candidate in decisions for field in ('grounds', 'trigger', 'consequence')
        )):
            return 'evidence_incomplete'
        if stage == 'triage':
            by_id = {candidate['candidate_id']: candidate for candidate in decisions}
            candidates = [by_id.get(candidate['candidate_id'], candidate) for candidate in prior_candidates]
        else:
            candidates = prior_candidates + [
                {**candidate, 'candidate_id': f'{self.scope_id}:candidate:{len(prior_candidates) + index + 1}'}
                for index, candidate in enumerate(decisions)
            ]
        # Completed excerpts support grounds, without authenticating individual citations.
        handoff = {
            'progress': self.handoff['progress'] + output['targets'],
            'notes': self.handoff['notes'] + [{
                'files': sorted(set(files) | {candidate['file'] for candidate in decisions}), 'text': output['notes'],
            }],
            'candidates': candidates, 'evidence': self.handoff['evidence'] + [
                {'stage': stage, 'assigned_target_ids': targets, 'excerpt': block}
                for block in evidence.blocks if any(candidate['file'] in block for candidate in decisions)
            ],
        }
        if (sum(map(len, handoff.values())) > HANDOFF_MAX_ITEMS
                or len(json.dumps(handoff, ensure_ascii=False).encode('utf-8')) > HANDOFF_MAX_BYTES):
            return 'evidence_incomplete'
        self.admitted_stages += 1
        self.handoff = handoff
        return 'evidence_incomplete' if output['contradictions'] else None
