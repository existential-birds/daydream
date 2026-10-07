"""Bounded, in-memory progress for one snapshot-bound public reviewer scope."""
from __future__ import annotations

import json
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
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.review_evidence import ReviewEvidence
from daydream.review_result import reason_for_exception
from daydream.trajectory import DaydreamPhase, LifecycleReasonCode, LifecycleStatus, phase_scope

STAGED_REVIEW_CONTRACT = 1
HANDOFF_MAX_BYTES = 64 * 1024
HANDOFF_MAX_ITEMS = 128


class ReviewInvestigation:
    """Admit successful stage assertions; observed spend belongs to the host."""

    def __init__(self, stack: StackAssignment, full_diff: str, revision: dict[str, Any]) -> None:
        self.scope_id = stack.stack_name
        self.revision = revision
        self.diff_weights = file_change_bytes(full_diff)
        self.batches = pack_file_batches(stack, self.diff_weights, 8, 48 * 1024,
                                         [[path] for path in sorted(stack.files)])
        self.handoff: dict[str, list[Any]] = {name: [] for name in ('progress', 'notes', 'candidates', 'evidence')}
        self.budget = ReviewInvestigationBudget.from_limits(ReviewLimits())
        self.reason: str | None = None
        self.admitted_stages = 0
        self.failed_invocation = False
        self.failure_diagnostic: str | None = None

    async def run(
        self, backend: Backend, cwd: Path, prompt: str, **agent_kwargs: Any,
    ) -> tuple[dict[str, Any], str | None]:
        """Fresh calls use one allowance; only admitted progress reaches publication."""
        reserve = (self.budget.limits.tool_calls + 3) // 4
        for index, batch in enumerate(self.batches):
            available = max(0, self.budget.remaining_tool_calls - reserve)
            allowance = min(16, available // (len(self.batches) - index))
            if not await self._stage(backend, cwd, prompt, 'first_pass', batch.files, allowance, agent_kwargs):
                break
        if self.reason is None and self.scope_id == STRUCTURE_STACK_NAME:
            await self._stage(backend, cwd, prompt, 'integration', [f'integration:{self.scope_id}'],
                              min(16, self.budget.remaining_tool_calls), agent_kwargs)
        open_ids = [candidate['candidate_id'] for candidate in self.handoff['candidates']
                    if candidate['disposition'] == 'open']
        for index in range(0, len(open_ids), 8):
            if self.reason is not None:
                break
            await self._stage(backend, cwd, prompt, 'triage', [], self.budget.remaining_tool_calls,
                              agent_kwargs, candidate_ids=open_ids[index:index + 8])
        if self.reason is None and (any(target['status'] != 'reviewed' for target in self.handoff['progress'])
                                    or any(candidate['disposition'] in {'open', 'unresolved'}
                                           for candidate in self.handoff['candidates'])):
            self.reason = 'evidence_incomplete'
        return {'issues': [candidate['finding'] for candidate in self.handoff['candidates']
                           if candidate['disposition'] == 'confirmed']}, self.reason

    async def _stage(self, backend: Backend, cwd: Path, prompt: str, stage: str, targets: list[str],
                     allowance: int, agent_kwargs: dict[str, Any], *, candidate_ids: list[str] | None = None) -> bool:
        if allowance <= 0:
            self.reason = 'tool_call_budget_exceeded'
            return False
        state = {
            'contract_version': STAGED_REVIEW_CONTRACT, 'scope_id': self.scope_id,
            'analyzed_revision': self.revision, 'stage': stage, 'assigned_target_ids': targets,
            **self.handoff, 'assigned_candidate_ids': candidate_ids or [],
            'target_diff_bytes': {target: self.diff_weights.get(target, 1) for target in targets},
            'observed_tool_starts': self.budget.observed_tool_starts,
            'remaining_tool_calls': self.budget.remaining_tool_calls, 'tool_call_allowance': allowance,
        }
        stage_prompt = prompt + '\n\nHost review stage:\n' + json.dumps(state, ensure_ascii=False)
        evidence = ReviewEvidence(REVIEW_STAGE_SCHEMA)
        async with phase_scope(DaydreamPhase.DEEP, stage=f'review-{stage.replace('_', '-')}') as transition:
            prior_admitted = self.admitted_stages
            try:
                output, _, self.reason = await agent.run_agent(
                    backend, cwd, stage_prompt, phase=DaydreamPhase.DEEP, output_schema=REVIEW_STAGE_SCHEMA,
                    require_full_schema=True, investigation_budget=self.budget, review_evidence=evidence,
                    tool_call_budget=allowance, **agent_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 -- retain successful-stage progress after ordinary failures
                self.reason = reason_for_exception(exc).value
                self.failed_invocation = True
                self.failure_diagnostic = f"{type(exc).__name__}: {exception_text(exc) or '(unavailable)'}"
                accepted = False
            else:
                if self.reason is None:
                    self.reason = self._admit(output, stage, targets, candidate_ids or [], evidence)
                accepted = self.reason is None
            finally:
                transition.extra.update(
                    review_scope_id=self.scope_id, observed_tool_starts=self.budget.observed_tool_starts,
                    remaining_tool_calls=self.budget.remaining_tool_calls,
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
               evidence: ReviewEvidence) -> str | None:
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
            'notes': self.handoff['notes'] + [output['notes']],
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
