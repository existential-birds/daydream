"""Small production committed-Pi review program, independent of flow shared state.

Review owns coupled claims, revisions, required challenge and coverage. Native
execution, captured Git source, artifact ownership and recording retain their
existing lifetimes. Route/default cutover is deliberately owned by later work.
"""

from __future__ import annotations

import math
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anyio
import jsonschema

from daydream import agent, clock, git_ops
from daydream.artifact_visibility import (
    ArtifactDisposition,
    OutputLabel,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.artifacts.ownership import operational_worktree_root
from daydream.backends import BackendExecutionInput
from daydream.backends.pi import PiBackend
from daydream.committed_review_model import (
    COMMITTED_RESULT_SCHEMA,
    CommittedReviewResult,
    challenge_schema,
    discovery_schema,
    planned_work_id,
    semantic_revision,
    validate_committed_result,
)
from daydream.committed_review_output import write_committed_review_output
from daydream.deep.detection import StackAssignment
from daydream.host_tools import _MAX_RESULT_BYTES, HostToolBinding, HostToolFatal, HostToolRejected
from daydream.hunk_index import write_hunk_index
from daydream.json_utils import canonical_json
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.review_profile import resolve_profile
from daydream.review_result import AnalyzedRevision, PlannedScope, ReasonCode, ReviewCoverage, reason_for_exception
from daydream.review_source_refs import source_options
from daydream.review_stage_inputs import StageInputFactory
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext, bind_run_context
from daydream.runner import _resolve_backend
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow
from daydream.workspace import WorkContext, open_workspace


@dataclass(frozen=True)
class CommittedReviewRequest:
    """Explicit committed subject and policy; no edit or external posting authority."""

    source: Path
    base: str
    config: RunConfig = field(default_factory=lambda: RunConfig(backend="pi", output_mode="review"))
    limits: ReviewLimits = field(default_factory=ReviewLimits)


class Review:
    """Only writer of live assignments, semantic revisions and challenge evidence."""

    def __init__(self, inputs: StageInputFactory, run_id: str, *, deadline: float) -> None:
        self.inputs, self.run_id, self.deadline = inputs, run_id, deadline
        self.assignments = deepcopy([part for batch in inputs.assignment_batches for part in batch])
        self._targets: list[dict[str, Any]] = []
        self._claims: list[dict[str, Any]] = []
        self._sealed = False
        self._discovered = False
        self._binding: HostToolBinding | None = None
        self.execution: list[dict[str, Any]] = []
        self._coverage = ReviewCoverage(
            run_id,
            AnalyzedRevision.from_dict(inputs.revision),
            [PlannedScope("committed", "committed", tuple(sorted({p["file"] for p in self.assignments})))]
            if self.assignments
            else [],
            ["challenge"] if self.assignments else ["no_diff"],
        )

    def _refs(self, selectors: Any, *, target: str | None = None, required: bool = True) -> list[dict[str, Any]]:
        refs = [
            ref
            for part in self.assignments
            if target is None or part["target_id"] == target
            for ref in part["source_refs"]
        ]
        return self.inputs.canonicalize_source_selectors(selectors, refs, require_assigned=required)

    def _claim(self, submitted: dict[str, Any], claims: list[dict[str, Any]]) -> dict[str, Any]:
        claim = deepcopy(submitted)
        claim["payload"]["evidence_refs"] = self._refs(claim["payload"]["evidence_refs"])
        payload = claim["payload"]
        if payload["file"] not in {p["file"] for p in self.assignments}:
            raise HostToolRejected("claim.payload.file: choose a file from the complete changed-target inventory")
        if payload["line"] is not None and not any(
            ref["path"] == payload["file"]
            and ref["source_range"] is not None
            and ref["source_range"]["start"] <= payload["line"] <= ref["source_range"]["end"]
            for ref in payload["evidence_refs"]
        ):
            raise HostToolRejected("claim.payload.line: choose a line covered by its canonical source references")
        known = {c["uid"]: c for c in self._claims}
        if not set(claim["source_uids"]) <= known.keys():
            raise HostToolRejected("claim.source_uids: use retained predecessor UIDs, or [] for no predecessors")
        if not claim["uid"]:
            if claim["revision"] != 0:
                raise HostToolRejected('new claims require uid="" and revision=0')
            ordinal = len(claims) + 1
            claim.update(
                uid=f"{self.run_id}:claim:{ordinal}",
                item_uid=f"{self.run_id}:item:{ordinal}",
                id=ordinal,
                revision=1,
                disposition="unresolved",
                challenge_evidence=[],
            )
            claims.append(claim)
            return claim
        existing = next((c for c in claims if c["uid"] == claim["uid"]), None)
        if existing is None or existing["revision"] != claim["revision"]:
            raise HostToolRejected("claim.uid/revision: revise a currently assigned retained claim revision")
        if claim["source_uids"] != existing["source_uids"]:
            raise HostToolRejected("claim.source_uids: an existing claim retains its predecessor identity")
        if semantic_revision(payload) != semantic_revision(existing["payload"]):
            existing.update(revision=existing["revision"] + 1, payload=payload, disposition="unresolved")
        existing["title"] = claim["title"]
        return existing

    def assignment(self, kind: str, *, deadline: float) -> tuple[HostToolBinding, dict[str, Any]]:
        """Open one native terminal capability for the current immutable work snapshot."""
        if self._sealed or self._binding is not None and self._binding.authority_open:
            raise ValueError("Review is sealed or already has an active assignment")
        if kind not in {"discovery", "challenge"} or kind == "discovery" and self._discovered:
            raise ValueError("invalid or already admitted assignment")
        revisions = {c["uid"]: c["revision"] for c in self._claims if c["disposition"] == "unresolved"}
        if kind == "challenge" and not revisions:
            raise ValueError("no challenge obligations")
        schema = discovery_schema(selectors=True) if kind == "discovery" else challenge_schema(selectors=True)

        def accept(value: dict[str, Any]) -> dict[str, Any]:
            try:
                jsonschema.validate(value, schema)
            except jsonschema.ValidationError as exc:
                raise HostToolRejected(f"{exc.json_path}: correct the submission to the supplied schema") from exc
            targets = deepcopy(self._targets)
            claims = deepcopy(self._claims)
            canonical = deepcopy(value)
            if kind == "discovery":
                identifiers = [t["target_id"] for t in value["targets"]]
                if len(set(identifiers)) != len(identifiers) or set(identifiers) != {
                    p["target_id"] for p in self.assignments
                }:
                    raise HostToolRejected("targets: submit each assigned target_id exactly once")
                targets = canonical["targets"]
                for target in targets:
                    if target["status"] == "not_reviewed" and not target["reason"].strip():
                        raise HostToolRejected("targets.reason: explain unfinished work")
                    target["evidence_refs"] = self._refs(
                        target["evidence_refs"], target=target["target_id"], required=target["status"] == "reviewed"
                    )
                if any(c["uid"] for c in value["claims"]):
                    raise HostToolRejected("discovery.claims: new concerns require empty uid and revision=0")
            else:
                seen: set[str] = set()
                for verdict in canonical["verdicts"]:
                    uid = verdict["uid"]
                    if uid in seen or revisions.get(uid) != verdict["revision"]:
                        raise HostToolRejected("verdicts: use each assigned uid/current revision at most once")
                    seen.add(uid)
                    if not verdict["grounds"].strip():
                        raise HostToolRejected("verdicts.grounds: retain substantive counterevidence or uncertainty")
                    verdict["evidence_refs"] = self._refs(verdict["evidence_refs"])
                    claim = next(c for c in claims if c["uid"] == uid)
                    claim["challenge_evidence"].append(deepcopy(verdict))
                    claim["disposition"] = verdict["verdict"]
                if any(c["uid"] and c["uid"] not in revisions for c in value["claims"]):
                    raise HostToolRejected("claims: revisions may change only currently assigned claims")
            changed: set[str] = set()
            for submitted in value["claims"]:
                if submitted["uid"] and submitted["uid"] in changed:
                    raise HostToolRejected("claims: update a retained uid at most once per submission")
                changed.add(submitted["uid"])
                self._claim(submitted, claims)
            # Return canonical selector-free submissions, without exposing mutable live state.
            canonical["claims"] = [deepcopy(c) for c in claims]
            if len(claims) > 128:
                raise HostToolRejected("claims: shared retention capacity is 128 claims; reduce new proposals")
            result_schema = self._result_schema(kind)
            try:
                jsonschema.validate(canonical, result_schema)
            except jsonschema.ValidationError as exc:
                raise HostToolRejected("claims: canonical retention limit exceeded; reduce optional proposals") from exc
            if (
                max(
                    len(canonical_json(value).encode()) for value in (canonical, {"targets": targets, "claims": claims})
                )
                > _MAX_RESULT_BYTES
            ):
                raise HostToolRejected("retained review exceeds shared byte capacity; reduce optional material")
            try:
                self._coverage.check_target_decisions_budget("committed", targets)
            except ValueError as exc:
                raise HostToolRejected(
                    "targets: reduce retained reasons/attribution to the shared coverage capacity"
                ) from exc
            # No await, I/O, or model decision between final checks and installation.
            binding.ensure_open()
            if self._sealed or clock.monotonic() >= self.deadline:
                raise HostToolFatal("Review authority has closed", code="authority_closed")
            if (
                kind == "discovery"
                and self._discovered
                or kind == "challenge"
                and any(
                    next((c["revision"] for c in self._claims if c["uid"] == uid), None) != revision
                    for uid, revision in revisions.items()
                )
            ):
                raise HostToolFatal("Assignment revisions changed before installation", code="authority_closed")
            self._targets, self._claims = targets, claims
            self._discovered |= kind == "discovery"
            return canonical

        binding = HostToolBinding(deadline=min(deadline, self.deadline), accept=accept)
        self._binding = binding
        return binding, schema

    @staticmethod
    def _result_schema(kind: str) -> dict[str, Any]:
        # Retained claim records add host identity/challenge evidence to the canonical payload.
        schema = discovery_schema(selectors=False) if kind == "discovery" else challenge_schema(selectors=False)
        schema["properties"]["claims"] = deepcopy(COMMITTED_RESULT_SCHEMA["properties"]["claims"])
        return schema

    def pending(self) -> list[dict[str, Any]]:
        """An immutable projection; a model cannot clear required work by declaring completion."""
        return deepcopy([c for c in self._claims if c["disposition"] == "unresolved"])

    def seal(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Called after native contexts/host callbacks have quiesced."""
        if self._sealed:
            raise ValueError("Review is already sealed")
        if self._binding is not None:
            self._binding.close_authority()
            # Closed synchronous callbacks cannot install after sealing. Failed
            # native/transport cleanup remains an execution failure, not lost work.
        self._sealed = True
        coverage = self._coverage
        reasons = {e["reason"] for e in self.execution if e["reason"] is not None}
        completed = (
            self._discovered
            and len(self._targets) == len(self.assignments)
            and all(t["status"] == "reviewed" for t in self._targets)
        )
        if self.assignments:
            scope_complete = completed and not reasons
            coverage.record_scope(
                "committed",
                "complete" if scope_complete else "incomplete",
                reasons=[] if scope_complete else sorted(reasons | {ReasonCode.EVIDENCE_INCOMPLETE.value}),
                partial_evidence=bool(self._targets) and not scope_complete,
                target_decisions=self._targets,
            )
            challenged = not self.pending() and completed and not reasons
            coverage.record_phase(
                "challenge",
                "complete" if challenged else "incomplete",
                reasons=[] if challenged else sorted(reasons | {ReasonCode.EVIDENCE_INCOMPLETE.value}),
                usable_evidence=bool(self._claims),
            )
        else:
            coverage.record_phase("no_diff", "complete", noop=True)
        pipeline = "failed" if reasons else "completed"
        terminal = coverage.finalize(pipeline)
        result = {
            "schema_version": 1,
            "run_id": self.run_id,
            "analyzed_revision": self.inputs.revision,
            "assignments": self.assignments,
            "planned_work_id": planned_work_id(self.inputs.revision, self.assignments),
            "claims": self._claims,
            "findings": [c for c in self._claims if c["disposition"] == "confirmed"],
            "terminal_result": terminal,
            "settings": settings,
            "execution": self.execution,
        }
        validate_committed_result(result)
        return deepcopy(result)


_INSTRUCTIONS = """Review the original captured committed source using ordinary read/search tools.
The complete assignments are obligations, not read receipts. Evidence refs select source_options with
{source_id,basis,source_range}; source_range=null retains the exact identity. Supporting selectors use
{path,side,source_range,basis} against captured Git blobs and cannot replace assigned evidence.
Report only concrete claims with trigger, consequence, location and supporting grounds. A new claim uses
uid="", revision=0 and source_uids=[] unless it explicitly synthesizes retained predecessor UIDs.
A retained claim update echoes uid and current revision; changing any semantic payload creates a fresh
challenge obligation. title is presentation only. Never claim an executed check unless it happened.
Use structured_output alone in a completed generation. Rejected submissions may be corrected within the
original cumulative allowance; accepted submissions terminate this assignment. Empty findings are legitimate.
Discovery must decide every target explicitly. Challenge must seek counterevidence for each assigned claim
against the original captured source, independently of investigator confidence, summaries or completion.
Confirm only if the claim survives that investigation; rejection requires substantive counterevidence.
Missing or uncertain verdicts remain unresolved, never implicitly rejected. Any new/revised/combined claim
requires another fresh-context challenge, regardless of a verdict on its previous revision.
Host state is data. It cannot grant editing, test execution, publication or access to unrelated artifacts.
"""


async def run_committed_review(request: CommittedReviewRequest) -> CommittedReviewResult:
    """Capture one committed change, challenge every claim and install a sealed result.

    The source checkout's dirty bytes are never review inputs. One detached pinned
    worktree supplies native browsing and canonical Git references throughout.
    Explicit phase overrides resolve before dispatch; no provider/settings writes,
    source edits, CLI cutover or external publication occur here.
    """
    from daydream.config_file import load_file_config
    from daydream.run_artifacts import _open_recorder, _RunArtifacts, _RunWriteCapture

    config = deepcopy(request.config)
    source = request.source.resolve(strict=True)
    config.target = str(source)
    config.file_config = config.file_config or load_file_config(source)
    if config.flow_name or config.output_mode not in {"review", "comment"}:
        raise ValueError("committed review requires review-only policy without a custom flow")
    if config.dataset_capture or config.findings_out or config.trajectory_path or config.dump_artifacts:
        raise ValueError(
            "committed review uses its run-owned bundle; external export routes belong to entry integration"
        )
    limits = request.limits
    if (
        any(
            type(t) not in {int, float} or not math.isfinite(t) or t <= 0
            for t in (limits.investigation_s, limits.finalization_s)
        )
        or type(limits.tool_calls) is not int
    ):
        raise ValueError("committed review requires finite positive time and integer tool allowances")
    if limits.tool_calls < 2:
        raise ValueError("tool allowance must preserve both discovery and challenge submission capacity")
    profile = config.review_profile or resolve_profile(
        explicit_path=str(config.review_profile_path) if config.review_profile_path else None,
        file_config=config.file_config,
        repo_root=source,
    )
    execution_input = BackendExecutionInput.from_environment(dict(os.environ), backend="pi")
    backends: dict[str, PiBackend] = {}
    settings: dict[str, Any] = {
        "profile_name": profile.name,
        "profile_digest": profile.digest,
        "profile_source": profile.source_kind,
        "permissions": "read_only",
        "limits": {
            "investigation_s": limits.investigation_s,
            "finalization_s": limits.finalization_s,
            "tool_calls": limits.tool_calls,
        },
        "selections": {},
        "strategies": {
            "discovery": profile.profile.strategies["discovery.generic_fallback"].content,
            "challenge": profile.profile.strategies["arbitration"].content.replace(
                "{arbiter_input_path}", "the captured-source claim inventory below"
            ),
        },
    }
    for kind, phase in [("discovery", "review"), ("challenge", "arbiter")]:
        backend = _resolve_backend(config, phase, cwd=source, execution_input=execution_input)
        if not isinstance(backend, PiBackend):
            raise ValueError("committed review currently requires native Pi for discovery and challenge")
        selection = backend.freeze_selection(source)
        backends[kind] = backend
        settings["selections"][kind] = {"backend": "pi", **selection, "phase": phase}
    owner = resolve_private_workspace_owner(source, locations=private_root_locations())
    context = RunContext(InteractionPolicy(interactive=False, quiet=config.quiet))
    async with open_workspace(
        source, branch=None, base=request.base, force_ephemeral=False, skip_tests=True, private_owner=owner
    ) as original:
        run_id = str(uuid.uuid4())
        captured_root = operational_worktree_root(owner) / run_id
        assert original.head_sha is not None and original.base_sha is not None
        git_ops.worktree_add(source, captured_root, original.head_sha, detach=True)
        work = replace(original, repo=captured_root, is_ephemeral=True, run_id=run_id)
        try:
            with bind_run_context(context):
                async with open_artifact_session(work, session_id=run_id, owner=owner) as session:
                    session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
                    session.register_destination(source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT)
                    trajectory = session.register_trajectory_output(None)
                    capture = _RunWriteCapture(session_id=run_id)
                    artifacts = _RunArtifacts(session, owner, trajectory, capture, None)
                    diff = git_ops.diff(captured_root, original.base_sha, original.head_sha)
                    import hashlib

                    revision = AnalyzedRevision(
                        original.head_sha, original.base_sha, hashlib.sha256(diff.encode()).hexdigest()
                    ).to_dict()
                    diff_path = session.write_review_input(work.repo, Path("committed/diff.patch"), diff)
                    index_path = write_hunk_index(diff_path.parent, diff)
                    files = git_ops.diff_name_only_strict(captured_root, original.base_sha, original.head_sha)
                    inputs = StageInputFactory(
                        backends["discovery"],
                        work,
                        StackAssignment("committed", files),
                        diff_path=diff_path,
                        hunk_index_path=index_path,
                        shared_paths={},
                        revision=revision,
                        artifact_session=session,
                        allow_standalone=False,
                        read_only=True,
                    )
                    started = clock.monotonic()
                    investigation_s = min(
                        limits.investigation_s, profile.profile.pipeline.reviewer_investigation_budget_s
                    )
                    finish = started + min(
                        investigation_s + limits.finalization_s, profile.profile.pipeline.review_wall_budget_s
                    )
                    if finish - started <= limits.finalization_s:
                        raise ValueError("profile wall allowance cannot preserve finalization capacity")
                    review_deadline = finish - limits.finalization_s
                    discovery_deadline = started + (review_deadline - started) / 2
                    settings["limits"].update(
                        effective_investigation_s=review_deadline - started,
                        protected_challenge_s=review_deadline - discovery_deadline,
                        protected_finalization_s=limits.finalization_s,
                    )
                    budget = ReviewInvestigationBudget(limits, review_deadline)
                    review = Review(inputs, run_id, deadline=review_deadline)
                    recorder = _open_recorder(
                        config=config,
                        target_dir=captured_root,
                        work=work,
                        flow_kind=DaydreamRunFlow.CUSTOM,
                        run_artifacts=artifacts,
                    )
                    recorder.record_profile(
                        schema_version=profile.profile.schema_version,
                        name=profile.name,
                        source_kind=profile.source_kind,
                        digest=profile.digest,
                    )
                    try:
                        async with recorder:
                            if files:
                                await _dispatch(
                                    review,
                                    "discovery",
                                    backends["discovery"],
                                    work,
                                    budget,
                                    deadline=discovery_deadline,
                                    allocation=limits.tool_calls - math.ceil(limits.tool_calls / 2),
                                    diff_path=diff_path,
                                    profile=settings,
                                )
                                attempted: set[tuple[str, int]] = set()
                                while review.pending():
                                    current = {(c["uid"], c["revision"]) for c in review.pending()}
                                    if (
                                        current <= attempted
                                        or budget.remaining_tool_calls < 1
                                        or clock.monotonic() >= review_deadline
                                    ):
                                        break
                                    attempted.update(current)
                                    await _dispatch(
                                        review,
                                        "challenge",
                                        backends["challenge"],
                                        work,
                                        budget,
                                        deadline=review_deadline,
                                        allocation=budget.remaining_tool_calls,
                                        diff_path=diff_path,
                                        profile=settings,
                                    )
                    finally:
                        for backend in backends.values():
                            with anyio.CancelScope(shield=True):
                                try:
                                    await backend.cancel()
                                except Exception:
                                    review.execution.append(
                                        {
                                            "assignment": "cleanup",
                                            "accepted": False,
                                            "settled": False,
                                            "cleanup_complete": False,
                                            "reason": ReasonCode.BACKEND_FAILURE.value,
                                        }
                                    )
                    payload = review.seal(settings)
                    output_error: str | None = None
                    installed = False
                    capture_state = (
                        "complete" if capture.final is not None and capture.validation_error is None else "failed"
                    )
                    try:
                        with anyio.CancelScope(shield=True):
                            if clock.monotonic() >= finish:
                                raise TimeoutError("finalization allowance exhausted before output staging")
                            write_committed_review_output(session, payload)
                            selected = capture.final or capture.partial
                            frozen = session.freeze(selected) if selected is not None else session.freeze_unproduced()
                            if clock.monotonic() >= finish:
                                raise TimeoutError("finalization allowance exhausted before installation")
                            session.finalize_frozen(frozen, disposition=ArtifactDisposition.PARTIAL_EVIDENCE)
                            installed = True
                    except Exception as exc:
                        output_error = type(exc).__name__
                    return CommittedReviewResult(
                        canonical_json(payload), installed, capture_state, output_error=output_error
                    )
        finally:
            with anyio.CancelScope(shield=True):
                git_ops.worktree_remove(source, captured_root, force=True)


async def _dispatch(
    review: Review,
    kind: str,
    backend: PiBackend,
    work: WorkContext,
    budget: ReviewInvestigationBudget,
    *,
    deadline: float,
    allocation: int,
    diff_path: Path,
    profile: dict[str, Any],
) -> None:
    binding, schema = review.assignment(kind, deadline=deadline)
    assignment = {
        "kind": kind,
        "run_id": review.run_id,
        "analyzed_revision": review.inputs.revision,
        "captured_root": str(work.repo),
        "assignments": [{**part, "source_options": source_options(part["source_refs"])} for part in review.assignments],
        "claims": review.pending() if kind == "challenge" else [],
        "profile": profile,
    }
    prompt = (
        _INSTRUCTIONS
        + "\nReview strategy:\n"
        + profile["strategies"][kind]
        + "\nCommitted review assignment:\n"
        + canonical_json(assignment)
        + "\nEnd committed review assignment.\n"
    )
    reason: str | None = None
    settled = False
    budget.deadline = deadline
    try:
        prepared = prepare_sanctioned_inputs(backend, work.repo, {"diff": diff_path}, read_only=True)
        _, _, stopped = await agent.run_agent(
            backend,
            work.repo,
            prompt,
            phase=DaydreamPhase.REVIEW if kind == "discovery" else DaydreamPhase.VERIFY,
            output_schema=schema,
            result_schema=review._result_schema(kind),
            read_only=True,
            persist_session=False,
            deadline=deadline,
            investigation_budget=budget,
            tool_call_budget=allocation,
            tool_call_start_floor=1,
            sanctioned_inputs=prepared,
            review_system_instructions=_INSTRUCTIONS,
            review_stage_scope_id=kind,
            host_tools=binding,
        )
        settled = stopped is None and binding.native_settled
        if stopped is not None:
            from daydream.review_result import reason_for_budget

            try:
                reason = reason_for_budget(stopped).value
            except KeyError:
                reason = ReasonCode.EVIDENCE_INCOMPLETE.value
        elif not binding.accepted:
            reason = ReasonCode.MISSING_OUTPUT.value
    except Exception as exc:
        reason = reason_for_exception(exc).value
    finally:
        binding.close_authority()
        review.execution.append(
            {
                "assignment": kind,
                "accepted": binding.accepted,
                "settled": settled,
                "cleanup_complete": binding.cleanup_complete,
                "reason": reason
                if reason is not None
                else (ReasonCode.BACKEND_FAILURE.value if not settled or not binding.cleanup_complete else None),
            }
        )
