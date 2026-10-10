"""Review and adjudication prompt builders for deep mode.

Profile strategies own judgment policy; these builders add host-owned scope,
context transport, grounding, and output instructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream import review_profile
from daydream.deep.detection import base_stack_name
from daydream.deep.diff import _full_diff_pointer, _hunk_index_authority
from daydream.phases.review_prompts import (
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
    _settled_decisions_block,
)
from daydream.phases.schemas import REVIEW_STAGE_SCHEMA
from daydream.prompt_budget import inline_context_file
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_BLOCK
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION, UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.prompts.wire_contract import (
    WIRE_CONTRACT_GENERIC_INSTRUCTION,
    WIRE_CONTRACT_RUST_INSTRUCTION,
)
from daydream.review_budget import STAGED_REVIEW_GUIDANCE
from daydream.severity import SEVERITY_RUBRIC

DOC_REVIEW_NOTICE = (
    "[Notice] Dedicated documentation review is planned but not yet "
    "implemented.\nThese documentation files are currently being reviewed by the "
    "generic-fallback agent (D-20)."
)

# Rubrics are inline because agents run in the reviewed repository: a bare
# skill-file pointer would resolve there and could silently omit the policy.
CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION = (
    "Cross-file symbol existence check (apply before flagging anything about a "
    "symbol defined OUTSIDE the diff):\n"
    "  1. Every referenced symbol not defined in this diff -- a function, a "
    "subcommand invoked by a CLI wrapper, a trait method implemented by "
    "generated code, a config field, a CLI flag -- must be verified to exist "
    "in the checked-out repo before you report a finding about it.\n"
    "  2. Evidence (Gate-2): `rg` for the definition in the repo and cite the "
    "file:line where it is declared. Never assert a symbol's behavior from the "
    "call site alone.\n"
    "  3. If no definition can be found, say so explicitly and downgrade the "
    "finding's confidence -- an unresolved reference is reportable only when "
    "the missing definition is real, never when you simply failed to locate it."
)

_CONFIG_FLOW_TRACE_RULES = (
    "  1. Trace the full path of each plumbed field: config struct -> driver "
    "config -> request construction.\n"
    "  2. During investigation, identify where each field is parsed, forwarded, "
    "and reaches the request. This is an investigation method, not extra output: "
    "report only substantiated findings inside the required schema.\n"
    "  3. Flag silent drops -- a field parsed but never forwarded to the next "
    "layer.\n"
    "  4. Flag double-resolves -- the same value read twice at different points "
    "with the source able to change between reads (TOCTOU)."
)

CONFIG_FLOW_TRACE_INSTRUCTION = (
    "Config/env flow trace (apply to every config field or env var plumbed "
    "through layers):\n" + _CONFIG_FLOW_TRACE_RULES
)

TRUST_MODEL_INSTRUCTION = (
    "Trust-model check (apply to every security-relevant marker: cache-control "
    "injection, trust boundaries, escaping, credential forwarding):\n"
    "  For each marker, state the trust model in one sentence: who is the "
    "untrusted party here, and does this path honor the boundary?\n"
    "  Flag any path that instructs an untrusted party to retain or forward "
    "sensitive content -- e.g. an edge proxy echoing an untrusted response's "
    "cache-control directive, or credentials passed through an intermediate hop."
)

VERIFICATION_PROTOCOL_INSTRUCTION = (
    "Before writing findings, apply the verification gates "
    "(stated inline here — no skill file read is required):\n"
    "  Gate-0 anti-confabulation (before ANY finding): echo the exact artifact "
    "you are judging — file:line plus the relevant changed code or investigation result. "
    "Never infer a finding from the branch name, cwd, or memory. A finding needs a concrete "
    "code basis; a path or speculative note alone is not enough.\n"
    "  Gate 1 (context): use the supplied diff and context first. Inspect the full enclosing "
    "symbol or configuration section when more context would resolve the concern; state the "
    "file path and line range you are judging.\n"
    "  Gate 2 (grounding): identify the concrete code basis for the finding from supplied "
    "context or ordinary investigation, and explain its trigger and consequence.\n"
    "  Gate 3 (severity): calibrate severity to impact; a request for net-new "
    "code that did not exist in scope is at most low.\n"
    "Do NOT report a finding that fails any gate."
)

_TEST_QUALITY_RULES = (
    "  1. Would this test fail if the behavior under test were wrong? Scan for "
    "vacuous assertions — e.g. `read_to_string(...).unwrap_or_default()` "
    "returning empty on failure, expected values built with the same helper "
    "under test, a wait/retry helper returning the last nonmatching frame.\n"
    "  2. Does it assert observable consequences (output, filesystem, exit code, "
    "store state) rather than internal fields/pointers/dispatch plumbing "
    "(`context as *const _ as usize`, dispatch internals, event payloads with no "
    "observable check)?\n"
    "  3. Is it deterministic (no sleeps, no `yield_now()` reaping assumptions, "
    "no environment leaks — require restore guards for any env mutation)?\n"
    "  4. Does it exercise the new behavior through the canonical public path (no "
    "raw `system_prompt` copies, no bypassing the public API the behavior lives "
    "behind)?\n"
    "  5. Does it compile on all platforms (`#[cfg]` gates)?\n"
    "Layering awareness: legitimate pure-function seams are fine — a unit test of "
    "a pure `build_driver_request` or driver-boundary propagation helper is NOT an "
    "internal-field assertion. Flag a seam ONLY when it bypasses the observable "
    "behavior the test claims to cover."
)

TEST_QUALITY_RUBRIC_INSTRUCTION = (
    "Apply the test-quality rubric to every test hunk in the diff "
    "(stated inline here — no skill file read is required):\n" + _TEST_QUALITY_RULES
)

ANTI_SLOP_RUBRIC_INSTRUCTION = (
    "Apply the maintainability rubric to code changed by this diff "
    "(stated inline here -- no skill file read is required):\n"
    "  1. Report added complexity or duplication only when it creates a concrete "
    "maintenance consequence or violates an established repository convention. "
    "Explain that consequence or cite the convention and its applicable source.\n"
    "  2. Check existing canonical helpers and surrounding ownership before "
    "recommending extraction or reuse. Size, single-use variables, and wrappers "
    "alone do not establish a defect.\n"
    "  3. Maintainability-only findings are medium/low, never high.\n"
    "  4. Report only newly introduced or worsened problems, scoped to this "
    "diff's contribution. Omit subjective refactoring preferences."
)


def _context_pointers(
    *,
    intent_path: Path,
    alternatives_path: Path,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
) -> str:
    """Reference TTT artifacts, inlining captured intent when available.

    Authoritative intent carries PR provenance and the precedence rule. Omit
    alternatives while the concurrent wonder pass has not written its artifact.
    """
    captured_intent = inline_context_file(intent_path)
    if captured_intent is not None:
        head = (
            f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n"
            "TTT intent summary (complete captured artifact; do not re-read intent.md):\n"
            + json.dumps(captured_intent, ensure_ascii=False)
        )
        if intent_authoritative:
            head += ("\nThis records the author's stated intent from the pull-request description.\n"
                     f"{AUTHORITATIVE_INTENT_BLOCK}")
        if include_alternatives:
            head += f"\nTTT alternative-review findings are at {alternatives_path}."
        return head
    alternatives_paragraph = (
        f"TTT alternative-review findings are at {alternatives_path}. Use them as a "
        f"starting point -- you may deepen, confirm, or dismiss each finding with "
        f"language-specific evidence."
    )
    if intent_authoritative:
        head = (
            f"TTT intent summary is at {intent_path}. Read it before starting your "
            f"review -- it records the author's stated intent from the pull-request "
            f"description."
            f"\n{AUTHORITATIVE_INTENT_BLOCK}"
        )
        return f"{head}\n{alternatives_paragraph}" if include_alternatives else head
    head = (
        f"TTT intent summary is at {intent_path}. Read it before starting your review "
        f"so your findings align with the author's stated intent."
    )
    return f"{head}\n{alternatives_paragraph}" if include_alternatives else head


def _review_context_parts(
    exploration_dir: Path | None,
    cwd: Path,
    intent_path: Path,
    alternatives_path: Path,
    *,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    prior_commits: str | None = None,
) -> list[str]:
    """Shared context block for the review/adjudication prompts: exploration
    pointer, settled-decisions block, CWD grounding, and TTT context pointers.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(
        intent_path=intent_path,
        alternatives_path=alternatives_path,
        intent_authoritative=intent_authoritative,
        include_alternatives=include_alternatives,
    ))
    return parts


def _artifact_footer(output_path: Path) -> str:
    """The host-managed review-artifact instruction shared by the review builders."""
    return (
        f"Host-managed review artifact: {output_path}. "
        "Return only the JSON object required by the output schema, with issues. "
        "Do not write review files or return a separate markdown report; the host persists the result."
    )


def _stack_scope_instruction(stack_name: str, files: list[str]) -> str:
    """Name the assigned files and the boundary of parallel stack reviews."""
    joined = ", ".join(files)
    return (
        f"You are reviewing the {stack_name} stack. Assigned files: {joined}\n"
        f"Do NOT review files from other stacks -- their reviews are running in "
        f"parallel and will be merged afterwards.\n"
        "Read another stack's file only to resolve a concrete candidate in your assigned "
        "changed behavior. End that context trace when the candidate is resolved; do "
        "not independently audit the other stack's workflow, tests, or configuration.\n"
    )


def _diff_instruction(
    diff_path: Path,
    files: list[str],
    *,
    inline_diff: str | None = None,
) -> str:
    """Inline complete stack hunks or point to the full diff artifact.

    Both forms name the persisted hunk index as changed-line authority. Inline
    assignments may be judged from supplied hunks when they provide enough context.
    """
    if inline_diff:
        return (
            "Relevant diff hunks for your stack (inlined; do NOT re-Read "
            "diff.patch for these — the hunks are already here):\n\n"
            f"{inline_diff.rstrip()}\n\n"
            f"{_hunk_index_authority(diff_path)}\n\n"
            "Focus on hunks that touch your stack's files. Use ordinary repository tools "
            "when additional context helps resolve a concrete concern."
        )
    joined = ", ".join(files)
    return (
        f"{_full_diff_pointer(diff_path)}\n"
        f"{_hunk_index_authority(diff_path)}\n\n"
        f"Focus on hunks that touch your stack's files: {joined}."
    )


def _frontier_read_instruction(frontier_files: list[str]) -> str:
    """Name sibling-shard interface files permitted as supporting context."""
    joined = ", ".join(frontier_files)
    return (
        f"Cross-shard interface file(s): this shard's review targets reference "
        f"files assigned to sibling shards. Read the following cross-shard "
        f"interface file(s) for context (they are NOT part of this shard's "
        f"review targets): {joined}."
    )


def _review_stage_context(review_stage: dict[str, Any], *, intent_authoritative: bool) -> str:
    """Use only host-selected, bounded inputs; their transport owns path confinement."""
    labels = review_stage.get("context_inputs", [])
    parts = ["Sanctioned inputs available for this assignment: " + ", ".join(labels) + "." if labels else
             "No shared artifact context is assigned to this stage. Use the supplied assignment and ordinary tools."]
    parts.append("For host artifacts, use the captured sanctioned bytes; do not read their private storage paths."
        if review_stage.get("context_transport") == "inline" else
        "For host artifacts, use only the exact sanctioned pointers supplied by the host.")
    if review_stage.get('supporting_bundle'):
        contents = (
            "a compact whole-change inventory and binding with navigation instructions for deferred "
            "bounded diff parts"
            if review_stage['stage'] == 'integration' else
            "the complete bounded assignment diff, hunk index and binding"
        )
        parts.append(
            f"The supporting_bundle contains {contents}; reuse its captured inline content without another read."
            if review_stage.get('context_transport') == 'inline' else
            f"Read the supporting_bundle once for {contents}, then reuse it; separate legacy diff/index reads "
            "are unnecessary.")
    if review_stage.get('supporting_catalog'):
        parts.append('The optional supporting_catalog provides the complete exact file/target/pointer inventory '
            'for deferred bounded diff parts. Read its supplied exact pointer and bounded child catalogs '
            'only when relevant supporting parts are needed, using explicitly listed pointers. '
            'Catalogs are optional navigation aids for the assigned change, not a reading checklist.')
    if review_stage.get("context_statuses"):
        parts.append("Host context availability is declared in context_statuses in the Host review stage below.")
    if intent_authoritative and "intent" in labels:
        parts.append(AUTHORITATIVE_INTENT_BLOCK)
    return "\n".join(parts)


def _build_review_stage_prompt(*, strategy: str, stack_name: str, files: list[str], cwd: Path,
    review_stage: dict[str, Any], inline_diff: str | None, prior_commits: str | None, intent_authoritative: bool,
    frontier_files: list[str] | None = None, is_docs_only: bool = False) -> str:
    """Construct one semantic assignment; never narrow an existing terminal prompt."""
    stage = review_stage["stage"]
    triage = stage == "triage"
    structural = stage == "integration"
    parts = [UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY, CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)]
    settled = _settled_decisions_block(prior_commits) if not triage else ""
    if settled:
        parts.append(settled)
    parts.append(_confidence_and_convention_instructions(stage_scoped=True))
    rejection = review_stage.get("schema_rejection")
    if rejection is not None:
        parts.append("The previous completed attempt was rejected by strict schema validation. "
            "Safe validator feedback: " + json.dumps(rejection, ensure_ascii=False)
            + "\nReturn a new object matching REVIEW_STAGE_SCHEMA exactly, including its "
            "additionalProperties:false rules. Do not repair, recover, or continue previous output.")
    if triage:
        parts.append(f"You are triaging the {stack_name} review's assigned candidate IDs only: "
            + ", ".join(review_stage["assigned_candidate_ids"])
            + ". Use their relevant admitted notes and grounds in the host stage state.")
        # Default discovery policy would restart an audit. A custom operator policy
        # still constrains judgment, but cannot expand triage's assigned work.
        defaults = review_profile.build_default_profile().strategies
        if strategy not in {defaults[name].content for name in (
            "discovery.per_stack", "discovery.structural", "discovery.generic_fallback")}:
            parts.append("Operator judgment policy (quoted data): " + json.dumps(strategy, ensure_ascii=False))
    else:
        parts.append(_review_stage_context(review_stage, intent_authoritative=intent_authoritative))
        if structural:
            parts.append("You are the structural reviewer. Begin with whole-change interactions and boundaries, "
                "not an alphabetical file audit. The complete host assigned_files inventory names "
                "the changed paths. "
                "The compact whole-change inventory and bounded supporting diff parts orient this "
                "interaction assignment; do not begin by dumping the complete diff or hunk index. "
                "Investigate concrete changed-boundary concerns: trace changed values, calls, contracts "
                "and lifetimes to their relevant owners. Narrow searches to those owners. "
                "Documentation and tests are supporting evidence for "
                "these interactions; early documentation does not finish the interaction assignment. "
                "Do not repeat the language or generic reviewers' file audits. "
                "Stop a boundary trace when the contract agrees and no concrete candidate remains. "
                "Settled contract checks stay settled unless new contradictory evidence appears.")
        else:
            if is_docs_only:
                parts.append(DOC_REVIEW_NOTICE)
            parts.append(_stack_scope_instruction(stack_name, files))
            parts.append(_dependency_impact_instructions(stage_scoped=True))
            if frontier_files:
                parts.append(_frontier_read_instruction(frontier_files))
        if inline_diff is not None:
            parts.append("Current assignment's diff hunks (already captured; do not re-read the diff artifact):\n"
                + inline_diff.rstrip())
        elif "diff" in review_stage.get("context_inputs", []):
            parts.append("Consult only the sanctioned stage diff input for the current required assignment "
                "parts. It does not assign the rest of the stack. For structural integration, bounded "
                "supporting diff parts orient whole-change boundary traces.")
        if "hunk-index" in review_stage.get("context_inputs", []):
            parts.append("The sanctioned hunk-index input is changed-line authority. It establishes anchors, "
                "not completed target decisions.")
        parts.append("Operator judgment policy for the assigned work:\n" + strategy)
        if not structural:
            parts.append("Apply the test-quality rubric only to assigned test hunks and tests needed to "
                "decide a concrete candidate in this assignment:\n" + _TEST_QUALITY_RULES)
            parts.append("Config/env flow trace (apply only to changed fields in this assignment or fields "
                "needed to decide its concrete candidates):\n" + _CONFIG_FLOW_TRACE_RULES)
            if base_stack_name(stack_name) == "rust":
                parts.append(WIRE_CONTRACT_RUST_INSTRUCTION)
            elif stack_name == "generic-fallback":
                parts.append(WIRE_CONTRACT_GENERIC_INSTRUCTION)
        parts.append(ANTI_SLOP_RUBRIC_INSTRUCTION)
        parts.append(TRUST_MODEL_INSTRUCTION)
        if structural:
            parts.append(CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION)
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(SEVERITY_RUBRIC)
    parts.append(STAGED_REVIEW_GUIDANCE)
    parts.append(f"Advisory stage tool-call target: {review_stage['advisory_tool_call_target']}. "
        f"Remaining hard cumulative tool allowance: {review_stage['remaining_tool_calls']}.")
    if not review_stage.get('response_contract'):
        parts.append("REVIEW_STAGE_SCHEMA:\n" + json.dumps(REVIEW_STAGE_SCHEMA, ensure_ascii=False))
    return "\n\n".join(parts)


def build_per_stack_prompt(
    *,
    strategy: str,
    stack_name: str,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
    prior_commits: str | None = None,
    inline_diff: str | None = None,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    frontier_files: list[str] | None = None,
    review_stage: dict[str, Any] | None = None,
) -> str:
    """Assemble a language review with profile policy and a host-owned file scope."""
    if review_stage is not None:
        return _build_review_stage_prompt(strategy=strategy, stack_name=stack_name, files=files, cwd=cwd,
            review_stage=review_stage, inline_diff=inline_diff, prior_commits=prior_commits,
            intent_authoritative=intent_authoritative, frontier_files=frontier_files)
    parts = _review_context_parts(
        exploration_dir, cwd, intent_path, alternatives_path,
        intent_authoritative=intent_authoritative, include_alternatives=include_alternatives,
        prior_commits=prior_commits,
    )
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction(stack_name, files))
    if frontier_files:
        parts.append(_frontier_read_instruction(frontier_files))
    parts.append(_diff_instruction(diff_path, files, inline_diff=inline_diff))
    parts.append(strategy)
    parts.append(TEST_QUALITY_RUBRIC_INSTRUCTION)
    parts.append(ANTI_SLOP_RUBRIC_INSTRUCTION)
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(CONFIG_FLOW_TRACE_INSTRUCTION)
    parts.append(SEVERITY_RUBRIC)
    parts.append(TRUST_MODEL_INSTRUCTION)
    if base_stack_name(stack_name) == "rust":
        parts.append(WIRE_CONTRACT_RUST_INSTRUCTION)
    parts.append(_artifact_footer(output_path))
    return "\n\n".join(parts)


def build_structural_prompt(
    *,
    strategy: str,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
    prior_commits: str | None = None,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    inline_diff: str | None = None, review_stage: dict[str, Any] | None = None,
) -> str:
    """Assemble a structural review of the full change.

    The changed-file list anchors the review but does not restrict source reads:
    structural findings may require tracing shared helpers or layering elsewhere.
    """
    if review_stage is not None:
        return _build_review_stage_prompt(strategy=strategy, stack_name="structural", files=files, cwd=cwd,
            review_stage=review_stage, inline_diff=inline_diff, prior_commits=prior_commits,
            intent_authoritative=intent_authoritative)
    joined = ", ".join(files)
    parts: list[str] = _review_context_parts(
        exploration_dir, cwd, intent_path, alternatives_path,
        intent_authoritative=intent_authoritative, include_alternatives=include_alternatives,
        prior_commits=prior_commits,
    )
    parts.append(
        f"You are the structural reviewer. The full change spans: {joined}."
    )
    parts.append(_full_diff_pointer(diff_path))
    parts.append(strategy)
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(SEVERITY_RUBRIC)
    parts.append(ANTI_SLOP_RUBRIC_INSTRUCTION)
    parts.append(CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION)
    parts.append(TRUST_MODEL_INSTRUCTION)
    parts.append(_artifact_footer(output_path))
    return "\n\n".join(parts)


def build_arbiter_prompt(
    *,
    strategy: str,
    arbiter_input_path: Path,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
    intent_authoritative: bool = False,
) -> str:
    """Adjudicate selected high-severity or contested findings, echoing each arb_id.

    The arbiter may refine or reject existing findings, but may not discover new ones.
    """
    parts = _review_context_parts(
        exploration_dir, cwd, intent_path, alternatives_path, intent_authoritative=intent_authoritative
    )
    parts.append(_full_diff_pointer(diff_path))
    parts.append(strategy.format(arbiter_input_path=arbiter_input_path))
    parts.append(
        "Return a single JSON object matching the structured-output schema: "
        '{"findings": [ ... ]}. Emit exactly one entry per input finding, echoing '
        "its `arb_id` unchanged. For each:\n"
        "  - keep: true if the finding is real and actionable; false to reject a "
        "false positive or a non-issue (rejected findings are dropped entirely).\n"
        "  - severity: your adjudicated severity per the rubric below (you may "
        "change it).\n"
        "  - confidence: your adjudicated HIGH | MEDIUM | LOW.\n"
        "  - description: a sharpened one-line summary (keep it about the same "
        "finding; do not repurpose the slot for a different issue).\n"
        "  - rationale: why it matters, grounded in what you actually read.\n\n"
        "Two input findings can be the same defect seen by two reviewers -- a "
        "language stack and the structural meta-stack read the same code at "
        "different altitudes, so a duplicate pair may be worded differently and "
        "one of them may be anchored at the whole file (`line: 0`) rather than a "
        "line. When two findings are the same defect, keep exactly one: set "
        "`keep: false` on the redundant entry and give the survivor the higher "
        "of the two severities that the rubric below actually supports for "
        "that defect -- never demote the survivor below either input, and "
        "never raise it past what the rubric licenses. Never drop both, and "
        "never reject a finding merely for overlapping with another -- only "
        "for being the same defect.\n\n" + SEVERITY_RUBRIC
    )
    return "\n\n".join(parts)


def build_supervise_prompt(
    *,
    strategy: str,
    supervise_input_path: Path,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Adjudicate canonical findings using the profile supervision strategy."""
    parts = _review_context_parts(exploration_dir, cwd, intent_path, alternatives_path)
    parts.append(_full_diff_pointer(diff_path))
    parts.append(strategy.format(supervise_input_path=supervise_input_path))
    parts.append(
        "Return one JSON object matching the structured-output schema with one "
        "verdict per finding when possible. Each verdict must echo the canonical "
        "id and choose exactly one action: allow, drop, edit, or hold. Explain "
        "the decision in reason. For edit, revise only severity, confidence, "
        "description, rationale, or evidence; never file, line, or id. Missing "
        "verdicts are treated as allow by the host.\n\n" + SEVERITY_RUBRIC
    )
    return "\n\n".join(parts)


def build_suppression_prompt(
    *,
    strategy: str,
    suppression_input_path: Path,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Re-examine borderline findings with a default-drop evidence requirement.

    This pass sees low-confidence or uncontested low-severity findings, distinct
    from the high-severity findings protected by the arbiter's fail-open behavior.
    """
    parts = _review_context_parts(exploration_dir, cwd, intent_path, alternatives_path)
    parts.append(_full_diff_pointer(diff_path))
    parts.append(strategy.format(suppression_input_path=suppression_input_path))
    parts.append(
        "Default to DROPPING each finding. Keep one ONLY when you can point at "
        "confirming evidence in the code that it is a real, actionable problem. "
        "Absence of evidence is a drop, not a keep -- a merely plausible or "
        "stylistic nit with no concrete grounding must be dropped."
    )
    parts.append(
        "Return a single JSON object matching the structured-output schema: "
        '{"findings": [ ... ]}. Emit exactly one entry per input finding, echoing '
        "its `sup_id` unchanged. For each:\n"
        "  - keep: true ONLY if you cite confirming evidence that the finding is "
        "real and actionable; false to drop an unconfirmed / immaterial finding.\n"
        "  - severity: your adjudicated severity per the rubric below.\n"
        "  - confidence: your adjudicated HIGH | MEDIUM | LOW.\n"
        "  - description: a sharpened one-line summary of the SAME finding.\n"
        "  - rationale: for a keep, the concrete evidence you found; for a drop, "
        "why it is not confirmable.\n"
        "  - evidence: the grounded `file:line` citation backing a kept finding.\n\n" + SEVERITY_RUBRIC
    )
    return "\n\n".join(parts)


def build_merge_prompt(
    *,
    strategy: str,
    per_stack_records_paths: list[Path],
    intent_path: Path,
    alternatives_path: Path,
    dedup_candidates_path: Path,
    exploration_dir: Path | None = None,
    failed_stacks: dict[str, str] | None = None,
    intent_authoritative: bool = False,
    resumed_from_arbiter: bool = False,
) -> str:
    """Merge parsed records and dedup candidates into canonical findings.

    The host appends structural findings after merge, stamps contiguous IDs, and
    validates source_uids against input records. Failed stacks remain explicit;
    resuming an arbiter session requires re-reading its rewritten input records.
    """
    records_block = "\n".join(f"  - {p}" for p in per_stack_records_paths)
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    context_lines: list[str] = [
        f"TTT intent summary: {intent_path}",
        f"TTT alternative-review findings: {alternatives_path}",
        f"Dedup pre-filter candidate pairs: {dedup_candidates_path}",
    ]
    if intent_authoritative:
        context_lines.insert(1, AUTHORITATIVE_INTENT_BLOCK)  # right after the intent line
    context_lines.append(f"Per-stack parsed records:\n{records_block}")
    parts.append("\n".join(context_lines))
    if failed_stacks:
        failed_block = "\n".join(
            f"  - {name}: {reason}" for name, reason in sorted(failed_stacks.items())
        )
        parts.append(
            "Uncovered stacks (review did not complete):\n"
            f"{failed_block}\n"
            "Some inputs may contain validated partial records from these stacks. Retain their "
            "substantiated findings while keeping coverage explicitly incomplete.\n"
            "Note these uncovered stacks in your reasoning. Do NOT silently omit "
            "them -- downstream readers must be able to tell 'no findings' apart "
            "from 'this stack never ran'."
        )
    parts.append(strategy)

    parts.append(
        "Dedup adjudication:\n"
        "  dedup-candidates.json has two sections:\n\n"
        "  record_alt_pairs (record ↔ TTT alt-review):\n"
        "  - For each candidate pair, decide whether the two findings describe the\n"
        "    same concern. If yes, emit ONE item citing both sources as combined\n"
        "    evidence. If no, emit both items independently.\n\n"
        "  record_duplicate_pairs (record ↔ record):\n"
        "  - These are per-stack records with near-identical descriptions across\n"
        "    different files (e.g. the same architectural concern reported once per\n"
        "    affected file). When two records describe the same conceptual finding,\n"
        "    emit ONE item listing all affected files rather than repeating the\n"
        "    finding verbatim for each file.\n\n"
        "  - Concerns that span multiple stacks (contract drift, shared-type "
        "mismatches, API-contract misalignment) are cross-stack findings."
    )
    parts.append(
        "Item fields (MANDATORY):\n"
        "  - id: integer; any value -- the host renumbers contiguously.\n"
        "  - lens: \"per-stack\" for a single-stack finding, \"cross-stack\" for a "
        "concern spanning multiple stacks, and \"wonder\" for an "
        "alternatives.json-sourced finding. (Structural findings are appended by "
        "the host -- do NOT emit them yourself.)\n"
        "  - severity: one level per the rubric below.\n"
        "  - confidence: \"HIGH\" | \"MEDIUM\" | \"LOW\".\n"
        "  - file: the FULL repo-relative path exactly as it appears in the per-stack "
        "records (e.g. `services/my-svc/handler.py`, not just `handler.py`). "
        "Downstream tooling uses `git show <sha>:<FILE>` to resolve lines, so "
        "abbreviated paths will fail to post as inline comments.\n"
        "  - line: integer line number for the finding.\n"
        "  - description: the finding title / one-line summary, plain text.\n"
        "  - rationale: why it matters; cite the actual records filename or stack "
        "name -- e.g. `(Sources: python-records item 6, alternatives item 4)`. "
        "NEVER use the `#N` notation (e.g. `#6`); GitHub auto-links `#N` to "
        "repository issues/PRs, creating misleading links.\n"
        "  - related_files: array of repo-relative paths of every OTHER "
        "file a deduplicated/cross-file finding affects (besides the primary "
        "`file`). Always emit this key -- use an empty array (or a null value) "
        "when the finding spans one file. "
        "Emit real paths exactly as they appear in the records -- downstream "
        "dispatch uses this footprint to send the finding to the agents that own "
        "every file the fix will touch.\n"
        "  - source_uids: array of the `uid` values of EVERY record this item "
        "derives from. Each record in the per-stack records files carries a "
        "`uid` field (e.g. `\"uid\": \"python:1\"`); copy those values "
        "VERBATIM. For a deduplicated cross-stack item, list ALL contributing "
        "uids, not just the first. Always emit this key -- use an empty array "
        "when the item cannot be attributed to a specific record. NEVER invent, "
        "guess, or reformat a uid: a value that is not in the records is "
        "discarded, taking the item's provenance with it. This is the "
        "machine-readable provenance handle; the human-readable "
        "`(Sources: ...)` citation still belongs in `rationale` for the reader, "
        "so emit both.\n\n"
        "Rules:\n"
        "  - Each item's `file` contains EXACTLY ONE path. For a concern that spans "
        "multiple files AND was NOT flagged as a duplicate in "
        "record_duplicate_pairs, emit a separate item per file. For deduplicated "
        "findings (same concern across files), emit ONE item with the primary file "
        "and list every other affected file in the `related_files` array (NOT buried "
        "in the rationale).\n"
        "  - Do not invent findings not supported by the source records.\n\n" + SEVERITY_RUBRIC
    )
    if resumed_from_arbiter:
        parts.append(
            "NOTE: this conversation is resumed from the arbitration turn. The "
            "per-stack record files listed above were REWRITTEN on disk after "
            "that turn (arbiter verdicts, and possibly suppression verdicts, "
            "were applied). You MUST re-read every record file from disk — the "
            "records held in the resumed context are pre-adjudication and are "
            "no longer authoritative."
        )
    return "\n\n".join(parts)


def build_generic_fallback_prompt(
    *,
    strategy: str,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
    is_docs_only: bool = False,
    prior_commits: str | None = None,
    inline_diff: str | None = None,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    frontier_files: list[str] | None = None,
    review_stage: dict[str, Any] | None = None,
) -> str:
    """Review files without a dedicated stack; prepend the notice for documentation."""
    if review_stage is not None:
        return _build_review_stage_prompt(strategy=strategy, stack_name="generic-fallback", files=files, cwd=cwd,
            review_stage=review_stage, inline_diff=inline_diff, prior_commits=prior_commits,
            intent_authoritative=intent_authoritative, frontier_files=frontier_files, is_docs_only=is_docs_only)
    parts: list[str] = []
    if is_docs_only:
        parts.append(DOC_REVIEW_NOTICE)
    parts.extend(
        _review_context_parts(
            exploration_dir, cwd, intent_path, alternatives_path,
            intent_authoritative=intent_authoritative, include_alternatives=include_alternatives,
            prior_commits=prior_commits,
        )
    )
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction("generic-fallback", files))
    if frontier_files:
        parts.append(_frontier_read_instruction(frontier_files))
    parts.append(_diff_instruction(diff_path, files, inline_diff=inline_diff))
    parts.append(strategy)

    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(CONFIG_FLOW_TRACE_INSTRUCTION)
    parts.append(SEVERITY_RUBRIC)
    parts.append(TRUST_MODEL_INSTRUCTION)
    parts.append(WIRE_CONTRACT_GENERIC_INSTRUCTION)
    parts.append(_artifact_footer(output_path))
    return "\n\n".join(parts)
