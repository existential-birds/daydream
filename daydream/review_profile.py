"""Strict, versioned, immutable review-profile model (issue #885).

One strict, versioned, immutable review-profile value — resolved once per run
and recorded by canonical digest — so a future optimizer can mutate it,
benchmark it, and attribute results to the exact policy it tested.

Task 1 (R2): the strict profile model + stage schema + packaged default.
Later tasks add canonical serialization (R4), digests (R4), fail-closed
validation (R3), host invariants/caps (R5), typed clone (R8), normal-run
precedence + path-escape guard (R9), and the Harbor explicit-only resolver
mode (R10).

The model is deliberately separate from the lenient ``config_file.py``
loader: an invalid profile fails the run naming its source, it never
degrades to a default (R3).

Default strategy content (R7) is copied verbatim from named production
symbols with ``copied:`` provenance — never newly-written prose. The full
#886-authored replacement blocks land with #886; every stage here has a real
nonempty copy already.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from daydream.improve.prompts import AUDIT_PLAYBOOK_SECTIONS


class ProfileError(Exception):
    """A strict review-profile parse/validation failure.

    Carries the offending field (``kind``) and the profile source
    (``source``); every failure message names both (R3).
    """

    def __init__(self, kind: str, source: str):
        self.kind = kind
        self.source = source
        super().__init__(f"invalid review profile: {kind} (source: {source})")


# Every model-bearing review-spine and Improve-judgment stage (R2). This is
# the canonical stage registry the completeness guard (R13) and the #886
# migration manifest must match.
STAGE_KEYS: frozenset[str] = frozenset(
    {
        "exploration.repository_survey",
        "exploration.pattern_scan",
        "exploration.dependency_trace",
        "exploration.test_mapping",
        "intent",
        "alternatives",
        "discovery.per_stack",
        "discovery.structural",
        "discovery.generic_fallback",
        "parse",
        "uncovered_review",
        "arbitration",
        "suppression",
        "merge",
        "supervision",
        "verification",
        "improve.audit.correctness",
        "improve.audit.security",
        "improve.audit.performance",
        "improve.audit.tests",
        "improve.audit.tech-debt",
        "improve.audit.dependencies",
        "improve.audit.dx",
        "improve.audit.docs",
        "improve.vetting",
    }
)


@dataclass(frozen=True)
class Strategy:
    """One stage's profile-owned strategy component.

    ``content`` is the strategy text (replaceable by the profile / clone).
    ``source`` records provenance: ``copied: <module>.<symbol>`` or
    ``authored: #886 <strategy name>`` (R7).
    """

    content: str
    source: str


@dataclass(frozen=True)
class Arbitration:
    """Bounded pipeline section: arbitration (R2)."""

    enabled: bool = True
    min_severity: str = "high"
    contested_location: bool = True


@dataclass(frozen=True)
class Suppression:
    """Bounded pipeline section: precision-mode suppression (R2).

    Mirrors the production opt-in default: the suppression pass is OFF by
    default (issue #232 precision mode; ``orchestrator._precision_mode``).
    """

    enabled: bool = False
    severity_classes: tuple[str, ...] = ("low", "medium")
    confidence_classes: tuple[str, ...] = ("low",)


@dataclass(frozen=True)
class Pipeline:
    """The bounded pipeline section (R2).

    Defaults mirror ``config.py:386-388`` (uncovered sweep),
    ``config.DEFAULT_DEEP_SHARD_*`` family, and the deep pipeline's product
    defaults (structural meta-stack always on; arbitration on
    high-severity/contested; suppression opt-in off).
    """

    structural_enabled: bool = True
    uncovered_sweep_enabled: bool = True
    uncovered_sweep_max_files: int = 10
    uncovered_sweep_min_hunk_lines: int = 5
    arbitration: Arbitration = field(default_factory=Arbitration)
    suppression: Suppression = field(default_factory=Suppression)


@dataclass(frozen=True)
class ReviewProfile:
    """The single per-run review-profile value.

    ``schema_version``: int; ``name``: human-readable; ``strategies``: one
    named ``Strategy`` per ``STAGE_KEYS`` entry; ``pipeline``: the bounded
    pipeline section.
    """

    schema_version: int = 1
    name: str = ""
    strategies: dict[str, Strategy] = field(default_factory=dict)
    pipeline: Pipeline = field(default_factory=Pipeline)


def build_default_profile() -> ReviewProfile:
    """Return the packaged default profile (R7).

    Every ``STAGE_KEYS`` entry gets nonempty, real strategy content copied
    verbatim from its named production symbol, with ``copied:`` provenance.
    """

    def _exploration(source_symbol: str, text: str) -> Strategy:
        return Strategy(content=text, source=f"copied: {source_symbol}")

    strategies: dict[str, Strategy] = {
        "exploration.repository_survey": _exploration(
            "daydream.prompts.exploration_subagents.build_repo_survey_prompt",
            (
                "You are the **repo-survey** specialist. Survey this repository as a whole\n"
                "and report the conventions an implementation plan would have to preserve. There\n"
                "is no change set here — you are describing the repository's steady state, not\n"
                "reviewing edits."
            ),
        ),
        "exploration.pattern_scan": _exploration(
            "daydream.prompts.exploration_subagents.build_pattern_scanner_prompt",
            (
                "You are the **pattern-scanner** specialist. Detect codebase conventions\n"
                "and read guideline files relevant to the changes below."
            ),
        ),
        "exploration.dependency_trace": _exploration(
            "daydream.prompts.exploration_subagents.build_dependency_tracer_prompt",
            (
                "You are the **dependency-tracer** specialist. Extend the affected-files\n"
                "list beyond the static-resolved imports by grepping for call sites and\n"
                "reading the implementations. For every import or call edge you confirm,\n"
                "emit a Dependency record."
            ),
        ),
        "exploration.test_mapping": _exploration(
            "daydream.prompts.exploration_subagents.build_test_mapper_prompt",
            (
                "You are the **test-mapper** specialist. Locate test files for each modified\n"
                "source file using conventional path mapping (tests/test_X.py, *.test.ts,\n"
                "*_test.go, tests/<crate>_test.rs). Emit a FileInfo with role=\"test\" for\n"
                "each test file you find, and set source_file to the source file it covers."
            ),
        ),
        "intent": Strategy(
            content=(
                "You have full access to explore the codebase. Read the diff file at "
                "{diff_path} and examine the codebase to understand the intent of these changes. "
                "That diff is the complete review target, already computed against the "
                "repository's base branch — this run is not tied to a GitHub pull request, so "
                "do not look up, list, or ask about pull requests. Do not invoke any skills or "
                "slash commands. Present your understanding concisely — what problem is being "
                "solved and how — as plain text in your reply."
            ),
            source="copied: daydream.phases.build_intent_prompt",
        ),
        "alternatives": Strategy(
            content=(
                "The intent of this PR has been confirmed as:\n\n"
                "{intent_summary}\n\n"
                "Given this intent, explore the codebase and evaluate the implementation "
                "in the diff at {diff_path}. Report only concrete problems you can substantiate "
                "with evidence — correctness bugs, design decisions that will cause a real "
                "failure, or violations of a Codebase Convention above. Do NOT list stylistic "
                "preferences, speculative 'nice to have' opinions, or alternatives you cannot "
                "tie to a concrete downside.\n\n"
                "Return a numbered list of issues. For each issue, include: a sequential id "
                "number, a brief title, a description of the concrete problem and the evidence "
                "for it, a severity level (high/medium/low), a concrete recommendation for how "
                "to address it, and the relevant file paths.\n\n"
                "If the implementation is solid and you wouldn't change anything, return an empty issues list."
            ),
            source="copied: daydream.phases.build_alternative_review_prompt",
        ),
        "discovery.per_stack": Strategy(
            content=(
                "You are reviewing the {stack_name} stack. Your assigned files are an "
                "inclusion obligation: read and review EACH one in full -- a file you "
                "did not read is not covered by this review.\n"
                "  Assigned files: {files}\n"
                "Do NOT review files from other stacks -- their reviews are running in "
                "parallel and will be merged afterwards."
            ),
            source="copied: daydream.deep.prompts._stack_scope_instruction",
        ),
        "discovery.structural": Strategy(
            content=(
                "You are the structural reviewer. The full change spans: {joined}. "
                "The structural rubric applies repo-wide -- read any file in the "
                "codebase as needed (Read/Grep/Bash) to judge whether canonical "
                "helpers exist, file-size budgets are honored, and the change makes "
                "the codebase easier or harder to live with."
            ),
            source="copied: daydream.deep.prompts.build_structural_prompt",
        ),
        "discovery.generic_fallback": Strategy(
            content=(
                "Review these files for correctness, clarity, and consistency with the "
                "author's intent. Apply language-agnostic review practices."
            ),
            source="copied: daydream.deep.prompts.build_generic_fallback_prompt",
        ),
        "parse": Strategy(
            content=(
                "Read the review output file at {review_output_path}.\n\n"
                "Extract ONLY actionable issues that need fixing. Skip these sections entirely:\n"
                "- \"Good Patterns\" or \"Strengths\"\n"
                "- \"Summary\" sections\n"
                "- Any positive observations\n"
                "{severity_hint}{verdicts_hint}\n"
                "For each issue found, return a JSON object with this structure:\n"
                "{{\"issues\": [\n"
                "  {{\"id\": 1, \"description\": \"Brief description of the issue\", \"file\": \"path/to/file.py\", \"line\": 42{severity_field}}}\n"
                "]{verdicts_example}}}\n\n"
                "If there are no actionable issues, return: {{\"issues\": []{verdicts_empty}}}\n"
            ),
            source="copied: daydream.phases.phase_parse_feedback",
        ),
        "uncovered_review": Strategy(
            content=(
                "You are the uncovered file sweep reviewer for the deep-review "
                "pipeline (issue #309).\n"
                "The changed file {file} was NOT read by any per-stack reviewer, "
                "so you are the second pass that covers it. Review ONLY this "
                "file's hunks below -- correctness, error handling, test quality, "
                "and maintainability. Do NOT review other files."
            ),
            source="copied: daydream.deep.coverage.build_uncovered_sweep_prompt",
        ),
        "arbitration": Strategy(
            content=(
                "You are the arbiter. The cheaper per-stack reviewers flagged the "
                "high-severity and contested findings listed in {arbiter_input_path}. "
                "Re-review each one against the actual code (Read/Grep/Bash) and the "
                "diff. You are adjudicating their work, NOT starting a fresh review: do "
                "not introduce findings that are not in the input list."
            ),
            source="copied: daydream.deep.prompts.build_arbiter_prompt",
        ),
        "suppression": Strategy(
            content=(
                "You are the suppression reviewer. The cheaper per-stack reviewers "
                "flagged the borderline, low-confidence / low-severity findings listed "
                "in {suppression_input_path}. These were NOT contested and NOT "
                "high-severity, so no heavyweight arbiter looked at them. Your job is to "
                "cut false positives: re-examine each one against the actual code "
                "(Read/Grep/Bash) and the diff. You are adjudicating their work, NOT "
                "starting a fresh review: do not introduce findings that are not in the "
                "input list."
            ),
            source="copied: daydream.deep.prompts.build_suppression_prompt",
        ),
        "merge": Strategy(
            content=(
                "You are the cross-stack merge agent. Read every artifact above by path -- "
                "do NOT re-run any reviews. Return a single JSON object matching the "
                "structured-output schema: {\"items\": [ ... ]}. Each item is one "
                "actionable finding. Emit nothing else."
            ),
            source="copied: daydream.deep.prompts.build_merge_prompt",
        ),
        "supervision": Strategy(
            content=(
                "Supervisor adjudication: review the canonical findings listed in "
                "{supervise_input_path}. This is an adjudication pass, not a fresh "
                "review: do not invent findings or change their file, line, or id."
            ),
            source="copied: daydream.deep.prompts.build_supervise_prompt",
        ),
        "verification": Strategy(
            content=(
                "You are the recommendation-verifier agent. Your job is to audit each "
                "numbered issue in the finding list below against the actual codebase "
                "and decide whether its recommendation is consistent with trait/interface "
                "specs and sibling implementations."
            ),
            source="copied: daydream.deep.prompts.build_verification_prompt",
        ),
    }

    # Improve audit playbooks: pure dict values, copied verbatim (R7).
    for category in AUDIT_PLAYBOOK_SECTIONS:
        strategies[f"improve.audit.{category}"] = Strategy(
            content=AUDIT_PLAYBOOK_SECTIONS[category],
            source=(
                "copied: daydream.improve.prompts.AUDIT_PLAYBOOK_SECTIONS"
                f"[{category}]"
            ),
        )

    strategies["improve.vetting"] = Strategy(
        content=(
            "You are the improve vet. Re-open every cited location before deciding\n"
            "whether to keep a candidate. Apply the `beagle-core:review-verification-protocol`\n"
            "skill while checking the evidence."
        ),
        source="copied: daydream.improve.prompts.build_vet_prompt",
    )

    return ReviewProfile(
        schema_version=1,
        name="default",
        strategies=strategies,
        pipeline=Pipeline(),
    )
