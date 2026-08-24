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

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

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


# Host-owned severity/confidence vocabularies (R5; mirror the repo's allowed
# sets: severity is the lowercase low|medium|high scale -- benchmark/mapping.py,
# benchmark/cli.py:284-287; confidence is the uppercase HIGH|MEDIUM|LOW schema
# enum -- phases.py:4048,4250, deep/prompts.py arbiter/suppression/merge).
_SEVERITY_LEVELS: frozenset[str] = frozenset(("low", "medium", "high"))
_CONFIDENCE_LEVELS: frozenset[str] = frozenset(("HIGH", "MEDIUM", "LOW"))


# Host-owned invariant keys (R5): a benchmark repository can never configure its
# own evaluator. The profile can only tune the enumerated strategy components and
# bounded pipeline fields; everything here stays host-owned: backends/models/
# effort, trust/egress/privacy, Harbor judge/verifier/matching/gold/scoring,
# skill names, finding/output schemas, evidence/location rules, and executable
# behavior (callbacks, commands, filesystem paths). Rejected wherever they appear.
HOST_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "backend",
        "provider",
        "model",
        "effort",
        "trust_mode",
        "egress",
        "harbor_judge_model",
        "skill_name",
        "findings_schema",
        "output_schema",
        "severity_vocabulary",
        "confidence_vocabulary",
        "evidence",
        "location_rules",
        "callbacks",
        "commands",
        "skill_invocation",
        "filesystem_paths",
        "privacy",
        "credentials",
        "matching",
        "gold",
        "scoring",
        "verifier",
        "judge",
    }
)

# Host safety/cost caps (mirror config.py:386-388). After parsing a profile,
# lower profile values are CLAMPED UP to the host floor (and capped at
# the host ceiling) BEFORE the digest is computed, so digest reflects the
# clamped semantic value. The caps themselves are production defaults: safety
# floors, not tunable budget knobs.
HOST_CAPS: dict[str, tuple[int | None, int | None]] = {
    # (floor, ceiling) — ceiling None = no ceiling.
    "uncovered_sweep_max_files": (1, 10),
    "uncovered_sweep_min_hunk_lines": (5, None),
}


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
    confidence_classes: tuple[str, ...] = ("LOW",)


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

    def to_canonical_dict(self) -> dict[str, object]:
        """Plain-dict projection of the fully-defaulted semantic value (R4).

        Excludes provenance (``Strategy.source``), raw source text, and any
        order/whitespace/comment artifacts so semantically-identical policies
        project identically.
        """
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "strategies": {
                key: strategy.content for key, strategy in sorted(self.strategies.items())
            },
            "pipeline": {
                "structural_enabled": self.pipeline.structural_enabled,
                "uncovered_sweep_enabled": self.pipeline.uncovered_sweep_enabled,
                "uncovered_sweep_max_files": self.pipeline.uncovered_sweep_max_files,
                "uncovered_sweep_min_hunk_lines": self.pipeline.uncovered_sweep_min_hunk_lines,
                "arbitration": {
                    "enabled": self.pipeline.arbitration.enabled,
                    "min_severity": self.pipeline.arbitration.min_severity,
                    "contested_location": self.pipeline.arbitration.contested_location,
                },
                "suppression": {
                    "enabled": self.pipeline.suppression.enabled,
                    "severity_classes": list(self.pipeline.suppression.severity_classes),
                    "confidence_classes": list(self.pipeline.suppression.confidence_classes),
                },
            },
        }

    @property
    def digest(self) -> str:
        """Canonical SHA-256 over sorted-key JSON of the defaulted semantic value (R4).

        Deterministic: independent of key order, whitespace, comments, and
        source path. Any semantic change to a strategy or pipeline value
        changes the digest.
        """
        canonical = json.dumps(
            self.to_canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
                "  {{\"id\": 1, \"description\": \"Brief description of the issue\", \"file\": \"path/to/file.py\", \"line\": 42{severity_field}}}\n"  # noqa: E501 (verbatim copy of the phase_parse_feedback literal)
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


# Host-owned protocol/envelope blocks each stage renders against (R13). The
# envelope classification is classification-only -- it names the host-owned
# protocol block (a real production symbol where one exists) each stage's
# strategy content is rendered against; the actual render split is #886. The
# strategy leg is derived from the packaged default's ``Strategy.source`` so
# the classification can never drift from the strategy content it classifies.
_ENVELOPE_BY_STAGE: dict[str, str] = {
    # Exploration specialists are cwd-grounded in the audited repo.
    "exploration.repository_survey": (
        "daydream.prompts.grounding.CWD_GROUNDING_INSTRUCTION"
    ),
    "exploration.pattern_scan": (
        "daydream.prompts.grounding.UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY"
    ),
    "exploration.dependency_trace": (
        "daydream.deep.prompts.CONFIG_FLOW_TRACE_INSTRUCTION"
    ),
    "exploration.test_mapping": (
        "daydream.deep.prompts.TEST_QUALITY_RUBRIC_INSTRUCTION"
    ),
    # Deep review spine.
    "intent": "daydream.prompts.grounding.CWD_GROUNDING_INSTRUCTION",
    "alternatives": "daydream.deep.prompts.TRUST_MODEL_INSTRUCTION",
    "discovery.per_stack": (
        "daydream.deep.prompts.CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION"
    ),
    "discovery.structural": "daydream.deep.prompts.VERIFICATION_PROTOCOL_INSTRUCTION",
    "discovery.generic_fallback": (
        "daydream.deep.prompts.VERIFICATION_PROTOCOL_INSTRUCTION"
    ),
    "parse": "daydream.improve.prompts.FINDING_FORMAT",
    "uncovered_review": (
        "daydream.deep.prompts.CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION"
    ),
    "arbitration": "daydream.deep.prompts.VERIFICATION_PROTOCOL_INSTRUCTION",
    "suppression": "daydream.deep.prompts.TRUST_MODEL_INSTRUCTION",
    "merge": "daydream.improve.prompts.FINDING_FORMAT",
    "supervision": "daydream.deep.prompts.CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION",
    "verification": "daydream.deep.prompts.VERIFICATION_PROTOCOL_INSTRUCTION",
    # Improve audits render against the host finding format + hard rules.
    "improve.audit.correctness": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.security": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.performance": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.tests": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.tech-debt": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.dependencies": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.dx": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.audit.docs": "daydream.improve.prompts.FINDING_FORMAT",
    "improve.vetting": "daydream.deep.prompts.VERIFICATION_PROTOCOL_INSTRUCTION",
}


ENVELOPE_CLASSIFICATION: dict[str, dict[str, str]] = {
    key: {
        "strategy": strategy.source,
        "envelope": _ENVELOPE_BY_STAGE[key],
    }
    for key, strategy in build_default_profile().strategies.items()
}


def parse_profile(toml_text: str, *, source: str = "<string>") -> ReviewProfile:
    """Strictly parse TOML into a fully-defaulted ``ReviewProfile`` (R3/R4).

    Fail-closed: an unknown key, an unsupported ``schema_version``, an
    invalid enum, a negative limit, or an inconsistent combination raises
    ``ProfileError`` naming the offending field and the source. A failed
    parse NEVER falls through to a default or lower-precedence profile.
    Omitted pipeline fields are filled from ``Pipeline()`` defaults so
    omitted-vs-explicit defaults hash identically (R4).

    Args:
        toml_text: Profile TOML source text.
        source: Human-readable source description for error messages.

    Raises:
        ProfileError: On any invalid profile, naming the offending field
            and the profile source.
    """
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"TOML parse failure: {exc}", source) from exc

    if not isinstance(data, dict):
        raise ProfileError("top level must be a table", source)

    _TOP_LEVEL_KEYS = frozenset({"schema_version", "name", "strategies", "pipeline"})

    # Host-owned keys are disjoint from the allowed top-level set, so check
    # them FIRST: otherwise an unoverridable host key would be swallowed by the
    # generic unknown-key branch and never surface the dedicated "host-owned"
    # rejection.
    host_owned = set(data) & HOST_OWNED_KEYS
    if host_owned:
        raise ProfileError(
            f"host-owned key `{sorted(host_owned)[0]}` cannot be set by a profile",
            source,
        )

    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ProfileError(f"unknown top-level key `{sorted(unknown)[0]}`", source)

    schema_version = data.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ProfileError("schema_version must be an integer", source)
    if schema_version != 1:
        raise ProfileError(
            f"unsupported schema_version {schema_version} (only 1 is supported)",
            source,
        )
    name = data.get("name", "")
    if not isinstance(name, str):
        raise ProfileError("name must be a string", source)

    strategies: dict[str, Strategy] = {}
    raw_strategies = data.get("strategies", {})
    if not isinstance(raw_strategies, dict):
        raise ProfileError("strategies must be a table", source)
    for key, raw in raw_strategies.items():
        if not isinstance(raw, dict):
            raise ProfileError(f"strategies.{key} must be a table", source)
        host_owned = set(raw) & HOST_OWNED_KEYS
        if host_owned:
            raise ProfileError(
                f"strategies.{key}: host-owned key `{sorted(host_owned)[0]}` "
                "cannot be set by a profile",
                source,
            )
        unknown = set(raw) - {"content", "source"}
        if unknown:
            raise ProfileError(
                f"strategies.{key}: unknown key `{sorted(unknown)[0]}`", source
            )
        content = raw.get("content", "")
        strat_source = raw.get("source", "")
        if not isinstance(content, str) or not isinstance(strat_source, str):
            raise ProfileError(f"strategies.{key}", source)
        strategies[key] = Strategy(content=content, source=strat_source)

    pipeline = _parse_pipeline(data.get("pipeline", {}), source=source)

    return ReviewProfile(
        schema_version=schema_version,
        name=name,
        strategies=strategies,
        pipeline=pipeline,
    )


def _parse_pipeline(data: object, *, source: str) -> Pipeline:
    """Parse the bounded pipeline section (R3 fail-closed, defaults for omitted fields)."""
    if not isinstance(data, dict):
        raise ProfileError("pipeline must be a table", source)
    _PIPELINE_KEYS = frozenset(
        {
            "structural_enabled",
            "uncovered_sweep_enabled",
            "uncovered_sweep_max_files",
            "uncovered_sweep_min_hunk_lines",
            "arbitration_enabled",
            "arbitration_min_severity",
            "arbitration_contested_location",
            "suppression_enabled",
            "suppression_severity_classes",
            "suppression_confidence_classes",
        }
    )
    unknown = set(data) - _PIPELINE_KEYS
    if unknown:
        raise ProfileError(f"pipeline: unknown key `{sorted(unknown)[0]}`", source)
    defaults = Pipeline()

    def _bool(key: str, fallback: bool) -> bool:
        value = data.get(key, fallback)
        if not isinstance(value, bool):
            raise ProfileError(f"pipeline.{key} must be a boolean", source)
        return value

    def _int(key: str, fallback: int) -> int:
        value = data.get(key, fallback)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProfileError(f"pipeline.{key} must be an integer", source)
        if value < 0:
            raise ProfileError(f"pipeline.{key} must not be negative", source)
        return value

    def _clamped_int(key: str, fallback: int) -> int:
        """Read a host-capped int, clamping into the host cap range (R5).

        Host caps are the floor: a profile supplying LOWER than the host cap
        is clamped up, never the reverse. The ceiling keeps a profile from
        raising a host cap. Clamping happens here (before digest) so the
        digest reflects the clamped semantic value.
        """
        value = _int(key, fallback)
        floor, ceiling = HOST_CAPS[key]
        if floor is not None:
            value = max(value, floor)
        if ceiling is not None:
            value = min(value, ceiling)
        return value

    def _severity_classes(
        key: str, fallback: tuple[str, ...], allowed: frozenset[str]
    ) -> tuple[str, ...]:
        value = data.get(key, fallback)
        if not isinstance(value, (list, tuple)):
            raise ProfileError(f"pipeline.{key} must be an array of strings", source)
        if not all(isinstance(item, str) for item in value):
            raise ProfileError(f"pipeline.{key} must be an array of strings", source)
        bad = [item for item in value if item not in allowed]
        if bad:
            raise ProfileError(
                f"pipeline.{key}: invalid class `{sorted(set(bad))[0]}`", source
            )
        return tuple(value)

    arbitration = Arbitration(
        enabled=_bool("arbitration_enabled", defaults.arbitration.enabled),
        min_severity=defaults.arbitration.min_severity,
        contested_location=_bool(
            "arbitration_contested_location", defaults.arbitration.contested_location
        ),
    )
    severity = data.get("arbitration_min_severity")
    if severity is not None:
        if not isinstance(severity, str) or severity not in _SEVERITY_LEVELS:
            raise ProfileError(
                f"pipeline.arbitration_min_severity must be one of "
                f"{sorted(_SEVERITY_LEVELS)}",
                source,
            )
        arbitration = Arbitration(
            enabled=arbitration.enabled,
            min_severity=severity,
            contested_location=arbitration.contested_location,
        )
    suppression = Suppression(
        enabled=_bool("suppression_enabled", defaults.suppression.enabled),
        severity_classes=_severity_classes(
            "suppression_severity_classes",
            defaults.suppression.severity_classes,
            _SEVERITY_LEVELS,
        ),
        confidence_classes=_severity_classes(
            "suppression_confidence_classes",
            defaults.suppression.confidence_classes,
            _CONFIDENCE_LEVELS,
        ),
    )
    if suppression.enabled and not suppression.confidence_classes:
        raise ProfileError(
            "pipeline.suppression: enabled but empty confidence class selection",
            source,
        )

    return Pipeline(
        structural_enabled=_bool("structural_enabled", defaults.structural_enabled),
        uncovered_sweep_enabled=_bool(
            "uncovered_sweep_enabled", defaults.uncovered_sweep_enabled
        ),
        uncovered_sweep_max_files=_clamped_int(
            "uncovered_sweep_max_files", defaults.uncovered_sweep_max_files
        ),
        uncovered_sweep_min_hunk_lines=_clamped_int(
            "uncovered_sweep_min_hunk_lines", defaults.uncovered_sweep_min_hunk_lines
        ),
        arbitration=arbitration,
        suppression=suppression,
    )


@dataclass(frozen=True)
class ResolvedProfile:
    """A resolved review profile with its source provenance (R9).

    ``source_kind`` is one of ``"explicit"``, ``"env"``, ``"repo"``, or
    ``"default"``; ``source_path`` is the path the profile came from (``None``
    for the packaged default); ``digest`` is the canonical digest of the
    resolved profile value.
    """

    profile: ReviewProfile
    source_kind: str
    source_path: Path | None = None

    @property
    def digest(self) -> str:
        return self.profile.digest

    @property
    def name(self) -> str:
        """Human-readable name of the resolved profile (delegates to the value)."""
        return self.profile.name


def _read_and_parse(path: Path, source: str) -> ReviewProfile:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileError(
            f"cannot read profile file (reason: {exc})", source
        ) from exc
    return parse_profile(text, source=source)


def _guard_repo_path(path: Path, repo_root: Path | None) -> Path:
    """Resolve a repo-committed profile path beneath ``repo_root`` (R9).

    The path-escape guard applies to REPOSITORY-committed paths: the value
    comes from the (untrusted) benchmarked repository's own config, so relative,
    absolute, and ``~``-expanded paths must ALL resolve beneath the repo root.
    ``Path.resolve()`` + a containment check reject ``..``, absolute,
    ``expanduser``, and symlink escapes to stop a benchmarked repository from
    pointing its own evaluator at an arbitrary filesystem path.
    """
    if repo_root is None:
        # No repo root supplied — resolve relative to the current dir (cwd is
        # the repo for normal runs).
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)
    expanded = path.expanduser()
    candidate = (repo_root / expanded).resolve()
    if not candidate.is_relative_to(repo_root.resolve()):
        raise ProfileError(
            f"repo-committed profile path escapes the repository root ({candidate})",
            str(path),
        )
    return candidate


def resolve_profile(
    *,
    explicit_path: str | None = None,
    file_config: object | None = None,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> ResolvedProfile:
    """Resolve the single per-run review profile from the four normal sources (R9).

    Precedence (highest wins; an invalid higher source raises and never falls
    through to a lower source):
    1. ``explicit_path`` — an explicit CLI/``RunConfig`` path.
    2. ``DAYDREAM_REVIEW_PROFILE`` env — a private user path.
    3. ``file_config.review_profile`` — a repo-committed path.
    4. Packaged default.
    """
    if explicit_path is not None:
        profile = _read_and_parse(Path(explicit_path), str(explicit_path))
        return ResolvedProfile(
            profile=profile,
            source_kind="explicit",
            source_path=Path(explicit_path),
        )

    if env is None:
        env = os.environ
    env_value = env.get("DAYDREAM_REVIEW_PROFILE")
    if env_value:
        raw = str(env_value)
        profile = _read_and_parse(Path(raw), raw)
        return ResolvedProfile(
            profile=profile, source_kind="env", source_path=Path(raw)
        )

    if file_config is not None:
        repo_path = getattr(file_config, "review_profile", None)
        if repo_path is not None:
            guarded = _guard_repo_path(Path(repo_path), repo_root)
            profile = _read_and_parse(guarded, str(guarded))
            return ResolvedProfile(
                profile=profile, source_kind="repo", source_path=guarded
            )

    return ResolvedProfile(
        profile=build_default_profile(), source_kind="default"
    )


def resolve_from_runconfig(cfg: object) -> ResolvedProfile:
    """Resolve the profile from a ``RunConfig`` (R1: once at composition root).

    The ``RunConfig`` carries the caller-derived ``review_profile_path``, the
    file config, and the target repo; this is the single seam the runner calls
    exactly once per run. The target is threaded as ``repo_root`` so a
    repo-committed RELATIVE ``file_config.review_profile`` path resolves beneath
    the target repo rather than the invoking cwd (R9).
    """
    file_config = getattr(cfg, "file_config", None)
    explicit = getattr(cfg, "review_profile_path", None)
    explicit_str = str(explicit) if explicit is not None else None
    repo_root = getattr(cfg, "target", None)
    repo_root = Path(repo_root) if repo_root else None
    return resolve_profile(
        explicit_path=explicit_str, file_config=file_config, repo_root=repo_root
    )


def resolve_harbor_profile(
    *,
    file_config: object | None = None,
    candidate_env: str = "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
    env: Mapping[str, str] | None = None,
) -> ResolvedProfile:
    """Resolve the Harbor run's review profile (R10: explicit-only mode).

    A DISTINCT resolver mode from :func:`resolve_profile`: a benchmarked
    repository can never configure its own evaluator, so this accepts ONLY the
    control-plane-supplied candidate (a dedicated
    ``DAYDREAM_REVIEW_PROFILE_CANDIDATE`` env var) or the packaged default. It
    must NOT read ``DAYDREAM_REVIEW_PROFILE`` (the normal-run env), the
    operator's normal defaults, or any ``file_config``/target-repo profile.

    When no candidate var is set -> the packaged default with
    ``source_kind="default"``. When set -> the candidate is parsed + validated
    fail-closed (naming its source); any failure raises ``ProfileError`` and
    the run aborts — never a fallback to a lower-precedence source.

    Args:
        file_config: Accepted for signature symmetry with
            :func:`resolve_profile`; deliberately UNREAD — a benchmarked
            repository can never point its own evaluation (the first Harbor
            test asserts exactly this).
        candidate_env: Env var name carrying the control-plane candidate path.
        env: Environment mapping; ``None`` reads ``os.environ`` (the trusted
            control-plane env in the Harbor agent container).

    Returns:
        The resolved ``ResolvedProfile``.

    Raises:
        ProfileError: On an invalid candidate, naming the candidate source.
    """
    if env is None:
        env = os.environ
    candidate = env.get(candidate_env)
    if candidate:
        raw = str(candidate)
        profile = _read_and_parse(Path(raw), raw)
        return ResolvedProfile(
            profile=profile, source_kind="candidate", source_path=Path(raw)
        )
    return ResolvedProfile(profile=build_default_profile(), source_kind="default")


def clone_with_overrides(
    base: ReviewProfile,
    overrides: dict[str, dict[str, object]],
) -> ReviewProfile:
    """Deep-copy ``base`` and apply named overrides, then re-validate (R8).

    ``overrides`` maps a ``STAGE_KEYS`` stage (or ``"pipeline"``) to a nested
    mapping of profile-owned fields to apply on top of the base. The clone is a
    full deep copy of the ``Strategy``/``Pipeline`` values, so un-overridden
    fields stay byte-identical (and a no-override clone preserves the digest).
    After applying overrides the result is re-validated — any host-owned
    override raises ``ProfileError`` — and the digest is recomputed from the
    updated canonical value. A no-op clone returns an equal canonical value
    (same digest); one override changes only that stage's bytes. Overrides
    operate on the typed model, never on rendered prompt text.
    """
    strategies: dict[str, Strategy] = {
        key: Strategy(content=strategy.content, source=strategy.source)
        for key, strategy in base.strategies.items()
    }
    pipeline = _copy_pipeline(base.pipeline)

    for key, raw_override in overrides.items():
        if not isinstance(raw_override, dict):
            raise ProfileError(
                f"clone override `{key}` must be a mapping", "<clone override>"
            )
        if key in STAGE_KEYS:
            unknown = set(raw_override) - {"content", "source"}
            if unknown:
                raise ProfileError(
                    f"clone {key}: unknown key `{sorted(unknown)[0]}`",
                    "<clone override>",
                )
            host_owned = set(raw_override) & HOST_OWNED_KEYS
            if host_owned:
                raise ProfileError(
                    f"clone {key}: host-owned key `{sorted(host_owned)[0]}` "
                    "cannot be set by an override",
                    "<clone override>",
                )
            existing = strategies[key]
            content = raw_override.get("content", existing.content)
            source = raw_override.get("source", existing.source)
            strategies[key] = Strategy(
                content=str(content),
                source=str(source),
            )
        elif key == "pipeline":
            pipeline = _pipeline_with_overrides(pipeline, raw_override)
        else:
            raise ProfileError(
                f"clone: unknown stage `{key}`", "<clone override>"
            )

    return ReviewProfile(
        schema_version=base.schema_version,
        name=base.name,
        strategies=strategies,
        pipeline=pipeline,
    )


def _pipeline_with_overrides(
    pipeline: Pipeline, override: dict[str, object]
) -> Pipeline:
    """Apply a pipeline override mapping onto a base ``Pipeline`` with re-validation.

    Flat keys mirror ``_parse_pipeline`` (``arbitration_enabled``,
    ``suppression_confidence_classes``, ...). ``override`` keys must be a subset
    of the parseable pipeline keys and must not be host-owned.
    """
    flattened: dict[str, object] = {
        "structural_enabled": pipeline.structural_enabled,
        "uncovered_sweep_enabled": pipeline.uncovered_sweep_enabled,
        "uncovered_sweep_max_files": pipeline.uncovered_sweep_max_files,
        "uncovered_sweep_min_hunk_lines": pipeline.uncovered_sweep_min_hunk_lines,
        "arbitration_enabled": pipeline.arbitration.enabled,
        "arbitration_min_severity": pipeline.arbitration.min_severity,
        "arbitration_contested_location": pipeline.arbitration.contested_location,
        "suppression_enabled": pipeline.suppression.enabled,
        "suppression_severity_classes": list(pipeline.suppression.severity_classes),
        "suppression_confidence_classes": list(
            pipeline.suppression.confidence_classes
        ),
    }
    unknown = set(override) - set(flattened)
    if unknown:
        raise ProfileError(
            f"clone.pipeline: unknown key `{sorted(unknown)[0]}`",
            "<clone override>",
        )
    host_owned = set(override) & HOST_OWNED_KEYS
    if host_owned:
        raise ProfileError(
            f"clone.pipeline: host-owned key `{sorted(host_owned)[0]}` "
            "cannot be set by an override",
            "<clone override>",
        )
    flattened.update(override)
    return _parse_pipeline(flattened, source="<clone override>")


def _copy_pipeline(source: Pipeline) -> Pipeline:
    """Return a structural deep copy of a ``Pipeline`` (dataclasses are frozen)."""
    return Pipeline(
        structural_enabled=source.structural_enabled,
        uncovered_sweep_enabled=source.uncovered_sweep_enabled,
        uncovered_sweep_max_files=source.uncovered_sweep_max_files,
        uncovered_sweep_min_hunk_lines=source.uncovered_sweep_min_hunk_lines,
        arbitration=Arbitration(
            enabled=source.arbitration.enabled,
            min_severity=source.arbitration.min_severity,
            contested_location=source.arbitration.contested_location,
        ),
        suppression=Suppression(
            enabled=source.suppression.enabled,
            severity_classes=source.suppression.severity_classes,
            confidence_classes=source.suppression.confidence_classes,
        ),
    )
