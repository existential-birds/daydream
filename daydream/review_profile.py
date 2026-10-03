"""Strict, versioned review policy resolved once per run and identified by content digest.

Invalid profiles fail with their source named; they never fall back to lower-
precedence policy. Packaged strategies retain provenance for their original text.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic.dataclasses import dataclass as validated_dataclass

from daydream import severity
from daydream.improve.prompts import AUDIT_PLAYBOOK_SECTIONS


class ProfileError(Exception):
    """Invalid profile field (kind) and source, both included in the error message."""

    def __init__(self, kind: str, source: str):
        self.kind = kind
        self.source = source
        super().__init__(f"invalid review profile: {kind} (source: {source})")


# Profiles tune strategy and bounded pipeline fields, never the host evaluator,
# executable behavior, schemas, credentials, or trust/privacy policy.
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

@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Strategy:
    """Stage strategy text and its copied/authored provenance."""

    content: StrictStr = ""
    source: StrictStr = ""


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Arbitration:
    """Bounded arbitration settings."""

    enabled: StrictBool = True
    min_severity: Annotated[
        severity.SeverityLevel, BeforeValidator(lambda value: "high" if value is None else value)
    ] = "high"
    contested_location: StrictBool = True


def _class_sequence(value: Any) -> Any:
    """Profile class selections accept arrays and tuples, never iterable coercion."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("class selection must be an array of strings")
    return value


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Suppression:
    """Precision suppression settings; disabled unless explicitly enabled."""

    enabled: StrictBool = False
    severity_classes: Annotated[tuple[severity.SeverityLevel, ...], BeforeValidator(_class_sequence)] = ("low",)
    confidence_classes: Annotated[
        tuple[Literal["HIGH", "MEDIUM", "LOW"], ...], BeforeValidator(_class_sequence)
    ] = ("LOW",)

    def __post_init__(self) -> None:
        if self.enabled and not self.confidence_classes:
            raise ValueError("suppression: enabled but empty confidence class selection")


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Pipeline:
    """Bounded pipeline policy with production defaults."""

    structural_enabled: StrictBool = True
    arbitration: Arbitration = field(default_factory=Arbitration)
    suppression: Suppression = field(default_factory=Suppression)
    review_wall_budget_s: Annotated[StrictInt, Field(ge=0)] = 2700

    @model_validator(mode="before")
    @classmethod
    def _toml_fields(cls, value: Any) -> Any:
        """The file contract is flat; the admitted owner groups each policy once.

        Dataclass constructors and replace() supply ArgsKwargs and retain their
        nested domain signature. Only file dictionaries use the flat spelling.
        """
        if not isinstance(value, dict):
            return value
        arbitration = ("enabled", "min_severity", "contested_location")
        suppression = ("enabled", "severity_classes", "confidence_classes")
        scalars = ("structural_enabled", "review_wall_budget_s")
        allowed = {
            *scalars, *(f"arbitration_{key}" for key in arbitration), *(f"suppression_{key}" for key in suppression)
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"pipeline: unknown key `{sorted(unknown)[0]}`")
        return {
            **{key: value[key] for key in scalars if key in value},
            "arbitration": {key: value[f"arbitration_{key}"] for key in arbitration if f"arbitration_{key}" in value},
            "suppression": {key: value[f"suppression_{key}"] for key in suppression if f"suppression_{key}" in value},
        }


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ReviewProfile:
    """Versioned per-run strategies and pipeline policy."""

    schema_version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    name: StrictStr = ""
    strategies: dict[str, Strategy] = field(default_factory=dict)
    pipeline: Pipeline = field(default_factory=Pipeline)

    @model_validator(mode="before")
    @classmethod
    def _deny_host_settings(cls, value: Any) -> Any:
        """Host policy denial precedes unknown-field validation at both file scopes."""
        if isinstance(value, dict):
            scopes = [("", value)]
            strategies = value.get("strategies")
            if isinstance(strategies, dict):
                scopes.extend((f"strategies.{key}: ", raw) for key, raw in strategies.items() if isinstance(raw, dict))
            for prefix, raw in scopes:
                host_owned = set(raw) & HOST_OWNED_KEYS
                if host_owned:
                    raise ValueError(f"{prefix}host-owned key `{sorted(host_owned)[0]}` cannot be set by a profile")
        return value

    def to_canonical_dict(self) -> dict[str, object]:
        """Project defaulted semantics, excluding strategy provenance and source formatting."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "strategies": {
                key: strategy.content for key, strategy in sorted(self.strategies.items())
            },
            "pipeline": asdict(self.pipeline),
        }

    @property
    def digest(self) -> str:
        """SHA-256 of sorted-key canonical JSON; provenance and formatting do not affect it."""
        canonical = json.dumps(
            self.to_canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Inline-diff prompts replace the strategy head and retain the judgment tail.
# Custom strategies preserve these markers to support the same splice.
INTENT_STRATEGY_JUDGMENT_MARKER = "That diff is the complete review target"
ALTERNATIVES_STRATEGY_JUDGMENT_MARKER = "Report only concrete problems you can substantiate "

FOLDED_ALTERNATIVES_INSTRUCTION = (
    "Within this same boundary review, check design choices that conflict with the confirmed intent, "
    "an existing canonical implementation, or an applicable repository convention. "
    "Report only a concrete downside supported by repository evidence; do not launch "
    "a separate file-by-file alternatives audit or propose hypothetical replacements."
)


def build_default_profile() -> ReviewProfile:
    """Build packaged strategies with nonempty production text and copied/authored provenance."""

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
                "and read guideline files relevant to the changes below. Report conventions, not defects."
            ),
        ),
        "exploration.dependency_trace": _exploration(
            "daydream.prompts.exploration_subagents.build_dependency_tracer_prompt",
            (
                "You are the **dependency-tracer** specialist. Extend the affected-files\n"
                "list beyond the static-resolved imports by grepping for call sites and\n"
                "reading the implementations. For every import or call edge you confirm,\n"
                "emit a Dependency record. Stop when relevant edges are mapped; do not audit their correctness."
            ),
        ),
        "exploration.test_mapping": _exploration(
            "daydream.prompts.exploration_subagents.build_test_mapper_prompt",
            (
                "You are the **test-mapper** specialist. Locate test files for each modified\n"
                "source file using conventional path mapping (tests/test_X.py, *.test.ts,\n"
                "*_test.go, tests/<crate>_test.rs). Emit a FileInfo with role=\"test\" for\n"
                "each test file you find, and set source_file to the source file it covers. "
                "Map coverage locations; do not conduct a test-quality review."
            ),
        ),
        "intent": Strategy(
            content=(
                "You have full access to explore the codebase. Read the diff file at "
                "{diff_path} and examine the codebase to understand the intent of these changes. "
                f"{INTENT_STRATEGY_JUDGMENT_MARKER}, already computed against the "
                "repository's base branch — this run is not tied to a GitHub pull request, so "
                "do not look up, list, or ask about pull requests. Do not invoke any skills or "
                "slash commands. Use the supplied author context and diff first; "
                "read source only to resolve ambiguity about intent, not to conduct a correctness review. "
                "Present your understanding concisely — what problem is being "
                "solved and how — as plain text in your reply."
            ),
            source="copied: daydream.phases.build_intent_prompt",
        ),
        "alternatives": Strategy(
            content=(
                "The intent of this PR has been confirmed as:\n\n"
                "{intent_summary}\n\n"
                "Given this intent, explore the codebase and evaluate the implementation "
                f"in the diff at {{diff_path}}. {ALTERNATIVES_STRATEGY_JUDGMENT_MARKER}"
                "with evidence. Focus on design choices that conflict with the confirmed intent or "
                "an existing canonical implementation. Stack reviewers handle local correctness; "
                "the structural pass handles cross-module contracts. Do not repeat their full audit. "
                "Consider design decisions that will cause a real "
                "failure, or violations of a Codebase Convention above. Do NOT list stylistic "
                "preferences, speculative 'nice to have' opinions, or alternatives you cannot "
                "tie to a concrete downside.\n\n"
                "Return the required JSON object with an issues array. For each issue, include: a sequential id "
                "number, a brief title, a description of the concrete problem and the evidence "
                "for it, a severity level (high/medium/low), a concrete recommendation for how "
                "to address it, and the relevant file paths.\n\n"
                "If the implementation is solid and you wouldn't change anything, return an empty issues list."
            ),
            source="copied: daydream.phases.build_alternative_review_prompt",
        ),
        "discovery.per_stack": Strategy(
            content=(
                "Review the changed behavior assigned to this stack and its integration "
                "with the rest of the repository. Use the stack name to orient repository "
                "searches, not as permission to apply a memorized framework checklist.\n"
                "\n"
                "Method:\n"
                "1. Read each relevant hunk in every assigned file with its full "
                "enclosing symbol or configuration section. Expand to other sections "
                "only when needed to resolve a concrete candidate. Follow changed callers, "
                "callees, types, configuration, "
                "persistence or network boundaries, error paths, and cleanup or lifecycle "
                "paths as needed.\n"
                "2. Compare the change with repository-local conventions and canonical "
                "helpers before asserting that something is missing, inconsistent, unused, "
                "or duplicated.\n"
                "3. Test each candidate with a concrete triggering input or state and an "
                "observable consequence. Search for evidence that disproves the candidate, "
                "including guards, validation, callers, tests, generated counterparts, and "
                "ownership or lifecycle behavior.\n"
                "4. Prioritize regressions in correctness, data integrity, security or "
                "trust boundaries, concurrency, resource lifetime, external or wire "
                "contracts, configuration flow, and tests that can pass while behavior is "
                "wrong.\n"
                "5. Report only defects introduced or made actionable by this diff. Exclude "
                "style preferences, generic best practices, speculative future "
                "requirements, and pre-existing issues not materially affected by the "
                "change.\n"
                "\n"
                "For every surviving finding, identify the precise file and line, the "
                "triggering path or state, the observable impact, and the smallest safe "
                "remediation. If no candidate survives after the required reads, report "
                "every read assigned file as clean and every unread assigned file as not "
                "reviewed; never invent a finding to fill the review."
            ),
            source="authored: #886 NATIVE_PER_STACK_DISCOVERY_STRATEGY",
        ),
        "discovery.structural": Strategy(
            content=(
                "Review the repository-wide interactions introduced or exposed by this "
                "diff. Concentrate on boundaries that a file-scoped reviewer can miss:\n"
                "\n"
                f"{FOLDED_ALTERNATIVES_INSTRUCTION}\n\n"
                "- incompatible contracts across modules or stacks, including types, "
                "schemas, CLI or API behavior, configuration, serialization, error "
                "semantics, and ownership or lifecycle expectations;\n"
                "- values parsed or accepted at one layer but dropped, re-resolved, "
                "renamed, or applied inconsistently downstream;\n"
                "- partial migrations in which callers, implementations, generated "
                "counterparts, tests, documentation, or compatibility paths no longer "
                "agree;\n"
                "- new dependency-direction or layering violations, duplicated sources of "
                "truth, and bypasses of an existing canonical helper;\n"
                "- partial-failure, rollback, cleanup, cancellation, and resource-lifetime "
                "gaps that emerge only across components;\n"
                "- diff-introduced branching, duplication, or module growth that creates a "
                "concrete correctness or maintenance hazard under an established repository "
                "convention.\n"
                "\n"
                "For each candidate, read both sides of the boundary and trace the relevant "
                "value, call, state transition, or resource lifetime end to end. Do not repeat "
                "the language reviewers’ file-by-file correctness audit. "
                "Stop tracing a boundary when its contract agrees and no concrete candidate remains. Search for "
                "repository evidence that disproves the concern. Verify that any "
                "recommended canonical helper, contract, or layer actually exists and is "
                "compatible before proposing reuse.\n"
                "\n"
                "Report only a concrete risk introduced or made actionable by this change. "
                "Do not report general refactoring wishes, subjective architecture "
                "preferences, arbitrary file-size complaints, or pre-existing design debt "
                "that the diff does not worsen. Every surviving finding must name the "
                "trigger, observable impact, precise evidence locations, and smallest safe "
                "remediation."
            ),
            source="authored: #886 NATIVE_STRUCTURAL_DISCOVERY_STRATEGY",
        ),
        "discovery.generic_fallback": Strategy(
            content=(
                "Review these files for correctness, clarity, and consistency with the "
                "author's intent. Read each relevant hunk in every assigned file with "
                "its full enclosing symbol or configuration section before judging it. "
                "Expand to other sections only to resolve a concrete candidate. Apply "
                "language-agnostic review practices."
            ),
            source="copied: daydream.deep.prompts.build_generic_fallback_prompt",
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
            source="copied: daydream.deep.verification_prompts.build_verification_prompt",
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
            "Treat every audit candidate as an untrusted hypothesis, not as evidence.\n"
            "\n"
            "For each candidate:\n"
            "1. Re-read every cited location and its full enclosing symbol in this turn.\n"
            "2. Search the definitions, callers, configuration, tests, repository "
            "decisions, and related implementations needed to prove or disprove the claim.\n"
            "3. Identify a concrete triggering input, state, or maintenance operation and "
            "its observable impact. A pattern that is merely unusual is not enough.\n"
            "4. Check whether the behavior is intentional, already guarded, unreachable, "
            "generated, externally constrained, or correctly handled at another layer.\n"
            "5. Verify that the proposed remediation fits existing contracts and layer "
            "boundaries. When reuse is proposed, read the claimed reuse target and confirm "
            "behavioral compatibility.\n"
            "6. Collapse duplicates that describe the same underlying problem, correcting "
            "citations and metadata only when the repository supports the correction.\n"
            "\n"
            "Keep a candidate only when current repository evidence establishes the claim "
            "and its impact. Reject candidates that are speculative, stylistic, by design, "
            "unsupported by the cited location, duplicates, or dependent on an unverified "
            "assumption. Never preserve a candidate merely because the audit stated it "
            "confidently."
        ),
        source="authored: #886 NATIVE_IMPROVE_VET_STRATEGY",
    )

    return ReviewProfile(
        schema_version=1,
        name="default",
        strategies=strategies,
        pipeline=Pipeline(),
    )



def parse_profile(toml_text: str, *, source: str = "<string>") -> ReviewProfile:
    """Parse TOML strictly; omitted fields receive defaults before digesting.

    Invalid keys, types, enums, limits or combinations raise ProfileError with the
    source. Failure never falls through to a lower-precedence profile.
    """
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"TOML parse failure: {exc}", source) from exc

    try:
        return TypeAdapter(ReviewProfile).validate_python(data)
    except ValidationError as exc:
        error = exc.errors(include_url=False, include_input=False)[0]
        location = ".".join(str(part) for part in error["loc"])
        raise ProfileError(f"{location}: {error['msg']}", source) from None


@dataclass(frozen=True)
class ResolvedProfile:
    """Profile with source kind (explicit/env/repo/default) and optional source path."""

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


def resolve_pipeline(profile: ResolvedProfile | None) -> Pipeline:
    """Return the resolved pipeline or packaged defaults when unresolved."""
    if profile is not None:
        return profile.profile.pipeline
    return build_default_profile().pipeline


def _read_and_parse(path: Path, source: str) -> ReviewProfile:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileError(
            f"cannot read profile file (reason: {exc})", source
        ) from exc
    return parse_profile(text, source=source)


def _resolved_file(path: str | Path, source_kind: str) -> ResolvedProfile:
    return ResolvedProfile(_read_and_parse(Path(path), str(path)), source_kind, Path(path))


def _guard_repo_path(path: Path, repo_root: Path | None) -> Path:
    """Confine repository-configured paths beneath repo_root (default cwd).

    Resolve user expansion, absolute paths and symlinks before containment checks:
    untrusted repository config must not select an evaluator outside its checkout.
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
    """Resolve CLI path, normal environment, repository config, then packaged default.

    An invalid higher-precedence source raises; only repository paths are confined.
    """
    if explicit_path is not None:
        return _resolved_file(explicit_path, "explicit")

    if env is None:
        env = os.environ
    env_value = env.get("DAYDREAM_REVIEW_PROFILE")
    if env_value:
        return _resolved_file(str(env_value), "env")

    if file_config is not None:
        repo_path = getattr(file_config, "review_profile", None)
        if repo_path is not None:
            guarded = _guard_repo_path(Path(repo_path), repo_root)
            return _resolved_file(guarded, "repo")

    return ResolvedProfile(
        profile=build_default_profile(), source_kind="default"
    )


def resolve_from_runconfig(cfg: object) -> ResolvedProfile:
    """Resolve once using caller policy and the target as the repository path root."""
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
    """Accept only the control-plane candidate environment variable or packaged default.

    Ignore normal-run environment, operator defaults and file_config: a benchmarked
    repository cannot configure its evaluator. Invalid candidates fail closed.
    None for env reads os.environ; candidate_env names the trusted override.
    """
    if env is None:
        env = os.environ
    candidate = env.get(candidate_env)
    if candidate:
        return _resolved_file(str(candidate), "candidate")
    return ResolvedProfile(profile=build_default_profile(), source_kind="default")
