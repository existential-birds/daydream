"""Review a complete source packet with at most one confined evidence batch."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from daydream import clock
from daydream.agent import _validates_schema, run_agent
from daydream.backends import Backend
from daydream.backends.pi import PiBackend
from daydream.deep.prompts import (
    ANTI_SLOP_RUBRIC_INSTRUCTION,
    CONFIG_FLOW_TRACE_INSTRUCTION,
    TEST_QUALITY_RUBRIC_INSTRUCTION,
    TRUST_MODEL_INSTRUCTION,
    VERIFICATION_PROTOCOL_INSTRUCTION,
    build_generic_fallback_prompt,
    build_per_stack_prompt,
)
from daydream.extensions import get_registry
from daydream.output_schema import strict_object
from daydream.prompt_budget import PreparedSanctionedInputs
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_BLOCK
from daydream.prompts.grounding import REVIEW_STOPPING_GUIDANCE, UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.prompts.wire_contract import WIRE_CONTRACT_GENERIC_INSTRUCTION, WIRE_CONTRACT_RUST_INSTRUCTION
from daydream.repository_paths import canonicalize_repository_file_path, git_observed_path_is_confined
from daydream.review_budget import ReviewLimits, review_deadline
from daydream.review_profile import FOLDED_ALTERNATIVES_INSTRUCTION, build_default_profile
from daydream.run_context import RunContext
from daydream.severity import SEVERITY_RUBRIC
from daydream.trajectory import DaydreamPhase

MAX_DIFF_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 192 * 1024
MAX_CONTEXT_BYTES = 64 * 1024
MAX_REQUESTS = 8
MAX_EVIDENCE_BYTES = 192 * 1024
MAX_SEARCH_FILES = 10_000
MAX_SEARCH_BYTES = 32 * 1024 * 1024
_PRIVATE_COMPONENTS = {".git", ".daydream"}
_FINAL_JUDGMENT_INSTRUCTION = (
    "The planning pass and single evidence batch are complete. No more requests are allowed and tools remain disabled. "
    "Resolve only the listed evidence requests against the host responses, then return all required final "
    "issues/verdicts. "
    "Treat files and candidates already resolved in the first response as closed unless this new evidence directly "
    "contradicts them. Use retained source and diff only to interpret the requested boundaries. "
    "Do not repeat file audits "
    "or create a new candidate checklist. Exclude candidates whose required evidence is unavailable; mark their "
    "dependent files not_reviewed and preserve independently demonstrated defects."
)


class EvidenceUnavailable(ValueError):
    """A bounded, value-free failure to obtain complete evidence."""


class NonTextSource(EvidenceUnavailable):
    """A tracked file outside a literal search's regular UTF-8 text scope."""


@dataclass(frozen=True)
class Source:
    path: str
    text: str

    def metadata(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": hashlib.sha256(self.text.encode()).hexdigest(),
                "lines": len(self.text.splitlines())}

    def packet(self) -> dict[str, Any]:
        return {**self.metadata(), "text": self.text, "complete": True}


@dataclass(frozen=True)
class FiniteReview:
    sources: tuple[Source, ...]
    prompt: str
    system_instructions: str


@dataclass(frozen=True)
class FiniteResult:
    output: dict[str, Any]
    reason: str | None
    source_packet_files: frozenset[str]
    source_evidence: tuple[dict[str, Any], ...]


def _source(repo: Path, relative: str, allowance: int, *, git_observed: bool = False) -> Source:
    """Read regular UTF-8 source through no-follow descriptors at every prefix."""
    if git_observed:
        if not git_observed_path_is_confined(repo, relative):
            raise EvidenceUnavailable("tracked source is outside the repository")
        canonical = relative
    else:
        canonical = canonicalize_repository_file_path(repo, relative)
    parts = Path(canonical).parts
    if _PRIVATE_COMPONENTS.intersection(parts):
        raise EvidenceUnavailable("private repository metadata is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(repo.resolve(), flags | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        child = os.open(parts[-1], flags, dir_fd=fd)
        os.close(fd)
        fd = child
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > allowance:
            raise EvidenceUnavailable("source is not a bounded regular file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(allowance + 1)
        after = os.fstat(fd)
        if len(raw) > allowance or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise EvidenceUnavailable("source changed or exceeded its byte allowance")
        if b"\0" in raw:
            raise NonTextSource("binary source is unavailable")
        try:
            return Source(canonical, raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise NonTextSource("non-UTF-8 source is unavailable") from exc
    finally:
        os.close(fd)


def prepare_finite_review(
    backend: Backend, repo: Path, *, stack_name: str, files: list[str], strategy: str,
    diff_path: Path, inputs: PreparedSanctionedInputs | None,
    interactive: bool, intent_authoritative: bool = False, prior_commits: str | None = None,
) -> FiniteReview | None:
    """Return a complete bounded packet, or leave the existing reviewer in charge."""
    role = "discovery.generic_fallback" if stack_name == "generic" else "discovery.per_stack"
    builder = get_registry().prompt("generic-fallback" if stack_name == "generic" else "per-stack")
    default_builder = build_generic_fallback_prompt if stack_name == "generic" else build_per_stack_prompt
    if (interactive or not isinstance(backend, PiBackend)
            or not getattr(backend, "supports_tools_disabled", False) or stack_name == "structure"
            or not 0 < len(files) <= 10 or len(set(files)) != len(files)
            or strategy != build_default_profile().strategies[role].content or inputs is None
            or builder is not default_builder):
        return None
    try:
        inputs.revalidate(backend, repo, False)
        with diff_path.open("rb") as stream:
            raw_diff = stream.read(MAX_DIFF_BYTES + 1)
        if len(raw_diff) > MAX_DIFF_BYTES or b"\0" in raw_diff:
            return None
        diff = raw_diff.decode("utf-8")
        sources: list[Source] = []
        remaining = MAX_SOURCE_BYTES
        for path in files:
            source = _source(repo, path, remaining, git_observed=True)
            if source.path != path:
                return None
            remaining -= len(source.text.encode())
            sources.append(source)
        context: dict[str, str] = {}
        remaining = MAX_CONTEXT_BYTES
        for item in inputs.inputs:
            if item.label == "diff":
                continue
            if item.size > remaining:
                return None
            text = item.text if item.text is not None else item.path.read_text(encoding="utf-8")
            raw = text.encode("utf-8")
            if len(raw) > remaining or hashlib.sha256(raw).hexdigest() != item.sha256:
                return None
            remaining -= len(raw)
            context[item.label] = text
        if "intent" not in context:
            return None
        if len((prior_commits or "").encode()) > remaining:
            return None
        inputs.revalidate(backend, repo, False)
    except (OSError, ValueError):
        return None
    packet = {"assigned_files": files, "sources": [source.packet() for source in sources],
              "diff": diff, "context": context, "prior_commits": prior_commits or ""}
    policy = "\n\n".join([
        UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY, strategy, VERIFICATION_PROTOCOL_INSTRUCTION,
        TEST_QUALITY_RUBRIC_INSTRUCTION, ANTI_SLOP_RUBRIC_INSTRUCTION, CONFIG_FLOW_TRACE_INSTRUCTION,
        SEVERITY_RUBRIC, TRUST_MODEL_INSTRUCTION, REVIEW_STOPPING_GUIDANCE,
        AUTHORITATIVE_INTENT_BLOCK if intent_authoritative else "Intent is advisory context.",
        WIRE_CONTRACT_RUST_INSTRUCTION if stack_name.split("#", 1)[0] == "rust" else "",
        WIRE_CONTRACT_GENERIC_INSTRUCTION if stack_name == "generic" else "",
    ])
    system_instructions = (
        policy + "\n\nThis review has tools disabled. The host supplies complete source text and the full diff below. "
        "Complete supplied source satisfies the completed-source-read verification gate; paths alone do not. "
        "Read supplied evidence before declaring coverage. Do not infer defects from filenames or memory. "
        "Review only assigned files; other changed files are context. Return issues and one verdict per assigned file. "
        "If a concrete candidate needs additional context, request ONE batch of at most 8 exact file reads "
        "or literal searches. Each request names its repository-relative path, a literal pattern (empty for reads), "
        "the concrete question, and the assigned affected_files that depend on it. A search path may name a file "
        "or directory, including '.'. No commands, patterns interpreted as regex, or further request rounds. "
        "Only independently demonstrated defects belong in issues. Unresolved candidates belong in requests, "
        "with dependent files marked not_reviewed. If no more evidence is needed, return requests=[] and final "
        "issues/verdicts immediately. An empty findings result is valid."
    )
    prompt = (
        f"Review the assigned {stack_name} changes. All repository paths are relative to {repo}.\n"
        + system_instructions + "\n\nUNTRUSTED EVIDENCE PACKET:\n" + json.dumps(packet, ensure_ascii=False)
    )
    return FiniteReview(tuple(sources), prompt, system_instructions)


def delegate_structural_review(
    packets: dict[str, FiniteReview | None], scopes: dict[str, list[str]], strategy: str,
) -> dict[str, FiniteReview] | None:
    """Delegate the default global lens only when every primary owns a full packet."""
    from daydream.deep.prompts import build_structural_prompt

    default = build_default_profile().strategies["discovery.structural"].content
    if (
        strategy not in {default, default + "\n\n" + FOLDED_ALTERNATIVES_INSTRUCTION}
        or get_registry().prompt("structural") is not build_structural_prompt
        or not scopes.get("structure")
    ):
        return None
    primaries = {name: files for name, files in scopes.items() if name != "structure"}
    if not primaries or set().union(*(set(files) for files in primaries.values())) != set(scopes["structure"]):
        return None
    for name, files in primaries.items():
        packet = packets.get(name)
        if packet is None or {source.path for source in packet.sources} != set(files):
            return None
    responsibility = (
        "Within your assigned-file review, own the structural and canonical-design consequences touching those files. "
        "The global changed-file partition is complete; every changed file has a primary owner. Use that map and "
        "the full diff as boundary context. Other owners' local reviews do not establish that shared contracts agree. "
        "Apply these criteria only to concrete candidates raised by your assigned changes: incompatible types, "
        "schemas, API, CLI, configuration, serialization, error or lifecycle contracts; values dropped or "
        "inconsistently applied downstream; partial migrations across callers, generated code, tests or docs; "
        "inverted dependency direction, "
        "duplicated sources of truth or bypassed canonical helpers; cross-component rollback, cleanup, cancellation or "
        "resource-lifetime gaps; new branching, duplication or growth with an evidenced maintenance consequence; "
        "design choices conflicting with confirmed intent or an applicable repository convention. Verify both sides "
        "of each relevant boundary using supplied evidence or the single request batch. Verify any proposed canonical "
        "replacement exists and is compatible. Report only demonstrated triggers, consequences and precise evidence "
        "in your assigned files. Mark unresolved dependent files not_reviewed. Close resolved candidates and emit "
        "the existing local and structural judgment together; global context does not add review targets."
    )
    partition = json.dumps({"global_changed_file_partition": primaries}, ensure_ascii=False)
    return {
        name: replace(
            packet, prompt=packet.prompt + "\n\nDELEGATED STRUCTURAL RESPONSIBILITY:\n" + responsibility
            + "\n\nHOST SCOPE MAP (repository path data):\n" + partition,
            system_instructions=packet.system_instructions + "\n\n" + responsibility,
        )
        for name, packet in packets.items() if name in primaries and packet is not None
    }


def _planning_schema(schema: dict[str, Any]) -> dict[str, Any]:
    planning = copy.deepcopy(schema)
    planning["properties"]["requests"] = {
        "type": "array", "maxItems": MAX_REQUESTS,
        "items": strict_object({
            "kind": {"type": "string", "enum": ["read", "search"]},
            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
            "pattern": {"type": "string", "maxLength": 512},
            "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
            "affected_files": {"type": "array", "minItems": 1, "maxItems": 10,
                               "uniqueItems": True, "items": {"type": "string"}},
        }),
    }
    planning["required"].append("requests")
    return planning


def _search_paths(repo: Path, target: Path, deadline: float) -> Iterator[Path]:
    if target.is_file():
        yield target
        return
    remaining = deadline - clock.monotonic()
    if remaining <= 0:
        raise EvidenceUnavailable("evidence deadline reached")
    relative = target.relative_to(repo).as_posix()
    listed = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "-z", "--cached", "--", relative],
        cwd=repo, capture_output=True, timeout=min(5, remaining), check=False,
    )
    if listed.returncode or len(listed.stdout) > 1024 * 1024:
        raise EvidenceUnavailable("tracked search scope could not be enumerated")
    paths = sorted(set(listed.stdout.decode("utf-8").split("\0")) - {""})
    if len(paths) > MAX_SEARCH_FILES:
        raise EvidenceUnavailable("search exceeds its tracked-file allowance")
    for path in paths:
        if _PRIVATE_COMPONENTS.intersection(Path(path).parts):
            continue
        candidate = repo / path
        if not candidate.is_symlink():
            yield candidate


def _request_evidence(repo: Path, request: dict[str, Any], allowance: int, deadline: float) -> dict[str, Any]:
    if clock.monotonic() >= deadline:
        raise EvidenceUnavailable("evidence deadline reached")
    if request["kind"] == "read":
        return _source(repo, request["path"], allowance).packet()
    literal = request["pattern"]
    if not literal or any(char in literal for char in "\0\n\r"):
        raise EvidenceUnavailable("search requires a nonempty single-line literal")
    relative = request["path"]
    canonical = "." if relative == "." else canonicalize_repository_file_path(repo, relative)
    if _PRIVATE_COMPONENTS.intersection(Path(canonical).parts):
        raise EvidenceUnavailable("private repository metadata is unavailable")
    target = repo / canonical
    if not target.exists():
        raise EvidenceUnavailable("search scope is unavailable")
    count = total = emitted = 0
    matches: list[dict[str, Any]] = []
    for candidate in _search_paths(repo, target, deadline):
        relative_path = candidate.relative_to(repo).as_posix()
        if _PRIVATE_COMPONENTS.intersection(Path(relative_path).parts):
            continue
        if candidate.is_symlink():
            raise EvidenceUnavailable("search scope contains a symlink")
        if candidate.is_dir():
            continue
        count += 1
        if count > MAX_SEARCH_FILES or clock.monotonic() >= deadline:
            raise EvidenceUnavailable("search exceeds its file or time allowance")
        try:
            source = _source(repo, relative_path, MAX_SEARCH_BYTES - total, git_observed=True)
        except NonTextSource:
            total += candidate.stat().st_size
            continue
        total += len(source.text.encode())
        for number, line in enumerate(source.text.splitlines(), 1):
            if literal in line:
                match = {"path": relative_path, "line": number, "text": line}
                emitted += len(json.dumps(match, ensure_ascii=False).encode())
                if emitted > allowance:
                    raise EvidenceUnavailable("search result exceeds its byte allowance")
                matches.append(match)
    return {"path": canonical, "literal": literal, "matches": matches, "complete": True,
            "scope": "tracked regular UTF-8 files under the requested directory (or the exact requested file); "
                     "untracked, binary, symlink, .git and .daydream content is excluded"}


def _finish(
    review: FiniteReview, output: dict[str, Any], incomplete: set[str], reason: str | None,
) -> FiniteResult:
    assigned = {source.path for source in review.sources}
    issues = [dict(item) for item in output.get("issues", []) if item.get("file") in assigned]
    verdicts: list[dict[str, Any]] = []
    complete: set[str] = set()
    for source in review.sources:
        returned = [v for v in output.get("verdicts", []) if v.get("path") == source.path]
        n_findings = sum(issue["file"] == source.path for issue in issues)
        if (source.path not in incomplete and len(returned) == 1
                and returned[0]["verdict"] in {"clean", "has_findings"}):
            complete.add(source.path)
            verdict = "has_findings" if n_findings else "clean"
        else:
            verdict = "not_reviewed"
            incomplete.add(source.path)
        lines_read = len(source.text.splitlines()) if verdict != "not_reviewed" else 0
        verdicts.append({"path": source.path, "lines_read": lines_read,
                         "verdict": verdict, "n_findings": n_findings})
    return FiniteResult(
        {"issues": issues, "verdicts": verdicts}, reason or ("evidence_incomplete" if incomplete else None),
        frozenset(complete) if reason is None else frozenset(), tuple(source.metadata() for source in review.sources),
    )


async def run_finite_review(
    backend: Backend, repo: Path, review: FiniteReview, *, schema: dict[str, Any], run_context: RunContext,
) -> FiniteResult:
    """One planning call, one bounded host batch, and at most one final call."""
    limits = ReviewLimits()
    started = clock.monotonic()
    hard_deadline = started + limits.investigation_s + limits.finalization_s
    shared = review_deadline(discovery=True)
    if shared is not None:
        hard_deadline = min(hard_deadline, shared)
    investigation_deadline = max(started, min(started + limits.investigation_s, hard_deadline - limits.finalization_s))
    planning = _planning_schema(schema)
    first, _, reason = await run_agent(
        backend, repo, review.prompt, phase=DaydreamPhase.DEEP, output_schema=planning,
        tools_disabled=True, tool_call_budget=0, deadline=investigation_deadline, run_context=run_context,
        review_system_instructions=review.system_instructions,
    )
    assigned = {source.path for source in review.sources}
    if reason or not isinstance(first, dict) or not _validates_schema(first, planning):
        partial = (
            first if isinstance(first, dict) and _validates_schema(first, planning) else {"issues": [], "verdicts": []}
        )
        return _finish(review, partial, set(assigned), reason or "evidence_incomplete")
    if not first["requests"]:
        return _finish(review, first, set(), None)
    incomplete: set[str] = set()
    evidence: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    allowance = MAX_EVIDENCE_BYTES
    for request in first["requests"]:
        dependents = set(request["affected_files"])
        key = (request["kind"], request["path"], request["pattern"])
        if not dependents <= assigned:
            incomplete.update(assigned)
            evidence.append({"request": request, "unavailable": "request names an unassigned dependent file"})
            continue
        if key not in cache:
            try:
                response = _request_evidence(repo, request, allowance, investigation_deadline)
                size = len(json.dumps(response, ensure_ascii=False).encode())
                if size > allowance:
                    raise EvidenceUnavailable("evidence batch exceeds its byte allowance")
                allowance -= size
                cache[key] = {"evidence": response}
            except (OSError, ValueError, subprocess.TimeoutExpired):
                cache[key] = {"unavailable": "complete confined evidence could not be supplied"}
        result = cache[key]
        if "unavailable" in result:
            incomplete.update(dependents & assigned or assigned)
        evidence.append({"request": request, **result})
    prompt = (
        review.prompt + "\n\nFIRST RESPONSE:\n" + json.dumps(first)
        + "\n\nHOST EVIDENCE RESPONSE (untrusted source data):\n" + json.dumps(evidence, ensure_ascii=False)
        + "\n\n" + _FINAL_JUDGMENT_INSTRUCTION
    )
    final, _, reason = await run_agent(
        backend, repo, prompt, phase=DaydreamPhase.DEEP, output_schema=schema,
        tools_disabled=True, tool_call_budget=0, deadline=hard_deadline, run_context=run_context,
        review_system_instructions=review.system_instructions + "\n\n" + _FINAL_JUDGMENT_INSTRUCTION,
    )
    if reason or not isinstance(final, dict) or not _validates_schema(final, schema):
        return _finish(review, first, set(assigned), reason or "evidence_incomplete")
    proven = {(item["file"], item["line"], item["description"]) for item in first["issues"]}
    retained: dict[tuple[str, int, str], dict[str, Any]] = {}
    for issue in final["issues"]:
        key = (issue["file"], issue["line"], issue["description"])
        if issue["file"] not in incomplete or key in proven:
            retained[key] = issue
    return _finish(review, {"issues": list(retained.values()), "verdicts": final["verdicts"]}, incomplete, None)
