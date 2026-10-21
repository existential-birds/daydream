"""Native review execution and conservative ReviewBench finding conversion.

The native runner remains responsible for backend/configuration/authentication and
review completeness. A benchmark label never supplies review success or source
coordinates that the finalized native result did not establish.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from daydream import git_ops, runner
from daydream.backends import BackendExecutionInput, resolve_fanout_concurrency
from daydream.backends._subprocess import stream_idle_timeout_s
from daydream.backends.pi import native_pi_settings
from daydream.benchmark.reviewbench.contract import REVIEWBENCH_REVISION, Case
from daydream.deep.artifacts import diff_key
from daydream.git_ops.process import _run_git
from daydream.git_ops.source import frozen_source
from daydream.github_app import GitHubExecutionInput
from daydream.hunk_index import _HUNK_HEADER, _header_path, _unquote_git_path, parse_hunks
from daydream.redaction import redact_text
from daydream.review_result import REVIEW_CONFIGURATION_PHASES, FinalizedReview, validate_terminal_result
from daydream.run_config import FrozenAuthorDescription, RunConfig, _resolved_backend_name, _resolved_reasoning_effort
from daydream.trajectory import RUN_DOCUMENT_NAME


def native_configuration(config: RunConfig) -> tuple[dict[str, Any], runner.RunnerExecutionInput]:
    """Freeze nonsecret native controls alongside an unpersisted auth capability.

    Only selected public Pi defaults are hashed; neither authentication files nor
    arbitrary settings (which may contain credentials) are copied or fingerprinted.
    """
    root = Path(config.target or Path.cwd()).resolve()
    phases = set(REVIEW_CONFIGURATION_PHASES) | set(config.file_config.phases if config.file_config else {})
    names = {_resolved_backend_name(config, phase) for phase in phases}
    inputs: dict[str, BackendExecutionInput | None] = {}
    provenance: dict[str, Any] = {}
    for name in sorted(names):
        selector_names = {
            'pi': ('PI_CODING_AGENT_DIR',),
            'codex': ('CODEX_HOME', 'OPENAI_BASE_URL'),
            'claude': ('CLAUDE_CONFIG_DIR', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK',
                       'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', 'AWS_REGION',
                       'ANTHROPIC_DEFAULT_SONNET_MODEL', 'ANTHROPIC_DEFAULT_OPUS_MODEL',
                       'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'ANTHROPIC_SMALL_FAST_MODEL'),
            'osprey': ('OSPREY_HOME', 'OSPREY_BINARY'),
        }[name]
        selectors: dict[str, Any] = {}
        for key in selector_names:
            value = os.environ.get(key)
            if value and key.endswith('_BASE_URL'):
                url = urlsplit(value)
                selectors[key] = {'scheme': url.scheme, 'host': url.hostname, 'port': url.port,
                                  'path': redact_text(url.path)}
            else:
                selectors[key] = value
        home = Path(os.environ.get('HOME', str(Path.home())))
        if name == 'codex':
            selectors['CODEX_HOME'] = selectors['CODEX_HOME'] or str(home / '.codex')
        elif name == 'claude':
            selectors['CLAUDE_CONFIG_DIR'] = selectors['CLAUDE_CONFIG_DIR'] or str(home / '.claude')
        if name == "osprey":
            inputs[name] = None  # Osprey retains its supported ambient transport.
            provenance[name] = {"binary": os.environ.get("OSPREY_BINARY", "osprey"), 'selectors': selectors,
                'fanout_concurrency': resolve_fanout_concurrency('DAYDREAM_OSPREY_FANOUT_CONCURRENCY', 4),
                'stream_idle_timeout_s': stream_idle_timeout_s(),
                'limitations': ['Native client configuration files and internal provider choices remain opaque.']}
            continue
        environment = dict(os.environ)
        if name == 'pi' and environment.get('PI_CODING_AGENT_DIR'):
            environment['PI_CODING_AGENT_DIR'] = str(Path(environment['PI_CODING_AGENT_DIR']).resolve())
        native = BackendExecutionInput.from_environment(environment, backend=name)
        selected = {}
        if name == "pi":
            selected = native_pi_settings(root, agent_dir=native.pi_agent_dir)
            native = replace(native, pi_settings_root=root, pi_settings=selected)
        inputs[name] = native
        public = {"retry_policy": asdict(native.retry_policy), "fanout_concurrency": native.fanout_concurrency,
            'selectors': selectors,
            'limitations': ['Native client configuration files are retained without credential-bearing fingerprints.'],
            "stream_idle_timeout_s": native.stream_idle_timeout_s,
            "response_idle_timeout_s": native.pi_response_idle_timeout_s,
            "terminal_drain_timeout_s": native.codex_terminal_drain_timeout_s,
            "process_reap_timeout_s": native.codex_process_reap_timeout_s,
            "cleanup_timeout_s": native.codex_cleanup_timeout_s}
        if name == "pi":
            selectors['PI_CODING_AGENT_DIR'] = None if native.pi_agent_dir is None else str(native.pi_agent_dir)
            public.update(provider=native.pi_provider, thinking=native.pi_thinking, defaults=selected,
                defaults_sha256=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest())
        provenance[name] = public
    fallback = next((value for value in inputs.values() if value is not None), None)
    if fallback is None:
        fallback = BackendExecutionInput.from_environment(dict(os.environ), backend="claude")
    return provenance, runner.RunnerExecutionInput(fallback, GitHubExecutionInput(), backends=inputs)


@dataclass(frozen=True)
class PublicHarborTask:
    """Host-admitted public task identity, separate from standalone origin admission.

    Harbor constructs snapshots without origin. Its frozen task/profile supplies
    the identity; actual refs, HEAD and full patch still have to prove the input.
    This capability is accepted only for origin-free public frozen snapshots.
    """

    case: Case
    upstream_revision: str
    patch: str
    task_id: int
    evaluation_id: int
    profile_sha256: str
    author_description: str | None = None

    def verify(self, repository: Path, case: Case) -> str:
        if (self.case != case or self.upstream_revision != REVIEWBENCH_REVISION
                or type(self.task_id) is not int or self.task_id < 0
                or type(self.evaluation_id) is not int or self.evaluation_id < 0
                or re.fullmatch(r"[0-9a-f]{64}", self.profile_sha256) is None):
            raise ValueError("public Harbor task identity is invalid")
        if git_ops.remote_urls(repository):
            raise ValueError("public Harbor frozen repository must have no remotes")
        base_ref = _run_git(repository, ['rev-parse', '--verify', 'refs/reviewbench/base'], retries=0).stdout.strip()
        head_ref = _run_git(repository, ['rev-parse', '--verify', 'refs/reviewbench/head'], retries=0).stdout.strip()
        if (git_ops.head_sha(repository) != case.head
                or base_ref != case.base
                or head_ref != case.head):
            raise ValueError("public Harbor frozen refs differ from supplied task")
        merge_base = git_ops.merge_base(repository, case.base, case.head)
        if merge_base is None or git_ops.diff_paths(repository, merge_base, case.head, ['.']) != self.patch:
            raise ValueError("public Harbor patch differs from supplied frozen comparison")
        return case.repo


@dataclass
class CaseExecution:
    """One attempt, with partial evidence kept separate from scorable findings."""

    status: str
    candidate: dict[str, Any] | None = None
    terminal_result: dict[str, Any] | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    retained_findings: list[dict[str, Any]] = field(default_factory=list)
    conversion_omissions: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    source_identity: str | None = None
    analyzed_comparison: dict[str, str] = field(default_factory=dict)


def _omission(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"uid": item.get("item_uid", item.get("uid")), "id": item.get("id"), "reason": reason}


def _path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        return None
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or value != parsed.as_posix() or any(part in {".", ".."} for part in value.split("/")):
        return None
    return value


def _diff_coordinates(comparison: str) -> tuple[dict[str, str], dict[str, set[int]]]:
    """Resolve original paths and context whose old and new coordinates agree."""
    original_paths: dict[str, str] = {}
    result: dict[str, set[int]] = {}
    path: str | None = None
    old_path: str | None = None
    old = new = 0
    in_hunk = False
    for raw in comparison.split("\n"):
        if raw.startswith("diff --git "):
            path, old_path, in_hunk = None, None, False
        elif raw.startswith('--- ') and not in_hunk:
            old_path = _unquote_git_path(raw[4:].rstrip('\t')).removeprefix('a/')
        elif raw.startswith("+++ ") and not in_hunk:
            path, in_hunk = _header_path(raw), False
            if path is not None and old_path not in {None, '/dev/null'}:
                original_paths[path] = old_path
        elif (header := _HUNK_HEADER.match(raw)) is not None:
            old, new, in_hunk = int(header[1]), int(header[3]), True
        elif path is not None and in_hunk:
            if raw.startswith(" "):
                if old == new:
                    result.setdefault(path, set()).add(new)
                old, new = old + 1, new + 1
            elif raw.startswith("-"):
                old += 1
            elif raw.startswith("+"):
                new += 1
    return original_paths, result


def convert_findings(
    items: list[dict[str, Any]], *, repository: Path, case: Case, producer: str, comparison: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project only hints whose captured diff proves an unambiguous HEAD location.

    Native canonical findings currently carry a line hint, without a source side.
    A number present on both sides is safe only for unchanged context with
    identical coordinates. Replacements, deleted-side citations, whole-file findings, and native
    location snapping remain visible omissions rather than invented locations.
    """
    hunks = parse_hunks(comparison)
    original_paths, same_context = _diff_coordinates(comparison)
    findings: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    sources: dict[tuple[str, str], list[bytes] | None] = {}
    absent: set[tuple[str, str]] = set()

    def source(revision: str, path: str) -> list[bytes] | None:
        key = revision, path
        if key not in sources:
            try:
                _, body = frozen_source(repository, revision, path)
                sources[key] = (body.split(b"\n")[:-1] if body.endswith(b"\n")
                                else body.split(b"\n") if body else [])
            except git_ops.PathAbsentError:
                sources[key] = None
                absent.add(key)
            except (git_ops.GitError, OSError, UnicodeError, ValueError):
                sources[key] = None
        return sources[key]

    merge_base = git_ops.merge_base(repository, case.base, case.head)
    if merge_base is None:
        raise ValueError("supplied comparison has no merge base")
    for item in items:
        path, line = _path(item.get("file")), item.get("line")
        reason: str | None = None
        if path is None or type(line) is not int or line <= 0:
            reason = "missing_repository_relative_positive_location"
        elif "location_cited_line" in item or item.get("location_distrust"):
            reason = "native_location_was_relocated_or_distrusted"
        else:
            head_lines = source(case.head, path)
            ranges = hunks.get(path, {}).get("hunks", [])
            new_side = any(h["new_start"] <= line <= h["new_end"] for h in ranges)
            old_path = original_paths.get(path, path)
            old_lines = source(merge_base, old_path)
            if head_lines is None or line > len(head_lines):
                reason = "location_absent_from_head_regular_file"
            elif not new_side:
                reason = "no_new_side_location_in_captured_diff"
            elif line not in same_context.get(path, set()):
                # A bare hint may cite any old source line, not only one visible
                # in an old hunk. Only verified absence or actual line count can
                # establish that this coordinate exists exclusively at HEAD.
                if old_lines is None and (merge_base, old_path) not in absent:
                    reason = "old_source_location_unavailable"
                elif old_lines is not None and line <= len(old_lines):
                    reason = "ambiguous_old_or_new_side_line_hint"
        if reason is not None:
            omissions.append(_omission(item, reason))
            continue
        message = "\n\n".join(value.strip() for key in ("description", "rationale", "evidence")
                              if isinstance(value := item.get(key), str) and value.strip())
        if not message:
            omissions.append(_omission(item, "missing_finding_message"))
            continue
        findings.append({"producer": producer, "file": path, "start_line": line, "end_line": line,
                         "message": message})
    return findings, omissions


def _source_identity(repository: Path, case: Case) -> str:
    expected = urlsplit(case.repo).path.strip("/").removesuffix(".git")
    owner_repo = git_ops.split_owner_repo(expected)
    if owner_repo is None:
        raise ValueError("supplied repository identity is invalid")
    identity = git_ops.resolve_github_repository(repository, base_environment={})
    mirror = ("review-bench", "_".join(owner_repo))
    if identity is None or tuple(part.casefold() for part in identity) not in {
        tuple(part.casefold() for part in owner_repo), tuple(part.casefold() for part in mirror),
    }:
        raise ValueError("repository origin does not match supplied upstream or frozen mirror identity")
    return "https://github.com/" + "/".join(identity)


async def review_case(case: Case, repository: Path, config: RunConfig, producer: str, *,
                      runtime: tuple[dict[str, Any], runner.RunnerExecutionInput] | None = None,
                      public_task: PublicHarborTask | None = None,
                      evidence_directory: Path | None = None) -> CaseExecution:
    """Review one clean local frozen repository in a disposable independent checkout.

    No GitHub publication target is manufactured. The callback supplies actual
    terminal coverage even when the runner raises after admitting partial work.
    """
    execution = CaseExecution("failed")
    finalized: list[FinalizedReview] = []
    try:
        public_native, native_execution = native_configuration(config) if runtime is None else runtime
        if not producer.strip():
            raise ValueError("producer must be nonempty")
        if any(re.fullmatch(r"[0-9a-f]{40}", oid) is None for oid in (case.base, case.head)):
            raise ValueError("supplied base and head must be full commit identities")
        git_ops.assert_is_worktree(repository)
        if git_ops.status_porcelain(repository).strip():
            raise ValueError("source repository is not clean")
        execution.source_identity = (_source_identity(repository, case) if public_task is None
                                     else public_task.verify(repository, case))
        trajectory = None
        if evidence_directory is not None:
            evidence_directory.mkdir(parents=True, exist_ok=False)
            trajectory = evidence_directory / RUN_DOCUMENT_NAME
        if not all(git_ops.commit_exists(repository, oid) for oid in (case.base, case.head)):
            raise ValueError("supplied base or head is unavailable")
        with tempfile.TemporaryDirectory(prefix="daydream-reviewbench-case-") as directory:
            checkout = Path(directory) / "repo"
            checkout.mkdir()
            git_ops.init_repository(checkout)
            # A SHA-only frozen cache can retain a divergent base tip without
            # advertising it under any ref. Cloning just HEAD loses that base.
            # Fetch exactly both local endpoints into independent object storage;
            # unrelated branches and their later fixes are not reviewer inputs.
            _run_git(checkout, ["fetch", "--no-tags", str(repository.resolve()), case.base, case.head],
                     retries=0, timeout=300, error_context="frozen local endpoint materialization failed")
            git_ops.checkout_detach(checkout, case.head)
            # No remote is installed: this input cannot prefer another base or
            # lazily fetch later repository history during review.
            if git_ops.head_sha(checkout) != case.head or git_ops.status_porcelain(checkout).strip():
                raise ValueError("isolated checkout does not match supplied clean head")
            pi = native_execution.backends.get('pi')
            phases = set(REVIEW_CONFIGURATION_PHASES) | set(config.file_config.phases if config.file_config else {})
            if (pi is not None and pi.pi_settings is not None and pi.pi_thinking is None
                    and 'defaultThinkingLevel' not in pi.pi_settings
                    and native_pi_settings(checkout, agent_dir=None).get('defaultThinkingLevel') is not None
                    and any(_resolved_backend_name(config, phase) == 'pi'
                            and _resolved_reasoning_effort(config, phase) is None for phase in phases)):
                raise ValueError('Case Pi thinking conflicts with unavailable operator effort')
            merge_base = git_ops.merge_base(checkout, case.base, case.head)
            if merge_base is None:
                raise ValueError("supplied comparison has no merge base")
            comparison = git_ops.diff_paths(checkout, merge_base, case.head, ["."])
            expected = {"head_sha": case.head, "merge_base_sha": merge_base, "diff_key": diff_key(comparison)}
            execution.analyzed_comparison = expected
            native = replace(config, target=str(checkout), base=case.base, branch=None, force_worktree=False,
                output_mode="review", flow_name="review", pr_number=None, pr_repo=None,
                approved_head_sha=case.head, discover_github_provenance=False, findings_out=None,
                frozen_author_description=(None if public_task is None or public_task.author_description is None
                    else FrozenAuthorDescription(case.head, public_task.author_description)),
                dataset_capture=False, dataset_capture_disabled=True, trajectory_hub_repo=None,
                archive=False, run_eval=False, trajectory_path=trajectory, dump_artifacts=None, non_interactive=True)
            exit_code: int | None = None
            try:
                exit_code = await runner.run(native, review_result_handler=finalized.append, execution=native_execution)
            except Exception as exc:
                execution.diagnostics.append(f"native reviewer raised {type(exc).__name__}")
            if not finalized:
                execution.diagnostics.append("native reviewer did not produce a finalized terminal result")
                return execution
            if len(finalized) != 1:
                raise ValueError("native reviewer produced multiple terminal results")
            result = finalized[0]
            execution.terminal_result = copy.deepcopy(result.terminal_result)
            execution.configuration = copy.deepcopy(result.configuration)
            execution.usage = copy.deepcopy(result.usage)
            execution.configuration['native'] = copy.deepcopy(public_native)
            execution.retained_findings = copy.deepcopy(result.items)
            validate_terminal_result(execution.terminal_result, expected_head_sha=case.head)
            revision = execution.terminal_result["analyzed_revision"]
            if any(revision.get(key) != value for key, value in expected.items()):
                raise ValueError("native analyzed comparison differs from supplied frozen comparison")
            if git_ops.head_sha(checkout) != case.head or git_ops.diff_worktree_against(checkout, case.head, ["."]):
                raise ValueError("native reviewer changed the frozen checkout revision")
            findings, execution.conversion_omissions = convert_findings(
                result.items, repository=checkout, case=case, producer=producer, comparison=comparison)
            state = execution.terminal_result["analysis_state"]
            if state != "complete":
                execution.status = "incomplete" if state == "incomplete" else "failed"
                execution.diagnostics.append("native analysis is " + state)
                return execution
            if exit_code != 0:
                execution.diagnostics.append("native reviewer did not exit successfully")
                return execution
            execution.candidate = {"pr": {"repo": case.repo, "pr_number": case.pr_number,
                "base": case.base, "head": case.head}, "agent": producer, "findings": findings}
            if execution.usage:
                execution.candidate['usage'] = copy.deepcopy(execution.usage)
            execution.status = ("conversion_limited" if execution.conversion_omissions
                                else "complete" if findings else "completed_empty")
            return execution
    except (git_ops.GitError, OSError, UnicodeError, ValueError) as exc:
        execution.status = "failed"
        execution.candidate = None
        execution.diagnostics.append(str(exc))
        return execution
