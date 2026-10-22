"""Native review execution and conservative ReviewBench finding conversion.

The native runner remains responsible for backend/configuration/authentication and
review completeness. A benchmark label never supplies review success or source
coordinates that the finalized native result did not establish.
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from daydream import git_ops, runner
from daydream.backends import BackendExecutionInput, create_backend
from daydream.backends.pi import _configured_pi_model, native_pi_provider, native_pi_settings
from daydream.benchmark.reviewbench.candidate import candidate_package_digest, selected_extension_identity
from daydream.benchmark.reviewbench.contract import REVIEWBENCH_REVISION, Case
from daydream.deep.artifacts import diff_key
from daydream.git_ops.process import _run_git
from daydream.git_ops.source import frozen_source
from daydream.github_app import GitHubExecutionInput
from daydream.hunk_index import parse_hunks
from daydream.review_result import REVIEW_CONFIGURATION_PHASES, FinalizedReview, validate_terminal_result
from daydream.run_config import (
    FrozenAuthorDescription,
    RunConfig,
    _resolved_backend_name,
    _resolved_model,
    _resolved_reasoning_effort,
)


def parse_native_config(arguments: list[str]) -> RunConfig:
    """Use the ordinary CLI's operator policy, with benchmark-owned outputs disabled."""
    from daydream.commands import review
    from daydream.run_artifacts import _resolve_review_profile

    config = review._parse_args(['--review', '--non-interactive', '--no-archive', '--no-eval',
                                '--no-capture-data', *(arguments[1:] if arguments[:1] == ['--'] else arguments)])
    if (config.base or config.branch or config.findings_out or config.flow_name or config.start_at != 'review'
            or config.dump_artifacts or config.trajectory_path or config.trajectory_hub_repo
            or config.dataset_store_path or config.dataset_capture or config.force_worktree
            or config.pr_number is not None or config.pr_repo or config.approved_head_sha):
        raise ValueError('benchmark owns source/output and disables publication, resumes and dataset capture')
    config = replace(config, discover_github_provenance=False)
    _resolve_review_profile(config)
    return config


def freeze_native_execution(
    config: RunConfig, environment: Mapping[str, str] | None = None,
) -> runner.RunnerExecutionInput:
    """Freeze only runtime capabilities needed across the disposable checkout.

    Pi settings retain project/global precedence and distinguish unavailable settings
    from an explicitly empty frozen map. Osprey retains its ordinary ambient transport.
    Credentials and transport settings never become benchmark configuration metadata.
    """
    root = Path(config.target or Path.cwd()).resolve()
    controls = dict(os.environ if environment is None else environment)
    if controls.get('PI_CODING_AGENT_DIR'):
        controls['PI_CODING_AGENT_DIR'] = str(Path(controls['PI_CODING_AGENT_DIR']).resolve())
    phases = set(REVIEW_CONFIGURATION_PHASES) | set(config.file_config.phases if config.file_config else {})
    inputs: dict[str, BackendExecutionInput | None] = {}
    for name in sorted({_resolved_backend_name(config, phase) for phase in phases}):
        if name == 'osprey':
            inputs[name] = None
            continue
        capability = BackendExecutionInput.from_environment(controls, backend=name)
        if name == 'pi':
            capability = replace(capability, pi_settings=native_pi_settings(root, agent_dir=capability.pi_agent_dir))
        inputs[name] = capability
    fallback = next((value for value in inputs.values() if value is not None), None)
    if fallback is None:
        fallback = BackendExecutionInput.from_environment(controls, backend='claude')
    return runner.RunnerExecutionInput(fallback, GitHubExecutionInput(), backends=inputs)


# These are ordinary review-policy overrides, not a second native configuration
# resolver. Source, authentication, outputs, publication and transport stay host-owned.
_REVIEW_OVERRIDES = ('stack', 'cleanup', 'quiet', 'diagram', 'backend', 'model', 'reasoning_effort',
    'review_backend', 'review_model', 'parse_model', 'exploration_model', 'log_mode', 'latency_profile', 'shallow',
    'shallow_fanout_threshold', 'precision_mode', 'ignore_paths',
    'deep_shard_enabled', 'deep_shard_max_files', 'deep_shard_max_bytes', 'deep_shard_fanout_cap',
    'deep_shard_frontier_max', 'review_cache_enabled', 'review_cache_max_entries', 'review_cache_max_bytes',
    'review_cache_max_age_days')


def freeze_reviewer(config: RunConfig) -> dict[str, Any]:
    """Freeze bounded nonsecret producer policy independently of the supplied judge."""
    from daydream.benchmark.reviewbench.harbor_profile import selected_reviewer_policy
    from daydream.run_artifacts import _resolve_review_profile

    _resolve_review_profile(config)
    if (config.extra_copy or config.test_command or config.test_required_suites or config.approve_on_clean
            or config.scope_issue_filing or (config.observability is not None and config.observability.destinations)):
        raise ValueError('unsupported benchmark native policy: '
                         'copying, tests, publication and tracing are not admitted')
    execution = freeze_native_execution(config)
    names = set(execution.backends)
    backend = _resolved_backend_name(config, 'per_stack_review')
    if backend not in {'pi', 'codex'} or names != {backend}:
        raise ValueError('public ReviewBench requires one selected native Pi or Codex reviewer for every phase')
    capability = execution.backends[backend]
    assert capability is not None
    root = Path(config.target or Path.cwd())
    phases = set(REVIEW_CONFIGURATION_PHASES) | set(config.file_config.phases if config.file_config else {})
    phase_models = {phase: create_backend(backend, model=_resolved_model(config, phase),
        reasoning_effort=_resolved_reasoning_effort(config, phase), cwd=root, execution_input=capability).model
        for phase in sorted(phases)}
    settings = None if capability.pi_settings is None else dict(capability.pi_settings)
    model_override = _resolved_model(config, 'per_stack_review')
    provider = 'openai' if backend == 'codex' else native_pi_provider(root, execution_input=capability,
        configured_model=None if model_override is not None else _configured_pi_model(root, execution_input=capability),
        model_override=model_override, model=phase_models['per_stack_review'])
    if backend == 'pi' and provider is None:
        raise ValueError('selected native Pi model has no explicit operator provider')
    file_config = {} if config.file_config is None else asdict(config.file_config)
    file_config.pop('review_profile', None)
    assert config.review_profile is not None
    return selected_reviewer_policy({'backend': backend, 'model': phase_models['per_stack_review'],
        'phase_models': phase_models,
        'reasoning_effort': _resolved_reasoning_effort(config, 'per_stack_review'),
        'provider': provider, 'thinking': capability.pi_thinking if backend == 'pi' else None,
        'pi_settings': settings if backend == 'pi' else None,
        'run_config': {name: copy.deepcopy(getattr(config, name)) for name in _REVIEW_OVERRIDES},
        'file_config': file_config,
        'profile': {'data': asdict(config.review_profile.profile), 'source_kind': config.review_profile.source_kind},
        'fanout_concurrency': 1})


def reviewer_config(reviewer: Mapping[str, Any], repository: Path) -> RunConfig:
    """Rehydrate the frozen ordinary policy without reading reviewed-source settings."""
    from daydream.config_file import DaydreamFileConfig
    from daydream.review_profile import Arbitration, Pipeline, ResolvedProfile, ReviewProfile, Strategy, Suppression

    overrides = reviewer['run_config']
    if not isinstance(overrides, dict) or set(overrides) != set(_REVIEW_OVERRIDES):
        raise ValueError('incompatible frozen native reviewer configuration; regenerate the export')
    frozen = reviewer['profile']
    profile = frozen['data']
    pipeline = profile['pipeline']
    resolved_profile = ResolvedProfile(ReviewProfile(schema_version=profile['schema_version'], name=profile['name'],
        strategies={name: Strategy(**strategy) for name, strategy in profile['strategies'].items()},
        pipeline=Pipeline(**{**pipeline, 'arbitration': Arbitration(**pipeline['arbitration']),
                             'suppression': Suppression(**pipeline['suppression'])})), frozen['source_kind'])
    config = RunConfig(target=str(repository), **copy.deepcopy(overrides),
        file_config=DaydreamFileConfig(**copy.deepcopy(reviewer['file_config'])), review_profile=resolved_profile,
        output_mode='review', non_interactive=True, dataset_capture_disabled=True, archive=False, run_eval=False)
    phases = set(REVIEW_CONFIGURATION_PHASES) | set(config.file_config.phases if config.file_config else {})
    if {_resolved_backend_name(config, phase) for phase in phases} != {reviewer['backend']}:
        raise ValueError('frozen reviewer requires an unavailable mixed native capability')
    return config


@dataclass(frozen=True)
class PublicHarborTask:
    """Host-admitted public task identity for origin-free frozen snapshots.

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


def convert_findings(
    items: list[dict[str, Any]], *, repository: Path, case: Case, producer: str, comparison: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project native HEAD line hints onto the captured new-side diff.

    Whole-file, relocated, distrusted and unlocatable findings remain visible
    omissions rather than invented locations.
    """
    hunks = parse_hunks(comparison)
    findings: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    sources: dict[tuple[str, str], list[bytes] | None] = {}

    def source(revision: str, path: str) -> list[bytes] | None:
        key = revision, path
        if key not in sources:
            try:
                _, body = frozen_source(repository, revision, path)
                sources[key] = (body.split(b"\n")[:-1] if body.endswith(b"\n")
                                else body.split(b"\n") if body else [])
            except (git_ops.GitError, OSError, UnicodeError, ValueError):
                sources[key] = None
        return sources[key]

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
            if head_lines is None or line > len(head_lines):
                reason = "location_absent_from_head_regular_file"
            elif not new_side:
                reason = "no_new_side_location_in_captured_diff"
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
                      execution_input: runner.RunnerExecutionInput | None = None,
                      candidate_sha256: str | None = None,
                      public_task: PublicHarborTask | None = None,
                      evidence_directory: Path | None = None) -> CaseExecution:
    """Review one clean local frozen repository in a disposable independent checkout.

    No GitHub publication target is manufactured. The callback supplies actual
    terminal coverage even when the runner raises after admitting partial work.
    Optional private evidence uses the ordinary finalized dump so immutable fork
    documents remain bound to the same artifact owner as the root trajectory.
    Its caller must confine DAYDREAM_ARCHIVE_DIR to disposable private storage:
    the ordinary dump assembly also maintains an archive index.
    """
    execution = CaseExecution("failed")
    finalized: list[FinalizedReview] = []
    try:
        extension = selected_extension_identity()
        admitted_candidate = candidate_package_digest()
        if candidate_sha256 is not None and candidate_sha256 != admitted_candidate:
            raise ValueError('candidate bytes changed before native review; create a fresh Harbor export')
        native_execution = freeze_native_execution(config) if execution_input is None else execution_input
        if not producer.strip():
            raise ValueError("producer must be nonempty")
        git_ops.assert_is_worktree(repository)
        if git_ops.status_porcelain(repository).strip():
            raise ValueError("source repository is not clean")
        execution.source_identity = (_source_identity(repository, case) if public_task is None
                                     else public_task.verify(repository, case))
        if evidence_directory is not None:
            evidence_directory.mkdir(parents=True, exist_ok=False)
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
                archive=False, run_eval=False, trajectory_path=None,
                dump_artifacts=None if evidence_directory is None else str(evidence_directory), non_interactive=True)
            exit_code: int | None = None
            try:
                exit_code = await runner.run(native, review_result_handler=finalized.append, execution=native_execution)
            except Exception as exc:
                execution.diagnostics.append(f"native reviewer raised {type(exc).__name__}")
            result = finalized[0] if len(finalized) == 1 else None
            if result is not None:
                execution.terminal_result = result.terminal_result
                execution.configuration = result.configuration
                execution.usage = result.usage
                execution.retained_findings = result.items
            if selected_extension_identity() != extension:
                raise ValueError('native extension changed during review; create a fresh Harbor export')
            if candidate_package_digest() != admitted_candidate:
                raise ValueError('candidate bytes changed during native review; create a fresh Harbor export')
            if not finalized:
                execution.diagnostics.append("native reviewer did not produce a finalized terminal result")
                return execution
            if result is None:
                raise ValueError("native reviewer produced multiple terminal results")
            validate_terminal_result(result.terminal_result, expected_head_sha=case.head)
            revision = result.terminal_result["analyzed_revision"]
            if any(revision.get(key) != value for key, value in expected.items()):
                raise ValueError("native analyzed comparison differs from supplied frozen comparison")
            if git_ops.head_sha(checkout) != case.head or git_ops.diff_worktree_against(checkout, case.head, ["."]):
                raise ValueError("native reviewer changed the frozen checkout revision")
            findings, execution.conversion_omissions = convert_findings(
                result.items, repository=checkout, case=case, producer=producer, comparison=comparison)
            state = result.terminal_result["analysis_state"]
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
