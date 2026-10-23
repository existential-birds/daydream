"""Native review execution and conservative conversion from finalized findings."""
from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from daydream import git_ops
from daydream.backends import BackendExecutionInput, RetryPolicy, create_backend
from daydream.backends.pi import _configured_pi_model, native_pi_provider, native_pi_settings
from daydream.benchmark.reviewbench.contract import REVIEWBENCH_REVISION, Case
from daydream.git_ops.process import _run_git
from daydream.git_ops.source import frozen_source
from daydream.hunk_index import parse_hunks
from daydream.run_config import (
    RunConfig,
    _resolved_backend_name,
    _resolved_model,
    _resolved_reasoning_effort,
)

_REVIEW_PHASES = ('review', 'intent', 'exploration', 'alternatives', 'per_stack_review',
                  'parse', 'wonder', 'merge', 'arbiter', 'suppression', 'supervise', 'diagram')


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


# These are ordinary review-policy overrides, not a second native configuration
# resolver. Source, authentication, outputs and publication stay host-owned.
_REVIEW_OVERRIDES = ('stack', 'cleanup', 'quiet', 'diagram', 'backend', 'model', 'reasoning_effort',
    'review_backend', 'review_model', 'parse_model', 'exploration_model', 'log_mode', 'latency_profile', 'shallow',
    'shallow_fanout_threshold', 'precision_mode', 'ignore_paths',
    'deep_shard_enabled', 'deep_shard_max_files', 'deep_shard_max_bytes', 'deep_shard_fanout_cap',
    'deep_shard_frontier_max', 'review_cache_enabled', 'review_cache_max_entries', 'review_cache_max_bytes',
    'review_cache_max_age_days')

_EXECUTION_OVERRIDES = ('stream_idle_timeout_s', 'pi_response_idle_timeout_s',
    'codex_terminal_drain_timeout_s', 'codex_process_reap_timeout_s', 'codex_cleanup_timeout_s')


def _execution_values(reviewer: Mapping[str, Any]) -> dict[str, Any]:
    error = 'incompatible frozen native execution policy; regenerate the export'
    policy = reviewer.get('execution')
    if not isinstance(policy, dict) or set(policy) != {'retry_policy', *_EXECUTION_OVERRIDES}:
        raise ValueError(error)
    retry = policy['retry_policy']
    if (not isinstance(retry, dict)
            or set(retry) != {'attempts', 'base_delay_s', 'max_delay_s', 'retry_recovery_allowance_s'}
            or type(retry['attempts']) is not int or retry['attempts'] < 0):
        raise ValueError(error)
    values = {name: value for name, value in retry.items() if name != 'attempts'}
    values.update((name, policy[name]) for name in _EXECUTION_OVERRIDES)
    for name, value in values.items():
        if value is None and name in {'retry_recovery_allowance_s', 'stream_idle_timeout_s',
                                      'pi_response_idle_timeout_s'}:
            continue
        if type(value) not in {int, float}:
            raise ValueError(error)
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (value == 0 and name in _EXECUTION_OVERRIDES):
            raise ValueError(error)
    return {**policy, 'retry_policy': RetryPolicy(**retry)}


def reviewer_execution(reviewer: Mapping[str, Any], environment: Mapping[str, str]) -> BackendExecutionInput:
    """Restore parsed nonsecret limits onto the isolated native capability."""
    return replace(BackendExecutionInput.from_environment(environment, backend=reviewer['backend']),
                   **_execution_values(reviewer))


def freeze_reviewer(config: RunConfig) -> dict[str, Any]:
    """Freeze bounded nonsecret producer policy independently of the supplied judge."""
    from daydream.benchmark.reviewbench.harbor_profile import selected_reviewer_policy
    from daydream.run_artifacts import _resolve_review_profile

    _resolve_review_profile(config)
    if (config.extra_copy or config.test_command or config.test_required_suites or config.approve_on_clean
            or config.scope_issue_filing or (config.observability is not None and config.observability.destinations)):
        raise ValueError('unsupported benchmark native policy: '
                         'copying, tests, publication and tracing are not admitted')
    phases = set(_REVIEW_PHASES) | set(config.file_config.phases if config.file_config else {})
    backend = _resolved_backend_name(config, 'per_stack_review')
    if backend not in {'pi', 'codex'} or {_resolved_backend_name(config, phase) for phase in phases} != {backend}:
        raise ValueError('public ReviewBench requires one selected native Pi or Codex reviewer for every phase')
    root = Path(config.target or Path.cwd()).resolve()
    controls = dict(os.environ)
    if controls.get('PI_CODING_AGENT_DIR'):
        controls['PI_CODING_AGENT_DIR'] = str(Path(controls['PI_CODING_AGENT_DIR']).resolve())
    capability = BackendExecutionInput.from_environment(controls, backend=backend)
    if backend == 'pi':
        capability = replace(capability, pi_settings=native_pi_settings(root, agent_dir=capability.pi_agent_dir))
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
        'execution': {'retry_policy': asdict(capability.retry_policy),
                      **{name: getattr(capability, name) for name in _EXECUTION_OVERRIDES}},
        'run_config': {name: copy.deepcopy(getattr(config, name)) for name in _REVIEW_OVERRIDES},
        'file_config': file_config,
        'profile': {'data': asdict(config.review_profile.profile), 'source_kind': config.review_profile.source_kind},
        'fanout_concurrency': 1})


def reviewer_config(reviewer: Mapping[str, Any], repository: Path) -> RunConfig:
    """Rehydrate the frozen ordinary policy without reading reviewed-source settings."""
    from daydream.config_file import DaydreamFileConfig
    from daydream.review_profile import Arbitration, Pipeline, ResolvedProfile, ReviewProfile, Strategy, Suppression

    _execution_values(reviewer)
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
    phases = set(_REVIEW_PHASES) | set(config.file_config.phases if config.file_config else {})
    if {_resolved_backend_name(config, phase) for phase in phases} != {reviewer['backend']}:
        raise ValueError('frozen reviewer requires an unavailable mixed native capability')
    return config


@dataclass(frozen=True)
class PublicHarborTask:
    """Frozen Harbor identity, verified against origin-free refs and the complete patch."""

    case: Case
    upstream_revision: str
    patch: str
    task_id: int
    evaluation_id: int
    profile_sha256: str
    author_description: str | None = None

    def verify(self, repository: Path) -> str:
        case = self.case
        if (self.upstream_revision != REVIEWBENCH_REVISION
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
        return merge_base


@dataclass
class CaseExecution:
    """One attempt, with partial evidence kept separate from scorable findings."""

    status: str
    candidate: dict[str, Any] | None = None
    terminal_result: dict[str, Any] | None = None
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
    """Convert proven new-side HEAD locations; retain whole-file or unreliable hints as explicit omissions."""
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
