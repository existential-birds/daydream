"""Controlled in-container runner for the privacy-safe Harbor review agent (issue #780).

Invoked inside the Harbor task container (``python -m
daydream.benchmark.harbor.entrypoint``) by :class:`DaydreamReviewAgent`'s
``run`` via ``environment.exec``. Owns the fail-closed review surface: it maps
only the ``DAYDREAM_REVIEW_*`` reviewer config/credential into the selected
backend's native env (OpenRouter for pi, Anthropic for claude), refuses any
unsupported backend *before* any credential mapping or reviewing, runs the real
Daydream runner **in-process** against the frozen ``base``/``head`` snapshot
with a fully controlled :class:`RunConfig` (review-only, non-interactive,
archiving and eval disabled, empty file config), then publishes the canonical
candidate artifact from the runner's ``merged-items.json``.

Every failure class surfaces as a typed exception carrying a ``kind`` -- the
trial is an unscored agent failure, never a silent pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from daydream.backends import BackendExecutionInput
from daydream.benchmark.harbor import candidate, env_policy
from daydream.config_file import DaydreamFileConfig
from daydream.git_ops import StaticGitHubAuth
from daydream.github_app import GitHubExecutionInput
from daydream.runner import RunConfig, RunnerExecutionInput

_DEFAULT_REPO_DIR = "/workspace/repo"
_DEFAULT_ARTIFACT_PATH = "/logs/artifacts/review.json"
_DEFAULT_TRAJECTORY_PATH = Path("/logs/agent/trajectory.json")

_CASE_ID_ENV = env_policy.CASE_ID_ENV
_BASE_REF_ENV = env_policy.BASE_REF_ENV
_HEAD_REF_ENV = env_policy.HEAD_REF_ENV
_BACKEND_ENV = env_policy.BACKEND_ENV
# Shared allowlist backing both the host-side agent gate and the in-container
# gate; any new reviewer backend must be added here.
_SUPPORTED_BACKENDS: tuple[str, ...] = ("pi", "claude")
_API_KEY_ENV = env_policy.API_KEY_ENV
_BASE_URL_ENV = env_policy.BASE_URL_ENV
_REPO_DIR_ENV = env_policy.REPO_DIR_ENV
_ARTIFACT_PATH_ENV = env_policy.ARTIFACT_PATH_ENV
_TRAJECTORY_PATH_ENV = env_policy.TRAJECTORY_PATH_ENV
_CANDIDATE_ENV = env_policy.CANDIDATE_ENV


class EntrypointError(Exception):
    """Typed agent-failure carrier for the in-container entrypoint.

    Raised on any fallible step (missing case key, unsupported backend, runner
    failure) so a failed run reports which failure class occurred instead of
    presenting silence.
    """


@dataclass(frozen=True)
class ParsedReviewerInput:
    """Validated container controls and opaque per-run execution inputs."""

    backend: str
    model: str | None
    repo_dir: Path
    artifact_path: Path
    trajectory_path: Path
    case_id: str
    base_ref: str
    head_ref: str
    profile_candidate: str | None
    execution: RunnerExecutionInput = field(repr=False, compare=False)


def require_supported_backend(environment: Mapping[str, str]) -> str:
    """Validate the selected Harbor reviewer backend from its container map."""
    backend = environment.get(_BACKEND_ENV, "pi").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        supported = ", ".join(repr(item) for item in _SUPPORTED_BACKENDS)
        raise EntrypointError(
            f"unsupported DAYDREAM_REVIEW_BACKEND={backend!r}; supported backends: {supported}"
        )
    return backend


def _required_env(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise EntrypointError(f"missing required environment variable {name!r}")
    return value


def _sanitize_reviewer_environment(
    environment: Mapping[str, str], *, backend: str,
) -> dict[str, str]:
    """Return a native backend map with all unrelated credentials removed."""
    source = dict(environment)
    container = env_policy.CONTAINER
    sanitized = {
        key: value
        for key, value in source.items()
        if key not in container.github_credential_vars
        and key not in container.unselected_pi_credentials
        and not any(key.startswith(prefix) for prefix in container.scrub_prefixes)
    }
    for alias in container.control_plane_aliases:
        sanitized.pop(alias, None)

    if backend == "claude":
        api_key = (source.get(env_policy.ANTHROPIC_API_KEY_ENV) or "").strip()
        auth_token = (source.get(env_policy.ANTHROPIC_AUTH_TOKEN_ENV) or "").strip()
        if not api_key and not auth_token:
            raise EntrypointError(
                "claude backend requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN "
                "in the reviewer environment"
            )
        base_url = (source.get(env_policy.ANTHROPIC_BASE_URL_ENV) or "").strip()
        if base_url:
            parsed = urllib.parse.urlsplit(base_url)
            if (
                parsed.scheme.lower() != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise EntrypointError("ANTHROPIC_BASE_URL must be an HTTPS endpoint")
            sanitized[env_policy.ANTHROPIC_BASE_URL_ENV] = base_url
        if api_key:
            sanitized[env_policy.ANTHROPIC_API_KEY_ENV] = api_key
        if auth_token:
            sanitized[env_policy.ANTHROPIC_AUTH_TOKEN_ENV] = auth_token
        return sanitized

    api_key = (source.get(_API_KEY_ENV) or "").strip()
    base_url = (source.get(_BASE_URL_ENV) or "").strip()
    if not base_url:
        raise EntrypointError(
            "missing required environment variable 'DAYDREAM_REVIEW_BASE_URL'"
        )
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != "openrouter.ai"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EntrypointError(
            "DAYDREAM_REVIEW_BASE_URL must be an HTTPS openrouter.ai endpoint"
        )
    if not api_key:
        raise EntrypointError(
            "missing required environment variable 'DAYDREAM_REVIEW_API_KEY'"
        )
    sanitized["PI_PROVIDER"] = "openrouter"
    sanitized[env_policy.PI_API_KEY_ENV] = api_key
    sanitized["PI_TELEMETRY"] = "0"
    for control in ("PI_THINKING", "PI_CODING_AGENT_DIR"):
        if control in source:
            sanitized[control] = source[control]
    return sanitized


def parse_reviewer_environment(environment: Mapping[str, str]) -> ParsedReviewerInput:
    """Parse one Harbor container map without reading or mutating process state."""
    source = dict(environment)
    backend = require_supported_backend(source)
    sanitized = _sanitize_reviewer_environment(source, backend=backend)
    backend_execution = BackendExecutionInput.from_environment(sanitized, backend=backend)
    github_environment = dict(sanitized)
    for credential in env_policy.CONTAINER.github_subprocess_drops:
        github_environment.pop(credential, None)
    execution = RunnerExecutionInput(
        backend=backend_execution,
        github=GitHubExecutionInput(auth=StaticGitHubAuth(github_environment)),
    )
    return ParsedReviewerInput(
        backend=backend,
        model=source.get(env_policy.MODEL_ENV),
        repo_dir=Path(source.get(_REPO_DIR_ENV, _DEFAULT_REPO_DIR)),
        artifact_path=Path(source.get(_ARTIFACT_PATH_ENV, _DEFAULT_ARTIFACT_PATH)),
        trajectory_path=Path(source.get(_TRAJECTORY_PATH_ENV, str(_DEFAULT_TRAJECTORY_PATH))),
        case_id=_required_env(source, _CASE_ID_ENV),
        base_ref=source.get(_BASE_REF_ENV, "base").strip() or "base",
        head_ref=source.get(_HEAD_REF_ENV, "head").strip() or "head",
        profile_candidate=source.get(_CANDIDATE_ENV) or None,
        execution=execution,
    )


def build_run_config(
    repo_dir: str | Path,
    trajectory_path: str | Path,
    *,
    backend: str,
    model: str | None,
    profile_candidate: str | None,
    base_ref: str = "base",
) -> RunConfig:
    """Build the fixed review configuration from already parsed controls."""
    from daydream.review_profile import ProfileError, resolve_harbor_profile

    try:
        profile_environment = (
            {} if profile_candidate is None else {_CANDIDATE_ENV: profile_candidate}
        )
        resolved = resolve_harbor_profile(
            candidate_env=_CANDIDATE_ENV, env=profile_environment
        )
    except ProfileError as exc:
        raise EntrypointError(f"invalid review-profile candidate: {exc}") from exc
    config = RunConfig(
        target=str(repo_dir),
        output_mode="review",
        base=base_ref,
        non_interactive=True,
        archive=False,
        run_eval=False,
        findings_out=None,
        trajectory_path=Path(trajectory_path),
        backend=backend,
        model=model,
        file_config=DaydreamFileConfig(),
    )
    config.review_profile = resolved
    return config


def publish_review(
    *,
    repo_dir: str | Path,
    artifact_path: str | Path,
    case_id: str,
    base_ref: str = "base",
    head_ref: str = "head",
) -> None:
    """Publish the candidate artifact from the runner's merged-items output."""
    repo_dir = Path(repo_dir)
    merged = repo_dir / ".daydream" / "deep" / "merged-items.json"
    try:
        raw: Any = json.loads(merged.read_text())
    except OSError as exc:
        raise candidate.CandidateError(
            f"merged items output is missing or unreadable at {merged}: {exc}",
            kind="missing_merged",
        ) from exc
    except json.JSONDecodeError as exc:
        raise candidate.CandidateError(
            f"merged items output {merged} is corrupt JSON: {exc}",
            kind="corrupt_merged",
        ) from exc
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise candidate.CandidateError(
            f"merged items output {merged} is corrupt (items is not a list)",
            kind="corrupt_merged",
        )
    findings = candidate.build_candidate_findings(items, case_id=case_id)
    artifact = candidate.build_candidate_artifact(
        case_id, findings, base_ref=base_ref, head_ref=head_ref
    )
    candidate.write_candidate_artifact_atomic(artifact_path, artifact)


async def main(environment: Mapping[str, str]) -> int:
    """Run one parsed Harbor container input and publish its candidate artifact."""
    try:
        parsed = parse_reviewer_environment(environment)
        config = build_run_config(
            repo_dir=parsed.repo_dir,
            trajectory_path=parsed.trajectory_path,
            backend=parsed.backend,
            model=parsed.model,
            profile_candidate=parsed.profile_candidate,
            base_ref=parsed.base_ref,
        )
        from daydream import runner

        if await runner.run(config, execution=parsed.execution) != 0:
            raise EntrypointError("daydream runner exited non-zero")
        publish_review(
            repo_dir=parsed.repo_dir,
            artifact_path=parsed.artifact_path,
            case_id=parsed.case_id,
            base_ref=parsed.base_ref,
            head_ref=parsed.head_ref,
        )
        return 0
    except (EntrypointError, candidate.CandidateError) as exc:
        print(
            f"daydream review agent failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(dict(os.environ))))
