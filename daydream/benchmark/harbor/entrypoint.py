"""Controlled in-container reviewer invoked by ``DaydreamReviewAgent.run``.

Validate the backend before mapping reviewer credentials into its native environment.
Run the shared immutable committed-review API against the attested two-commit
projection. Only a complete sealed current result can publish a candidate.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from daydream import git_ops
from daydream.artifact_visibility import ArtifactVisibilityError
from daydream.backends import BackendExecutionInput
from daydream.benchmark.harbor import candidate, env_policy
from daydream.committed_review import CommittedReviewRequest, run_committed_review
from daydream.hosted_review import HostedReviewPolicy
from daydream.immutable_review import ImmutableReviewInput
from daydream.json_utils import atomic_write_bytes
from daydream.review_profile import ProfileError, resolve_harbor_profile

_DEFAULT_REPO_DIR = "/workspace/repo"
_DEFAULT_ARTIFACT_PATH = "/logs/artifacts/review.json"
_DEFAULT_TRAJECTORY_PATH = Path("/logs/agent/trajectory.json")
_MAX_IMMUTABLE_INPUT_BYTES = 128 * 1024

# Shared allowlist backing both the host-side agent gate and the in-container
# gate; any new reviewer backend must be added here.
_SUPPORTED_BACKENDS: tuple[str, ...] = ("pi",)


class EntrypointError(Exception):
    """Agent failure during reviewer configuration or execution."""


def _load_immutable_input(path: Path) -> ImmutableReviewInput:
    """Admit only bounded regular trusted metadata, outside captured model source."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_IMMUTABLE_INPUT_BYTES:
        raise EntrypointError("immutable review metadata must be a bounded regular file")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_IMMUTABLE_INPUT_BYTES + 1)
    if len(raw) > _MAX_IMMUTABLE_INPUT_BYTES:
        raise EntrypointError("immutable review metadata exceeds its supported size")
    return ImmutableReviewInput.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ParsedReviewerInput:
    """Validated container controls and opaque per-run execution inputs."""

    backend: str
    model: str
    repo_dir: Path
    artifact_path: Path
    trajectory_path: Path
    case_id: str
    base_ref: str
    head_ref: str
    profile_candidate: str | None
    immutable_path: Path
    bundle_path: Path
    agent_dir: Path | None
    execution: BackendExecutionInput = field(repr=False, compare=False)


def require_supported_backend(environment: Mapping[str, str]) -> str:
    """Validate the selected Harbor reviewer backend from its container map."""
    backend = environment.get(env_policy.BACKEND_ENV, "pi").strip().lower()
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


def _require_openrouter_endpoint(url: str) -> None:
    """Require the supported HTTPS endpoint without embedded userinfo."""
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.hostname.lower() != "openrouter.ai"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EntrypointError("DAYDREAM_REVIEW_BASE_URL must be an HTTPS openrouter.ai endpoint")


def _sanitize_reviewer_environment(environment: Mapping[str, str]) -> dict[str, str]:
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

    base_url = _required_env(source, env_policy.BASE_URL_ENV)
    _require_openrouter_endpoint(base_url)
    api_key = _required_env(source, env_policy.API_KEY_ENV)
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
    model = env_policy.require_reviewer_model(source)
    sanitized = _sanitize_reviewer_environment(source)
    backend_execution = BackendExecutionInput.from_environment(sanitized, backend=backend)
    return ParsedReviewerInput(
        backend=backend,
        model=model,
        repo_dir=Path(source.get(env_policy.REPO_DIR_ENV, _DEFAULT_REPO_DIR)),
        artifact_path=Path(source.get(env_policy.ARTIFACT_PATH_ENV, _DEFAULT_ARTIFACT_PATH)),
        trajectory_path=Path(source.get(env_policy.TRAJECTORY_PATH_ENV, str(_DEFAULT_TRAJECTORY_PATH))),
        case_id=_required_env(source, env_policy.CASE_ID_ENV),
        base_ref=source.get(env_policy.BASE_REF_ENV, "base").strip() or "base",
        head_ref=source.get(env_policy.HEAD_REF_ENV, "head").strip() or "head",
        profile_candidate=source.get(env_policy.CANDIDATE_ENV) or None,
        immutable_path=Path(source.get(env_policy.IMMUTABLE_INPUT_ENV, "/opt/daydream-input/immutable-request.json")),
        bundle_path=Path(source.get(env_policy.BUNDLE_ENV, "/opt/daydream-input/repository.bundle")),
        agent_dir=Path(source[env_policy.AGENT_DIR_ENV]) if source.get(env_policy.AGENT_DIR_ENV) else None,
        execution=backend_execution,
    )


async def main(environment: Mapping[str, str]) -> int:
    """Run one parsed Harbor container input and publish its candidate artifact."""
    try:
        parsed = parse_reviewer_environment(environment)
        profile = resolve_harbor_profile(
            candidate_env=env_policy.CANDIDATE_ENV,
            env={} if parsed.profile_candidate is None else {env_policy.CANDIDATE_ENV: parsed.profile_candidate},
        )
        immutable = _load_immutable_input(parsed.immutable_path)
        if immutable.to_dict()["materialization"]["kind"] != "synthetic":
            raise EntrypointError("Harbor requires an attested synthetic two-commit projection")
        parsed.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        agent_dir = parsed.agent_dir or Path(tempfile.mkdtemp(prefix="pi-", dir=parsed.trajectory_path.parent))
        private_base = Path(tempfile.mkdtemp(prefix="private-", dir=parsed.trajectory_path.parent))
        output_dir = parsed.trajectory_path.parent / ("committed-" + private_base.name.removeprefix("private-"))
        # Native receives process controls and its selected provider credential only.
        # Trusted provenance paths and operator control-plane metadata stay on host.
        native = parsed.execution.child_environment()
        native = {
            key: native[key] for key in ("PATH", "LANG", "PI_API_KEY", "PI_THINKING", "PI_TELEMETRY") if key in native
        }
        policy = HostedReviewPolicy(
            agent_dir=agent_dir.resolve(), environment=native, output_dir=output_dir.resolve(),
            private_base=private_base.resolve(), profile=profile, trusted_prepared_source=True,
            provider="openrouter", model=parsed.model,
        )
        result = await run_committed_review(CommittedReviewRequest(
            source=parsed.repo_dir.resolve(), base=immutable.to_dict()["materialization"]["requested_base_sha"],
            hosted_policy=policy, immutable_input=immutable, immutable_bundle=parsed.bundle_path.resolve(),
        ))
        artifact = candidate.build_committed_candidate(
            result, immutable_input=immutable.to_dict(), case_id=parsed.case_id,
            base_ref=parsed.base_ref, head_ref=parsed.head_ref,
        )
        # Delivery is a separate required step; a failed trajectory delivery must
        # not leave a successful candidate while local sealed evidence is retained.
        atomic_write_bytes(parsed.trajectory_path, (output_dir / "trajectory.json").read_bytes())
        candidate.write_candidate_artifact_atomic(parsed.artifact_path, artifact)
        return 0
    except (EntrypointError, candidate.CandidateError, ArtifactVisibilityError,
            git_ops.GitError, ProfileError, ValueError, OSError) as exc:
        print(
            f"daydream review agent failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(dict(os.environ))))
