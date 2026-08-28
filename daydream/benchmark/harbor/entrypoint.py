"""Controlled in-container runner for the privacy-safe Harbor review agent (issue #780).

Invoked inside the Harbor task container (``python -m
daydream.benchmark.harbor.entrypoint``) by :class:`DaydreamReviewAgent`'s
``run`` via ``environment.exec``. Owns the fail-closed review surface: it maps
only the ``DAYDREAM_REVIEW_*`` reviewer config/credential into Pi's native
OpenRouter env, refuses any unsupported backend *before* reviewing, runs the real
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
from pathlib import Path
from typing import Any, Mapping

from daydream.benchmark.harbor import candidate
from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig

_DEFAULT_REPO_DIR = "/workspace/repo"
_DEFAULT_ARTIFACT_PATH = "/logs/artifacts/review.json"
_DEFAULT_TRAJECTORY_PATH = Path("/logs/agent/trajectory.json")

_CASE_ID_ENV = "DAYDREAM_REVIEW_CASE_ID"
_BASE_REF_ENV = "DAYDREAM_REVIEW_BASE_REF"
_HEAD_REF_ENV = "DAYDREAM_REVIEW_HEAD_REF"
_BACKEND_ENV = "DAYDREAM_REVIEW_BACKEND"
# Shared allowlist backing both the host-side agent gate and the in-container
# gate; any new reviewer backend must be added here.
_SUPPORTED_BACKENDS: tuple[str, ...] = ("pi", "claude")
_API_KEY_ENV = "DAYDREAM_REVIEW_API_KEY"
_BASE_URL_ENV = "DAYDREAM_REVIEW_BASE_URL"
_REPO_DIR_ENV = "DAYDREAM_REVIEW_REPO_DIR"
_ARTIFACT_PATH_ENV = "DAYDREAM_REVIEW_ARTIFACT_PATH"
_TRAJECTORY_PATH_ENV = "DAYDREAM_REVIEW_TRAJECTORY_PATH"
# Control-plane candidate channel (R10/R11): a dedicated var, distinct from the
# normal-run ``DAYDREAM_REVIEW_PROFILE``, so a benchmarked repository can never
# configure its own evaluator. Carried into the container through the
# ``DAYDREAM_REVIEW_*`` child-env allowlist (agent.build_child_env).
_CANDIDATE_ENV = "DAYDREAM_REVIEW_PROFILE_CANDIDATE"


class EntrypointError(Exception):
    """Typed agent-failure carrier for the in-container entrypoint.

    Raised on any fallible step (missing case key, unsupported backend, runner
    failure) so a failed run reports which failure class occurred instead of
    presenting silence.
    """


def build_run_config(
    repo_dir: str,
    trajectory_path: str | Path,
    *,
    backend: str,
    model: str | None,
    base_ref: str = "base",
) -> RunConfig:
    """Build the fully controlled in-process :class:`RunConfig`.

    Review-only (``output_mode="review"``), headless, with archiving and
    evaluation disabled and a controlled-empty :class:`DaydreamFileConfig` so
    the target repository's ``.daydream.toml`` is never loaded. ``findings_out``
    stays ``None`` -- the ``--findings-out`` emission path performs a live PR
    lookup and must be forbidden for an offline snapshot. ``base_ref`` carries
    the ``DAYDREAM_REVIEW_BASE_REF`` value rendered into the container env.

    The review profile is resolved through the Harbor explicit-only resolver
    (R10): the control-plane ``DAYDREAM_REVIEW_PROFILE_CANDIDATE`` env (or the
    packaged default) is parsed + validated BEFORE the run, so an invalid
    candidate aborts here — no review ever starts and no artifact is written.
    The target repo's own config can never change the candidate.

    Raises:
        EntrypointError: On an invalid review-profile candidate (the process
            exits non-zero and ``publish_review`` is never reached).
    """
    from daydream.review_profile import ProfileError, resolve_harbor_profile

    try:
        resolved = resolve_harbor_profile(candidate_env=_CANDIDATE_ENV)
    except ProfileError as exc:
        raise EntrypointError(
            f"invalid review-profile candidate: {exc}"
        ) from exc
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


def require_supported_backend() -> str:
    """Refuse any backend other than ``pi`` or ``claude``, before any reviewing.

    Reads ``DAYDREAM_REVIEW_BACKEND`` (default ``"pi"``). An unsupported
    backend raises :class:`EntrypointError` before tools are installed or
    network access is widened.
    """
    backend = os.environ.get(_BACKEND_ENV, "pi").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        supported = ", ".join(repr(b) for b in _SUPPORTED_BACKENDS)
        raise EntrypointError(
            f"unsupported DAYDREAM_REVIEW_BACKEND={backend!r}; supported backends: {supported}"
        )
    return backend


def apply_reviewer_env(env: Mapping[str, str] | None = None) -> None:
    """Map only reviewer config/credential into Pi's OpenRouter env.

    The Harbor reviewer is intentionally OpenRouter-only. The control-plane
    credential becomes Pi's generic ``PI_API_KEY`` and the Daydream Pi backend
    maps it to ``OPENROUTER_API_KEY`` only in the child process. Any inherited
    raw provider or judge credential is cleared first so it cannot leak into
    the reviewed scope. A non-HTTPS or non-OpenRouter base URL fails closed.
    """
    source = dict(os.environ if env is None else env)
    for prefix in (
        "DAYDREAM_JUDGE_",
        "ANTHROPIC_",
        "OPENAI_",
        "OPENROUTER_",
        "PI_",
    ):
        for key in [k for k in os.environ if k.startswith(prefix)]:
            os.environ.pop(key, None)
    api_key = (source.get(_API_KEY_ENV) or "").strip()
    base_url = (source.get(_BASE_URL_ENV) or "").strip()
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
    os.environ["PI_PROVIDER"] = "openrouter"
    os.environ["PI_API_KEY"] = api_key
    os.environ["PI_TELEMETRY"] = "0"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EntrypointError(f"missing required environment variable {name!r}")
    return value


def publish_review(
    *,
    repo_dir: str | Path,
    artifact_path: str | Path,
    case_id: str,
    base_ref: str = "base",
    head_ref: str = "head",
) -> None:
    """Publish the candidate artifact from the runner's merged-items output.

    Locates ``<repo_dir>/.daydream/deep/merged-items.json``; raises
    :class:`CandidateError` (via its ``kind``) on a missing/corrupt merged
    output, an over-limit candidate set, or a failed artifact write -- never
    silently truncates or substitutes a fallback artifact.
    """
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


async def main(*, monkeypatch_env: Mapping[str, str] | None = None) -> int:
    """Run the in-process reviewer and publish the candidate artifact.

    Returns an exit code (0 on a completed review, non-zero on any failure).
    *monkeypatch_env* is a test-only seam overriding/augmenting the process
    environment from the caller; production defaults remain the fixed
    in-container paths.
    """
    saved_env = None
    if monkeypatch_env is not None:
        saved_env = os.environ.copy()
        for key, value in monkeypatch_env.items():
            os.environ[key] = value
    try:
        apply_reviewer_env()
        backend = require_supported_backend()
        case_id = _required_env(_CASE_ID_ENV)
        repo_dir = os.environ.get(_REPO_DIR_ENV, _DEFAULT_REPO_DIR)
        artifact_path = os.environ.get(_ARTIFACT_PATH_ENV, _DEFAULT_ARTIFACT_PATH)
        trajectory_path = os.environ.get(_TRAJECTORY_PATH_ENV, str(_DEFAULT_TRAJECTORY_PATH))
        base_ref = os.environ.get(_BASE_REF_ENV, "base").strip() or "base"
        head_ref = os.environ.get(_HEAD_REF_ENV, "head").strip() or "head"
        config = build_run_config(
            repo_dir=repo_dir,
            trajectory_path=trajectory_path,
            backend=backend,
            model=os.environ.get("DAYDREAM_REVIEW_MODEL"),
            base_ref=base_ref,
        )
        from daydream import runner

        if await runner.run(config) != 0:
            raise EntrypointError("daydream runner exited non-zero")
        publish_review(
            repo_dir=repo_dir,
            artifact_path=artifact_path,
            case_id=case_id,
            base_ref=base_ref,
            head_ref=head_ref,
        )
        return 0
    except (EntrypointError, candidate.CandidateError) as exc:
        print(
            f"daydream review agent failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        # Test-only seam hygiene: never leak monkeypatched vars (or the
        # reviewer env vars) into the surrounding process after main returns.
        if saved_env is not None:
            os.environ.clear()
            os.environ.update(saved_env)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
