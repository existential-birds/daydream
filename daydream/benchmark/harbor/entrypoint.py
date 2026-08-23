"""Controlled in-container runner for the privacy-safe Harbor review agent (issue #780).

Invoked inside the Harbor task container (``python -m
daydream.benchmark.harbor.entrypoint``) by :class:`DaydreamReviewAgent`'s
``run`` via ``environment.exec``. Owns the fail-closed review surface: it maps
only the ``DAYDREAM_REVIEW_*`` reviewer config/credential into the Anthropic
SDK env, refuses any unsupported backend *before* reviewing, runs the real
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
from pathlib import Path
from typing import Any, Mapping

from daydream.benchmark.harbor import candidate
from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig

_DEFAULT_REPO_DIR = "/workspace/repo"
_DEFAULT_ARTIFACT_PATH = "/logs/artifacts/review.json"
_DEFAULT_TRAJECTORY_PATH = Path("/logs/agent/trajectory.json")

_CASE_ID_ENV = "DAYDREAM_REVIEW_CASE_ID"
_BACKEND_ENV = "DAYDREAM_REVIEW_BACKEND"
_API_KEY_ENV = "DAYDREAM_REVIEW_API_KEY"
_BASE_URL_ENV = "DAYDREAM_REVIEW_BASE_URL"
_REPO_DIR_ENV = "DAYDREAM_REVIEW_REPO_DIR"
_ARTIFACT_PATH_ENV = "DAYDREAM_REVIEW_ARTIFACT_PATH"
_TRAJECTORY_PATH_ENV = "DAYDREAM_REVIEW_TRAJECTORY_PATH"


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
) -> RunConfig:
    """Build the fully controlled in-process :class:`RunConfig`.

    Review-only (``output_mode="review"``), headless, with archiving and
    evaluation disabled and a controlled-empty :class:`DaydreamFileConfig` so
    the target repository's ``.daydream.toml`` is never loaded. ``findings_out``
    stays ``None`` -- the ``--findings-out`` emission path performs a live PR
    lookup and is forbidden for an offline snapshot.
    """
    return RunConfig(
        target=str(repo_dir),
        output_mode="review",
        base="base",
        non_interactive=True,
        archive=False,
        run_eval=False,
        findings_out=None,
        trajectory_path=Path(trajectory_path),
        backend=backend,
        model=model,
        file_config=DaydreamFileConfig(),
    )


def require_supported_backend() -> None:
    """Refuse any backend other than ``claude``, before any reviewing.

    Reads ``DAYDREAM_REVIEW_BACKEND`` (default ``"claude"``). An unsupported
    backend raises :class:`EntrypointError` before tools are installed or
    network access is widened.
    """
    backend = os.environ.get(_BACKEND_ENV, "claude").strip().lower()
    if backend != "claude":
        raise EntrypointError(
            f"unsupported DAYDREAM_REVIEW_BACKEND={backend!r}; only 'claude' is supported"
        )


def apply_reviewer_env(env: Mapping[str, str] | None = None) -> None:
    """Map only reviewer config/credential into the Claude SDK env.

    Propagates ``DAYDREAM_REVIEW_API_KEY``/``DAYDREAM_REVIEW_BASE_URL`` into
    ``ANTHROPIC_API_KEY``/``ANTHROPIC_BASE_URL`` on ``os.environ``; never
    silently substitutes a default credential.
    """
    source = dict(os.environ if env is None else env)
    api_key = source.get(_API_KEY_ENV)
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    base_url = source.get(_BASE_URL_ENV)
    if base_url:
        os.environ["ANTHROPIC_BASE_URL"] = base_url


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EntrypointError(f"missing required environment variable {name!r}")
    return value


def publish_review(*, repo_dir: str | Path, artifact_path: str | Path, case_id: str) -> None:
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
    artifact = candidate.build_candidate_artifact(case_id, findings)
    candidate.write_candidate_artifact_atomic(artifact_path, artifact)


async def main(*, monkeypatch_env: Mapping[str, str] | None = None) -> int:
    """Run the in-process reviewer and publish the candidate artifact.

    Returns an exit code (0 on a completed review, non-zero on any failure).
    *monkeypatch_env* is a test-only seam overriding/augmenting the process
    environment from the caller; production defaults remain the fixed
    in-container paths.
    """
    if monkeypatch_env is not None:
        for key, value in monkeypatch_env.items():
            os.environ[key] = value
    try:
        apply_reviewer_env()
        require_supported_backend()
        case_id = _required_env(_CASE_ID_ENV)
        repo_dir = os.environ.get(_REPO_DIR_ENV, _DEFAULT_REPO_DIR)
        artifact_path = os.environ.get(_ARTIFACT_PATH_ENV, _DEFAULT_ARTIFACT_PATH)
        trajectory_path = os.environ.get(_TRAJECTORY_PATH_ENV, str(_DEFAULT_TRAJECTORY_PATH))
        config = build_run_config(
            repo_dir=repo_dir,
            trajectory_path=trajectory_path,
            backend="claude",
            model=os.environ.get("DAYDREAM_REVIEW_MODEL"),
        )
        from daydream import runner

        if await runner.run(config) != 0:
            raise EntrypointError("daydream runner exited non-zero")
        publish_review(repo_dir=repo_dir, artifact_path=artifact_path, case_id=case_id)
        return 0
    except (EntrypointError, candidate.CandidateError) as exc:
        print(
            f"daydream review agent failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))