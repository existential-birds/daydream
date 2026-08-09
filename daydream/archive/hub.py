"""Opt-in HuggingFace dataset repo upload of daydream run bundles.

Resolves the 3-tier ``trajectory_hub_repo`` source (CLI flag -> env var ->
file config) and uploads a completed run bundle (``~/.daydream/archive/runs/
<session_id>/``) to a private HuggingFace dataset repo, one folder per run
keyed by session id. Everything here is non-fatal: the archive callback that
invokes it must never fail the run, so every failure mode degrades to a
warning and ``False``, and a pre-existing public target repo is reused only
with a visibility warning (the upload still proceeds).

``huggingface_hub`` is imported lazily inside :func:`upload_run_bundle` so
users who never enable the feature carry no hard dependency.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.runner import RunConfig

# Set lazily on first upload to avoid a hard dependency on huggingface_hub;
# tests swap it for a fake via monkeypatch.setattr(hub, "HfApi", ...).
HfApi: type | None = None

# Exponential backoff for commit-conflict upload retries (2s, 4s, ... capped at
# 120s), mirroring the shape agent.py applies to backend retries. Tests lower
# the base delay via monkeypatch to keep the retry tests fast.
_UPLOAD_RETRY_BASE_DELAY_S = 2.0
_UPLOAD_RETRY_MAX_DELAY_S = 120.0


def _warn(message: str) -> None:
    """Print a one-line warning through the daydream console (never raises)."""
    from daydream.ui import create_console, print_warning

    print_warning(create_console(), message)


def resolve_hub_repo(config: RunConfig) -> str | None:
    """Resolve the configured HuggingFace dataset repo id, or None if unset.

    Tier order (highest first): the CLI-tier ``config.trajectory_hub_repo``,
    then the ``DAYDREAM_TRAJECTORY_HUB_REPO`` env var, then the file-config
    ``trajectory_hub_repo``. Empty strings are treated as unset.
    """
    if config.trajectory_hub_repo:
        return config.trajectory_hub_repo
    env = os.environ.get("DAYDREAM_TRAJECTORY_HUB_REPO")
    if env:
        return env
    # The file config is read straight off the public ``RunConfig.file_config``
    # field — no deferred import from runner is needed (runner itself imports
    # archive lazily), and no private helper is reached into.
    file_config = config.file_config
    if file_config is not None and file_config.trajectory_hub_repo:
        return file_config.trajectory_hub_repo
    return None


def upload_run_bundle(run_dir: Path, repo_id: str, session_id: str) -> bool:
    """Upload ``run_dir`` to ``repo_id`` under ``path_in_repo=session_id``.

    Never raises: all failures degrade to a one-line warning and ``False`` so
    the archive callback cannot fail the run. Skips (``False`` + warning) when
    ``HF_TOKEN`` is absent. A pre-existing repo is reused with its current
    visibility (documented behavior), but a public one triggers a warning
    before the upload proceeds. Retries the upload commit up to 3 total
    attempts on a commit-conflict shape (concurrent commits from parallel
    processes), backing off exponentially between attempts.

    Returns:
        True on success, False when skipped or failed.
    """
    if not os.environ.get("HF_TOKEN"):
        _warn(f"Skip HF upload of {session_id}: HF_TOKEN not set (set it to upload run bundles)")
        return False

    global HfApi
    if HfApi is None:
        try:
            import huggingface_hub  # noqa: PLC0415 - optional dep, keep lazy

            HfApi = huggingface_hub.HfApi
        except ImportError:
            _warn("Skip HF upload: huggingface_hub not installed (pip install huggingface-hub)")
            return False

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(repo_id=repo_id, repo_type="dataset").private:
            _warn(
                f"HF repo {repo_id} already exists and is public; the run bundle for "
                f"{session_id} will be uploaded to a public repo"
            )
    except Exception as exc:  # noqa: BLE001 - absorb, the run must not fail
        _warn(f"HF upload of {session_id} to {repo_id} failed (non-fatal): {exc}")
        return False

    for attempt in range(1, 4):
        try:
            api.upload_folder(
                folder_path=str(run_dir),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=session_id,
                commit_message=f"daydream run {session_id}",
            )
            return True
        except Exception as exc:  # noqa: BLE001 - absorb, the run must not fail
            message = str(exc)
            conflict = "concurrent update" in message
            if conflict and attempt < 3:
                time.sleep(min(_UPLOAD_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), _UPLOAD_RETRY_MAX_DELAY_S))
                continue
            _warn(f"HF upload of {session_id} to {repo_id} failed (non-fatal): {message}")
            break
    return False
