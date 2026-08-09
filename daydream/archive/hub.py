"""Opt-in HuggingFace dataset repo upload of daydream run bundles.

Resolves the 3-tier ``trajectory_hub_repo`` source (CLI flag -> env var ->
file config) and uploads a completed run bundle (``~/.daydream/archive/runs/
<session_id>/``) to a private HuggingFace dataset repo, one folder per run
keyed by session id. Everything here is non-fatal: the archive callback that
invokes it must never fail the run, so every failure mode degrades to a
warning and ``False``.

``huggingface_hub`` is imported lazily inside :func:`upload_run_bundle` so
users who never enable the feature carry no hard dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.runner import RunConfig

# Set lazily on first upload to avoid a hard dependency on huggingface_hub;
# tests swap it for a fake via monkeypatch.setattr(hub, "HfApi", ...).
HfApi: type | None = None


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
    if config.file_config is not None and config.file_config.trajectory_hub_repo:
        return config.file_config.trajectory_hub_repo
    return None


def upload_run_bundle(run_dir: Path, repo_id: str, session_id: str) -> bool:
    """Upload ``run_dir`` to ``repo_id`` under ``path_in_repo=session_id``.

    Never raises: all failures degrade to a one-line warning and ``False`` so
    the archive callback cannot fail the run. Skips (``False`` + warning) when
    ``HF_TOKEN`` is absent. Retries the upload commit up to 3 total attempts on
    a commit-conflict shape (concurrent sprite commits).

    Returns:
        True on success, False when skipped or failed.
    """
    from daydream.ui import create_console, print_warning

    if not os.environ.get("HF_TOKEN"):
        print_warning(
            create_console(),
            f"Skip HF upload of {session_id}: HF_TOKEN not set (set it to upload run bundles)",
        )
        return False

    global HfApi
    if HfApi is None:
        try:
            import huggingface_hub  # noqa: PLC0415 - optional dep, keep lazy

            HfApi = huggingface_hub.HfApi
        except ImportError:
            print_warning(
                create_console(),
                "Skip HF upload: huggingface_hub not installed (pip install huggingface-hub)",
            )
            return False

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

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
            conflict = "concurrent update" in message or "revision" in message
            if conflict and attempt < 3:
                continue
            print_warning(
                create_console(),
                f"HF upload of {session_id} to {repo_id} failed (non-fatal): {message}",
            )
            return False
    return False
