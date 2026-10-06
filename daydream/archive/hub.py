"""Opt-in HuggingFace dataset upload of completed run bundles.

Destination resolution and upload admission live here. Upload failures preserve
the local run; ``huggingface_hub`` is loaded only when upload is requested.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from daydream.archive._console import warn as _warn

if TYPE_CHECKING:
    from daydream.run_config import RunConfig

# Lazy optional dependency.
HfApi: type | None = None

# Exponential commit-conflict backoff, capped at 120 seconds.
_UPLOAD_RETRY_BASE_DELAY_S = 2.0
_UPLOAD_RETRY_MAX_DELAY_S = 120.0


def resolve_hub_repo(config: RunConfig) -> str | None:
    """Resolve the dataset ID from CLI config, then ``DAYDREAM_TRAJECTORY_HUB_REPO``.

    Empty values are unset; target-checkout file config cannot select a destination.
    """
    if config.trajectory_hub_repo:
        return config.trajectory_hub_repo
    env = os.environ.get("DAYDREAM_TRAJECTORY_HUB_REPO")
    if env:
        return env
    return None


def upload_run_bundle(run_dir: Path, repo_id: str, session_id: str) -> bool:
    """Upload the complete bundle to dataset ``repo_id`` under ``session_id``.

    Return True on success; skips and failures warn and return False. Missing
    ``HF_TOKEN`` or the optional Hub dependency skips upload. New repos are private;
    existing visibility is retained, with a warning before uploading to a public repo.
    Blocking credentials always refuse upload.
    Scanner errors always refuse upload with value-free diagnostics;
    advisory name/template matches are reported and allow upload. Concurrent commit
    conflicts retry up to three total attempts with exponential backoff.
    """
    if not os.environ.get("HF_TOKEN"):
        _warn(f"Data Collection: skip HF upload of {session_id}: HF_TOKEN not set (set it to upload run bundles)")
        return False

    global HfApi
    if HfApi is None:
        try:
            import huggingface_hub  # noqa: PLC0415 - optional dep, keep lazy

            HfApi = huggingface_hub.HfApi
        except ImportError:
            _warn("Data Collection: skip HF upload: huggingface_hub not installed (pip install huggingface-hub)")
            return False

    from daydream.archive import scan

    try:
        scan_result = scan.scan_run_dir(run_dir)
    except Exception:  # noqa: BLE001 - scanner failures never release payloads or exception text
        _warn(f"Data Collection: refusing HF upload of {session_id}: bundle scanner failed")
        return False
    scan_failed = (
        any(f.category == "scan_error" for f in scan_result.findings)
        or (not scan_result.clean and not scan_result.findings)
    )
    if scan_failed or scan_result.blocking:
        _warn(
            f"Data Collection: refusing HF upload of {session_id}: bundle secret scan found "
            f"problems ({scan_result.summary()})"
        )
        return False
    if scan_result.findings:
        # Advisory-only: a name/template shape, not a credential. Reported
        # value-free so the operator sees it, never silently swallowed.
        _warn(
            f"HF upload of {session_id} proceeding with advisory-only scan "
            f"findings ({scan_result.summary()})"
        )

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(repo_id=repo_id, repo_type="dataset").private:
            _warn(
                f"HF repo {repo_id} already exists and is public; the run bundle for "
                f"{session_id} will be uploaded to a public repo"
            )
    except Exception as exc:  # noqa: BLE001 - absorb, the run must not fail
        _warn(f"Data Collection: HF upload of {session_id} failed ({type(exc).__name__})")
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
            _warn(f"Data Collection: HF upload of {session_id} failed ({type(exc).__name__})")
            break
    return False
