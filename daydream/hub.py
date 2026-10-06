"""Optional Hugging Face boundary for private, commit-pinned JSONL datasets."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from daydream.run_config import RunConfig

_REVISION = re.compile(r"[0-9a-f]{40}")


class HubError(ValueError):
    """A value-free diagnostic safe to persist in the local upload queue."""

    def __init__(self, code: str) -> None:
        self.code = code if re.fullmatch(r"[a-z][a-z0-9_]*", code) else "network_failed"
        super().__init__(self.code)


class HubConflict(Exception):
    """The branch changed before a guarded commit could be installed."""


class DatasetHub(Protocol):
    """The only remote operations needed by the JSONL queue and reader."""

    def private_revision(self, repo_id: str) -> str: ...

    def read_file(self, repo_id: str, path: str, revision: str) -> bytes | None: ...

    def commit(self, repo_id: str, files: Mapping[str, bytes], parent_revision: str) -> str: ...


def _revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise HubError("invalid_revision")
    return value


def _load_hf() -> tuple[Any, Any]:
    try:
        import huggingface_hub  # noqa: PLC0415 - optional, only explicit Hub work imports it
        from huggingface_hub import errors  # noqa: PLC0415
    except ImportError:
        raise HubError("missing_hub_dependency") from None
    return huggingface_hub, errors


class HfDatasetHub:
    """One guarded HF tree commit; credentials remain inside the SDK boundary."""

    def __init__(self) -> None:
        self._hf, self._errors = _load_hf()
        try:
            self._api = self._hf.HfApi(token=os.environ.get("HF_TOKEN"))
        except Exception:
            raise HubError("network_failed") from None

    def private_revision(self, repo_id: str) -> str:
        try:
            info = self._api.repo_info(repo_id=repo_id, repo_type="dataset", revision="main")
        except Exception:
            raise HubError("network_failed") from None
        if getattr(info, "private", None) is not True:
            raise HubError("public_destination")
        return _revision(getattr(info, "sha", None))

    def read_file(self, repo_id: str, path: str, revision: str) -> bytes | None:
        revision = _revision(revision)
        try:
            local = self._api.hf_hub_download(
                repo_id=repo_id, filename=path, repo_type="dataset", revision=revision,
            )
        except self._errors.RemoteEntryNotFoundError:
            return None
        except Exception:
            raise HubError("network_failed") from None
        try:
            # HF's normal snapshot cache uses symlinks to immutable blob files.
            source = Path(local).resolve(strict=True)
            if not source.is_file():
                raise OSError("not a regular file")
            return source.read_bytes()
        except Exception:
            raise HubError("network_failed") from None

    def commit(self, repo_id: str, files: Mapping[str, bytes], parent_revision: str) -> str:
        parent_revision = _revision(parent_revision)
        try:
            operations = [self._hf.CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)
                          for path, data in sorted(files.items())]
        except Exception:
            raise HubError("network_failed") from None
        try:
            result = self._api.create_commit(
                repo_id=repo_id, repo_type="dataset", operations=operations,
                commit_message="daydream JSONL snapshot", revision="main", parent_commit=parent_revision,
                create_pr=False, run_as_future=False,
            )
        except Exception as error:
            if (isinstance(error, self._errors.HfHubHTTPError)
                    and getattr(getattr(error, "response", None), "status_code", None) == 412):
                raise HubConflict("concurrent_update") from None
            raise HubError("network_failed") from None
        return _revision(getattr(result, "oid", None))


def resolve_hub_repo(config: RunConfig) -> str | None:
    """Select an operator destination from CLI config, then the environment.

    Empty values are unset. Credentials and target-checkout config cannot enable
    publication or choose its destination.
    """
    return config.trajectory_hub_repo or os.environ.get("DAYDREAM_TRAJECTORY_HUB_REPO") or None
