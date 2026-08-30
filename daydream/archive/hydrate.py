"""Hub hydration for issue #982 (task 1: client seam).

Defines the :class:`HubClient` protocol that isolates ``huggingface_hub`` behind
a narrow surface (list/download/commit/repo-info), the lazy production adapter
:class:`HfHubClient`, and the fatal :class:`HubUnavailableError` raised when the
optional ``hub`` extra (or its ``HF_TOKEN``) is missing. Unlike
``daydream.archive.hub`` — whose upload callback must never fail a run and
therefore warns — hydration is an explicit operator command, so every unmet
prerequisite is fatal and fail-closed.

``huggingface_hub`` is imported only inside :func:`_import_hf_hub`; production
code and tests that use :class:`~daydream.archive.hydrate_client.FakeHub` never
need it installed. The module-level :func:`_make_client` factory is the
monkeypatch seam used by tests.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class HydrationError(Exception):
    """Base class for every hydrate failure mode; the orchestrator decides."""


class HubUnavailableError(HydrationError):
    """The ``huggingface_hub`` extra or its ``HF_TOKEN`` prerequisite is missing."""


class HubDownloadError(HydrationError):
    """A requested path/revision does not exist in the Hub repo (fail-closed)."""


@dataclass(frozen=True)
class RepoInfo:
    """Minimal repo metadata the hydration flow needs."""

    sha: str
    private: bool


@runtime_checkable
class HubClient(Protocol):
    """Narrow Hub surface hydration depends on (list/download/commit/repo-info)."""

    def repo_info(self, revision: str | None = None) -> RepoInfo: ...

    def list_repo_files(self) -> list[str]: ...

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes: ...

    def upload_files(
        self, mapping: dict[str | Path, Path], commit_message: str
    ) -> None: ...

    @property
    def repo_private(self) -> bool: ...


def _import_hf_hub() -> Any:
    """Return the ``huggingface_hub`` module, or ``None`` when not installed.

    Kept as a module-level function so tests can monkeypatch it to ``None`` to
    simulate the missing-extra environment.
    """
    if importlib.util.find_spec("huggingface_hub") is None:
        return None
    import huggingface_hub  # noqa: PLC0415  # lazy: optional extra

    return huggingface_hub


def _make_client(repo_id: str, *, token_present: bool | None = None) -> HfHubClient:
    """Build the production :class:`HfHubClient` for ``repo_id``.

    ``token_present`` overrides the ``HF_TOKEN`` environment check when
    explicitly given (``None`` derives it from the environment). Raises
    :class:`HubUnavailableError` — fatally, never a warning — when either the
    package or the token is absent. Error messages name prerequisites only;
    token material is never echoed.
    """
    if token_present is False:
        raise HubUnavailableError(
            "HF_TOKEN is not set; hydration requires a read token for the "
            "private Hub repo. Export HF_TOKEN (or pass --token-source) and retry."
        )
    if _import_hf_hub() is None:
        raise HubUnavailableError(
            "The 'huggingface-hub' package is required for hydrate but is not "
            "installed. Install the optional extra: `uv sync --extra hub` "
            "(or `pip install 'daydream[hub]'`)."
        )
    if token_present is None:
        token_present = bool(os.environ.get("HF_TOKEN"))
    if not token_present:
        raise HubUnavailableError(
            "HF_TOKEN is not set; hydration requires a read token for the "
            "private Hub repo. Export HF_TOKEN and retry."
        )
    return HfHubClient(repo_id)


class HfHubClient:
    """Production :class:`HubClient` adapter over a lazily imported ``huggingface_hub``.

    The token is read from ``os.environ["HF_TOKEN"]`` only and is never stored
    on argv, logged, or included in ``repr``.
    """

    def __init__(self, repo_id: str) -> None:
        hf = _import_hf_hub()
        if hf is None:
            raise HubUnavailableError(
                "The 'huggingface-hub' package is required for hydrate but is not "
                "installed. Install the optional extra: `uv sync --extra hub`."
            )
        self._repo_id = repo_id
        self._hf: Any = hf
        self._api: Any = hf.HfApi(token=os.environ.get("HF_TOKEN"))

    def __repr__(self) -> str:  # never leaks the token
        return f"HfHubClient(repo_id={self._repo_id!r})"

    @property
    def repo_private(self) -> bool:
        return bool(self.repo_info().private)

    def repo_info(self, revision: str | None = None) -> RepoInfo:
        try:
            info = self._api.repo_info(revision or "main", repo_type="dataset")
        except Exception as exc:  # mapped, never swallowed
            raise HubDownloadError(f"repo_info failed for {self._repo_id}: {exc}") from exc
        return RepoInfo(sha=str(info.sha), private=bool(info.private))

    def list_repo_files(self) -> list[str]:
        try:
            return list(self._api.list_repo_files(self._repo_id, repo_type="dataset"))
        except Exception as exc:
            raise HubDownloadError(f"list_repo_files failed for {self._repo_id}: {exc}") from exc

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
        try:
            local = self._api.hf_hub_download(
                self._repo_id, path_in_repo, repo_type="dataset", revision=revision
            )
        except Exception as exc:
            raise HubDownloadError(
                f"download failed for {self._repo_id}:{path_in_repo}: {exc}"
            ) from exc
        return Path(local).read_bytes()

    def upload_files(self, mapping: dict[str | Path, Path], commit_message: str) -> None:
        try:
            for path_in_repo, local_path in mapping.items():
                self._api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=str(path_in_repo),
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    commit_message=commit_message,
                )
        except Exception as exc:
            raise HydrationError(f"upload failed for {self._repo_id}: {exc}") from exc
