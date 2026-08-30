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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from daydream.trajectory import redact_text

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{4,39}$")


class HydrationError(Exception):
    """Base class for every hydrate failure mode; the orchestrator decides."""


class HubUnavailableError(HydrationError):
    """The ``huggingface_hub`` extra or its ``HF_TOKEN`` prerequisite is missing."""


class HubDownloadError(HydrationError):
    """A requested path/revision does not exist in the Hub repo (fail-closed)."""


class MovingBranchError(HydrationError):
    """A symbolic ref (moving branch/tag) was requested without ``exploratory=True``."""


def resolve_source_revision(client: HubClient, revision: str, *, exploratory: bool) -> str:
    """Resolve ``revision`` to a pinned, immutable commit SHA (issue #982 M2).

    - A full 40-hex SHA is verified to exist and returned unchanged.
    - A hex short prefix (4-39 chars) resolves to the unique matching commit
      SHA; an ambiguous or unknown prefix is a fail-closed :class:`HydrationError`.
    - Any other name is a symbolic ref (moving branch/tag) and raises
      :class:`MovingBranchError` — naming the ref and ``exploratory`` — unless
      ``exploratory=True``, in which case it resolves to the ref's current SHA.
      Exploratory output is flagged non-canonical downstream; canonical v1 runs
      must pin an exact SHA.

    Client errors are redacted via ``daydream.trajectory.redact_text`` before
    being re-raised as :class:`HydrationError`, so no credential material ever
    reaches the console or ledger.
    """
    revision = revision.strip().lower()
    if _FULL_SHA_RE.fullmatch(revision):
        try:
            client.repo_info(revision=revision)  # verify it exists
        except HydrationError as exc:
            raise HydrationError(redact_text(str(exc))) from exc
        return revision

    if _HEX_PREFIX_RE.fullmatch(revision):
        list_revisions = getattr(client, "list_revisions", None)
        matches = (
            [r for r in list_revisions() if r.startswith(revision)]
            if callable(list_revisions)
            else []
        )
        if len(matches) > 1:
            raise HydrationError(
                redact_text(f"ambiguous revision prefix {revision!r}: {len(matches)} matching commits")
            )
        if len(matches) == 1:
            return str(matches[0])
        # Not a known prefix — fall through to symbolic-ref resolution so a
        # hex-named ref still gets the moving-branch treatment below.

    try:
        info = client.repo_info(revision=revision)
    except HydrationError as exc:
        raise HydrationError(redact_text(f"unknown revision {revision!r}: {exc}")) from exc
    if not exploratory:
        raise MovingBranchError(
            f"ref {revision!r} is a moving branch/tag, not a pinned commit; pass "
            "exploratory=True to accept it (output is non-canonical), or pin an "
            "exact 40-char commit SHA"
        )
    return info.sha


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

    def list_revisions(self) -> list[str]:
        """Commit SHAs known to the Hub, for short-prefix resolution (best effort)."""
        try:
            commits = self._api.list_repo_commits(self._repo_id, repo_type="dataset")
            return [str(c.commit_id) for c in commits]
        except Exception as exc:  # enumeration is best-effort; fail closed on use
            raise HubDownloadError(f"list_repo_commits failed for {self._repo_id}: {exc}") from exc

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
