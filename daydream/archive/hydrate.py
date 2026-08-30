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

import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from daydream.archive import sanitize
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.hydrate_rules import (
    REASON_CODE_BUNDLE_UNREADABLE,
    REASON_CODE_SANITIZE_FAILED,
    REASON_CODE_SECRETS_SCAN_DIRTY,
    REASON_CODE_UNTRUSTED_REMOTE_HOST,
)
from daydream.archive.index import upsert_run
from daydream.archive.manifest import Manifest
from daydream.trajectory import redact_text

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{4,39}$")


class HydrationError(Exception):
    """Base class for every hydrate failure mode; the orchestrator decides."""


class HubUnavailableError(HydrationError):
    """The ``huggingface_hub`` extra or its ``HF_TOKEN`` prerequisite is missing."""


class HubDownloadError(HydrationError):
    """A requested path/revision does not exist in the Hub repo (fail-closed)."""


class StageError(HydrationError):
    """Staging the pinned snapshot failed (download, digest, or path violation)."""


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


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one :func:`download_snapshot` pass over the pinned revision."""

    downloaded: int = 0
    skipped: int = 0
    digests: dict[str, str] = field(default_factory=dict)  # relpath -> sha256


def _validate_relpath(relpath: str, root: Path) -> Path:
    """Enforce the M4 trust boundary: resolve ``relpath`` strictly under ``root``.

    Raises :class:`StageError` naming "traversal" when the path is absolute,
    carries ``..`` segments, or otherwise escapes the staging root. Nothing is
    ever written before this check passes.
    """
    p = PurePosixPath(relpath)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise StageError(
            redact_text(
                f"refusing relpath {relpath!r}: traversal — path escapes the staging root"
            )
        )
    target = (root / relpath).resolve()
    if not target.is_relative_to(root.resolve()):
        raise StageError(
            redact_text(
                f"refusing relpath {relpath!r}: traversal — resolved path escapes the staging root"
            )
        )
    return target


def download_snapshot(
    client: HubClient,
    *,
    revision: str,
    stage_dir: Path,
    expect: dict[str, str] | None = None,
) -> DownloadResult:
    """Resumable, content-addressed download of a pinned snapshot revision (issue #982 M3).

    Lists the repo's bundle content (paths under ``bundles/``), writes each
    artifact to ``stage_dir/<revision>/<relpath>``, and records a per-artifact
    ledger (relpath, sha256, size, fetched_at) in
    ``stage_dir/<revision>/_download_manifest.json``.

    Resume: an on-disk artifact whose sha256 matches the existing ledger record
    is skipped, not re-downloaded; missing or mismatched artifacts are fetched,
    re-hashed, and their records updated. ``expect`` — ``{relpath: sha256}``
    from a pinned manifest — makes any disagreement a hard :class:`StageError`
    naming "digest". Every relpath is validated against the staging root before
    any write (see :func:`_validate_relpath`); download failures delete the
    partial artifact and raise :class:`StageError` with redacted messages.
    """
    revision = str(revision)
    root = stage_dir / revision
    manifest_path = root / "_download_manifest.json"
    records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            records = {a["relpath"]: a for a in loaded.get("artifacts", [])}
        except (OSError, ValueError, KeyError, TypeError):
            records = {}  # corrupt ledger: rebuild from disk state

    relpaths = list(client.list_repo_files())
    downloaded = 0
    skipped = 0
    digests: dict[str, str] = {}
    artifacts: list[dict[str, Any]] = []

    for relpath in relpaths:
        target = _validate_relpath(relpath, root)
        if not relpath.startswith("bundles/"):
            continue  # non-bundle paths (top-level manifests, etc.) are not staged here
        expected_sha = (expect or {}).get(relpath)
        try:
            if target.exists():
                existing = hashlib.sha256(target.read_bytes()).hexdigest()
                record_sha = records.get(relpath, {}).get("sha256")
                if expected_sha in (None, existing) and record_sha in (None, existing):
                    digests[relpath] = existing
                    skipped += 1
                    artifacts.append({**records.get(relpath, {}), "relpath": relpath, "sha256": existing})
                    continue
            data = client.download_file(relpath, revision=revision)
        except HydrationError as exc:
            raise StageError(redact_text(str(exc))) from exc
        sha = hashlib.sha256(data).hexdigest()
        if expected_sha is not None and sha != expected_sha:
            raise StageError(
                redact_text(f"digest mismatch for {relpath!r}: expected {expected_sha}, got {sha}")
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        try:
            tmp.write_bytes(data)
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StageError(redact_text(f"write failed for {relpath!r}: {exc}")) from exc
        downloaded += 1
        digests[relpath] = sha
        artifacts.append(
            {
                "relpath": relpath,
                "sha256": sha,
                "size": len(data),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"revision": revision, "artifacts": artifacts}, indent=2))
    return DownloadResult(downloaded=downloaded, skipped=skipped, digests=digests)


def _read_manifest_dict(bundle_dir: Path) -> dict[str, Any] | None:
    """Read ``manifest.json`` from ``bundle_dir``; ``None`` when absent/unparseable."""
    path = bundle_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class IngestResult:
    """Outcome of the ingest gate for one staged session bundle (issue #982 M4/M6)."""

    session_id: str
    status: str  # "admitted" | "quarantined"
    reason_code: str | None = None


def ingest_bundles(stage: Path, *, revision: str) -> list[IngestResult]:
    """Run every staged bundle through the #981 ingest gate (issue #982 M4/M6).

    For each session bundle under ``stage/downloads/<revision>/bundles/``:

    1. The manifest must parse and carry a trusted remote host
       (:func:`daydream.archive.git_safe.normalize_remote_url` identity); an
       unreadable manifest or non-allowlisted host is quarantined (fail-closed,
       stable reason codes) without ever dereferencing embedded paths.
    2. ``sanitize.import_bundle`` is the sole secrets gate: a dirty bundle is
       moved to ``stage/quarantine/<name>`` by the #981 implementation itself —
       hydrate never forks the scan or the quarantine move.
    3. A clean bundle is sanitized by ``sanitize.sanitize_bundle`` (release
       scan fail-closed), and the released derivative is placed at
       ``stage/runs/<session_id>/``. Any ``.git`` directory inside the
       downloaded copy is stripped before sanitize so no hydrated bundle ever
       ships one (harvest priority-1 safety, Task 0B constraint).

    Identity-collision dedupe is Task 8's concern; this pass guarantees one
    derivative per staged bundle or a quarantine result.
    """
    bundles_root = stage / "downloads" / str(revision) / "bundles"
    if not bundles_root.is_dir():
        return []
    results: list[IngestResult] = []
    for bundle_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        name = bundle_dir.name
        data = _read_manifest_dict(bundle_dir)
        if data is None:
            results.append(IngestResult(name, "quarantined", REASON_CODE_BUNDLE_UNREADABLE))
            continue
        session_id = str(data.get("session_id") or name)
        raw_url = data.get("remote_url")
        if isinstance(raw_url, str) and raw_url.strip():
            identity, _canonical = normalize_remote_url(raw_url)
            if identity is None:
                # Non-allowlisted host: rejected as admission data before any
                # gate work; the raw copy stays in downloads, never indexed.
                results.append(
                    IngestResult(session_id, "quarantined", REASON_CODE_UNTRUSTED_REMOTE_HOST)
                )
                continue
        gate = sanitize.import_bundle(bundle_dir, stage)
        if gate.quarantined or not gate.imported:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SECRETS_SCAN_DIRTY))
            continue
        # Task 0B constraint: hydrated staging bundles must exclude .git (the
        # raw download copy is daydream-staged data, safe to prune locally).
        for git_dir in list(bundle_dir.rglob(".git")):
            if git_dir.is_dir():
                shutil.rmtree(git_dir)
        try:
            sanitized = sanitize.sanitize_bundle(bundle_dir, stage)
        except Exception:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SANITIZE_FAILED))
            continue
        if not sanitized.released:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SECRETS_SCAN_DIRTY))
            continue
        derivative = stage / "sanitized" / session_id
        target = stage / "runs" / session_id
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(derivative), str(target))  # staging layout only, not the gate
        results.append(IngestResult(session_id, "admitted"))
    return results


def _staging_local_source_path(raw: Any, stage: Path) -> str | None:
    """Rewrite an embedded ``source_path`` to a staging-local value or ``None``.

    Embedded paths are data: an absolute path outside the staging root is
    dropped (never dereferenced, never carried into the index).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw)
    if not p.is_absolute():
        return raw
    try:
        p.relative_to(stage)
    except ValueError:
        return None
    return raw


def rebuild_index(stage: Path) -> None:
    """Index every admitted derivative under ``stage/runs/`` (issue #982 M6).

    Each derivative's manifest is loaded and its path/URL fields rewritten to
    staging-local, credential-free values before ``index.upsert_run`` writes
    the row: ``archive_path`` points inside the staging root (never the raw
    download tree), ``source_path`` is staging-local or ``None``, and
    ``remote_url``/``repo_slug`` come from ``normalize_remote_url`` output.
    ``upsert_run`` errors propagate — an admitted bundle is never silently
    skipped.
    """
    runs_dir = stage / "runs"
    if not runs_dir.is_dir():
        return
    for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        data = _read_manifest_dict(derivative)
        if data is None:
            raise HydrationError(
                redact_text(f"admitted derivative {derivative.name} has an unreadable manifest")
            )
        valid = {f.name for f in dataclass_fields(Manifest)}
        rewritten = {"archive_path", "source_path", "remote_url", "repo_slug"}
        kwargs = {k: v for k, v in data.items() if k in valid and k not in rewritten}
        raw_url = data.get("remote_url")
        if isinstance(raw_url, str) and raw_url.strip():
            slug, canonical = normalize_remote_url(raw_url)
            kwargs["repo_slug"] = slug
            kwargs["remote_url"] = canonical
        else:
            kwargs["repo_slug"] = None
            kwargs["remote_url"] = None
        kwargs["source_path"] = _staging_local_source_path(data.get("source_path"), stage)
        kwargs["archive_path"] = str(derivative)
        upsert_run(stage, Manifest(**kwargs))


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
