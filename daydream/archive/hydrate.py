"""Hydrate pinned Hub snapshots into private, verified curated corpora.

Admission, license decisions, content identity, and remote checkpoints are
persisted separately. Publication remains incomplete until clean-room verification
succeeds. Hub dependencies are optional and loaded through ``_make_client``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from daydream.archive import hydrate_rules, sanitize
from daydream.archive._console import warn as _warn
from daydream.archive.hydrate_admission import (
    _iter_enrichment_cache,
    _session_identity,
    admission_summary_buckets as admission_summary_buckets,
    apply_license_gate as apply_license_gate,
    build_import_ledger as build_import_ledger,
    build_resolution_map as build_resolution_map,
    dedupe_admitted as dedupe_admitted,
    license_admission_by_repo as license_admission_by_repo,
    license_admission_summary as license_admission_summary,
    rebuild_index as rebuild_index,
    resolve_curation_identity as resolve_curation_identity,
    restamp_admitted_digests as restamp_admitted_digests,
)
from daydream.archive.hydrate_discovery import (
    _is_bare_segment,
    _validate_relpath,
)
from daydream.archive.hydrate_stage import (
    _bundle_dirs,
    _manifest_remote_fields,
    _read_manifest_dict,
    download_snapshot as download_snapshot,
    ingest_bundles as ingest_bundles,
)
from daydream.archive.hydrate_types import (
    DedupeResult as DedupeResult,
    DownloadResult as DownloadResult,
    HubClient as HubClient,
    HubConcurrentUpdateError as HubConcurrentUpdateError,
    HubDownloadError as HubDownloadError,
    HubUnavailableError as HubUnavailableError,
    HydrateHubConfig as HydrateHubConfig,
    HydrateSummary as HydrateSummary,
    HydrationError as HydrationError,
    IngestResult as IngestResult,
    MovingBranchError as MovingBranchError,
    NoSessionCandidatesError as NoSessionCandidatesError,
    PublicDestinationError as PublicDestinationError,
    RepoInfo as RepoInfo,
    ResumeState as ResumeState,
    StageError as StageError,
    VerificationError as VerificationError,
)
from daydream.archive.index import manifest_index_fields, query_runs, upsert_run
from daydream.archive.scan import scan_run_dir
from daydream.json_utils import atomic_write_json
from daydream.redaction import redact_text
from daydream.timeutil import now_iso_utc
from daydream.trajectory import RUNS_DIRNAME

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{4,39}$")
ANNOTATION_BRANCH = "main"

def resolve_source_revision(client: HubClient, revision: str, *, exploratory: bool) -> str:
    """Pin a verified full SHA, unique hex prefix, or explicitly allowed moving ref."""
    revision = revision.strip()
    if _FULL_SHA_RE.fullmatch(revision.lower()):
        # Normalize hex only; symbolic refs remain case-sensitive.
        revision = revision.lower()
        try:
            client.repo_info(revision=revision)  # verify it exists
        except HydrationError as exc:
            raise HydrationError(redact_text(str(exc))) from exc
        return revision

    if _HEX_PREFIX_RE.fullmatch(revision.lower()):
        prefix = revision.lower()
        list_revisions = getattr(client, "list_revisions", None)
        matches = (
            [r for r in list_revisions() if r.startswith(prefix)]
            if callable(list_revisions)
            else []
        )
        if len(matches) > 1:
            raise HydrationError(
                redact_text(f"ambiguous revision prefix {prefix!r}: {len(matches)} matching commits")
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


_UPLOAD_ATTEMPTS = 6
_UPLOAD_BASE_DELAY_S = 2.0
_UPLOAD_MAX_DELAY_S = 120.0


def _retry_upload(client: HubClient, mapping: dict[str | Path, Path], commit_message: str) -> None:
    """Upload with the hub.py commit-conflict retry shape (exponential backoff)."""
    for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
        try:
            client.upload_files(mapping, commit_message)
            return
        except HydrationError as exc:
            conflict = "concurrent update" in str(exc)
            if conflict and attempt < _UPLOAD_ATTEMPTS:
                time.sleep(min(_UPLOAD_BASE_DELAY_S * (2 ** (attempt - 1)), _UPLOAD_MAX_DELAY_S))
                continue
            raise HydrationError(redact_text(str(exc))) from exc


def _curated_upload_paths(stage: Path, curation_id: str) -> list[Path]:
    """All staging files under ``stage/curated/<curation-id>/`` (relative, sorted)."""
    curated = stage / "curated" / curation_id
    if not curated.is_dir():
        return []
    return sorted(p for p in curated.rglob("*") if p.is_file())


def _policy_binding_record(binding: dict[str, Any]) -> str:
    """Canonical JSON with sorted keys, casefolded opt-ins, all binding digests, and a trailing newline."""
    record = {
        "policy_digest": str(binding["policy_digest"]),
        "policy_version": str(binding["policy_version"]),
        "allow_copyleft": sorted(str(slug).casefold() for slug in binding["allow_copyleft"]),
        "exclusions_digest": str(binding["exclusions_digest"]),
        "resolved_decisions_digest": str(binding["decisions_digest"]),
        "distribution_digest": str(binding["distribution_digest"]),
        "schema_version": "2",
    }
    return json.dumps(record, sort_keys=True) + "\n"


def check_prefix_binding(
    client: HubClient, *, curation_id: str, binding: dict[str, Any],
    allow_unbound_resume: bool = False,
) -> None:
    """Reject conflicting or legacy-unbound published prefixes before any upload."""
    prefix = f"curated/{curation_id}/"
    current = _policy_binding_record(binding).encode("utf-8")
    try:
        remote = client.download_file(f"{prefix}policy-binding.json")
    except HubDownloadError:
        remote = None  # no binding record yet (fresh or legacy prefix)
    except HydrationError as exc:
        raise HydrationError(
            redact_text(f"cannot read the published policy binding under {prefix}: {exc}")
        ) from exc
    if remote is not None:
        if remote == current:
            return
        detail = ""
        try:
            remote_digest = json.loads(remote.decode("utf-8")).get("policy_digest", "?")
            detail = f" (remote policy_digest={remote_digest}, current policy_digest={binding['policy_digest']})"
        except (ValueError, UnicodeDecodeError):
            detail = " (remote record is not a readable binding record)"
        raise HydrationError(
            redact_text(
                f"conflicting policy binding under {prefix}: the prefix was "
                "published under a different policy; refuse to publish "
                f"(fail-closed){detail}"
            )
        )
    # Absent record: fresh prefix, interrupted v2 run, or pre-v2 legacy prefix.
    repo_files = client.list_repo_files()
    has_batches = any(p.startswith(f"{prefix}batches/") for p in repo_files)
    if not has_batches:
        return  # fresh prefix: nothing published yet
    has_ledger = f"{prefix}resume/ledger.jsonl" in set(repo_files)
    if allow_unbound_resume and has_ledger:
        return  # interrupted v2 run; finalize is about to publish the record
    raise HydrationError(
        redact_text(
            f"conflicting policy binding under {prefix}: the prefix has "
            "published batches but no policy-binding record (pre-v2 legacy "
            "prefix); refuse to publish (fail-closed)"
        )
    )


def publish_batches(
    client: HubClient, stage: Path, *, curation_id: str, skip_sessions: set[str] | None = None
) -> None:
    """Publish sanitized batches and supporting ledgers under a private curated prefix."""
    if not client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; hydration "
            "publishes sanitized corpora only to private repos (M17)"
        )
    curated = stage / "curated" / curation_id
    _stage_batches(stage, curated)
    _write_resolution_map(stage, curated)
    _write_resume_ledger(curated, curation_id)
    files = _curated_upload_paths(stage, curation_id)
    if not files:
        raise HydrationError(redact_text(f"nothing to publish under curated/{curation_id}"))

    prefix = f"curated/{curation_id}/"
    relpaths = sorted(f.relative_to(curated).as_posix() for f in files)
    # Bronze safety gate: assert nothing escapes the curated prefix (M10/M13).
    assert all(not p.startswith(("bronze", f"{RUNS_DIRNAME}/", "downloads/")) and ".." not in p
               for p in relpaths), relpaths

    _write_sha256sums(curated, prefix, relpaths)
    files = _curated_upload_paths(stage, curation_id)

    mapping: dict[str | Path, Path] = {
        f"{prefix}{f.relative_to(curated).as_posix()}": f
        for f in files
        if not any(
            f.relative_to(curated).as_posix().startswith(f"batches/{sid}/")
            for sid in skip_sessions or ()
        )
    }
    if mapping:
        _retry_upload(client, mapping, f"daydream hydrate {curation_id}: additive batch publication")


def _stage_batches(stage: Path, curated: Path) -> None:
    """Copy every admitted derivative (``stage/runs/<sid>``) into ``batches/<sid>/``."""
    for derivative in _bundle_dirs(stage / RUNS_DIRNAME):
        target = curated / "batches" / derivative.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)  # content-addressed rewrite: same content, same digest
        shutil.copytree(derivative, target)


def _repo_commits_from_enrichment_cache(stage: Path) -> dict[str, str]:
    """Latest resolved full repository SHA per slug from enrichment, never the Hub SHA."""
    commits: dict[str, str] = {}
    for entry in _iter_enrichment_cache(stage):
        if entry.get("status") != "resolved":
            continue
        slug = entry.get("repo_slug")
        commit = entry.get("repo_commit")
        if isinstance(slug, str) and isinstance(commit, str) and _FULL_SHA_RE.fullmatch(commit):
            commits[slug] = commit
    return commits


def _curation_source_commit(curated: Path) -> str | None:
    """Read ``source_commit`` from the curated import ledger, or ``None``."""
    ledger_path = curated / "import-ledger.json"
    if not ledger_path.is_file():
        return None
    try:
        source_commit: str | None = json.loads(ledger_path.read_text(encoding="utf-8")).get("source_commit")
    except (OSError, ValueError):
        return None
    return source_commit


def _write_resolution_map(stage: Path, curated: Path) -> None:
    """Write the resolution map once, preserving prior publication bytes on resume."""
    map_path = curated / "resolution-map.json"
    if map_path.exists():
        return
    cmap = build_resolution_map(
        stage,
        repo_commits=_repo_commits_from_enrichment_cache(stage),
    )
    atomic_write_json(map_path, cmap)


def _write_resume_ledger(curated: Path, curation_id: str) -> None:
    """Write (append) one resume record per admitted batch under ``resume/ledger.jsonl``."""
    source_commit = _curation_source_commit(curated)
    entries = {
        batch.name: {
            "session_id": batch.name,
            "batch_digest": sanitize._derivative_digest(batch),
            "source_commit": source_commit,
            "curation_id": curation_id,
            "at": now_iso_utc(),
        }
        for batch in _bundle_dirs(curated / "batches")
    }
    resume_path = curated / "resume" / "ledger.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if resume_path.is_file():
        try:
            for line in resume_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("session_id"):
                        existing[str(rec["session_id"])] = rec
        except (OSError, ValueError):
            existing = {}
    # Additive ledger: first record per session wins; re-publishing identical
    # content never rewrites history (content-addressed idempotence).
    for sid, entry in entries.items():
        existing.setdefault(sid, entry)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path.write_text(
        "".join(json.dumps(existing[sid], sort_keys=True) + "\n" for sid in sorted(existing)),
        encoding="utf-8",
    )


def resume_state(client: HubClient, *, curation_id: str, stage_dir: Path) -> ResumeState:
    """Verify remote checkpoint batch digests into staging, without trusting local state."""
    prefix = f"curated/{curation_id}/"
    try:
        raw = client.download_file(f"{prefix}resume/ledger.jsonl")
    except HydrationError:
        raw = None  # no checkpoint yet: nothing completed, nothing redownloaded
    completed: set[str] = set()
    redownloaded: list[str] = []
    if raw is None:
        return ResumeState(completed_sessions=completed, redownloaded=redownloaded)
    repo_files = set(client.list_repo_files())
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise HydrationError(redact_text(f"corrupt resume ledger entry: {exc}")) from exc
        sid = str(entry.get("session_id") or "")
        digest = str(entry.get("batch_digest") or "")
        if not sid or not digest:
            continue
        batch_prefix = f"{prefix}batches/{sid}/"
        batch_relpaths = sorted(p[len(batch_prefix):] for p in repo_files if p.startswith(batch_prefix))
        if not batch_relpaths:
            redownloaded.append(sid)
            continue
        if not _is_bare_segment(sid):
            raise StageError(redact_text(f"refusing resume session id {sid!r}: traversal"))
        batch_dir = stage_dir / "batches" / sid
        for rel in batch_relpaths:
            data = client.download_file(f"{batch_prefix}{rel}")
            target = _validate_relpath(rel, batch_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        local_digest = sanitize._derivative_digest(batch_dir)
        if local_digest == digest:
            completed.add(sid)
        else:
            redownloaded.append(sid)
    return ResumeState(completed_sessions=completed, redownloaded=redownloaded)


def _import_hf_hub() -> Any:
    """Return the ``huggingface_hub`` module, or ``None`` when not installed."""
    if importlib.util.find_spec("huggingface_hub") is None:
        return None
    import huggingface_hub  # noqa: PLC0415  # lazy: optional extra

    return huggingface_hub


def _write_sha256sums(curated: Path, prefix: str, relpaths: Iterable[str]) -> None:
    """Write ``SHA256SUMS`` over ``relpaths`` (excluding itself) under ``prefix``."""
    checksums = "".join(
        f"{hashlib.sha256((curated / p).read_bytes()).hexdigest()}  {prefix}{p}\n"
        for p in relpaths if p != "SHA256SUMS"
    )
    (curated / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def _make_client(repo_id: str, *, token_present: bool | None = None) -> HfHubClient:
    """Require the optional Hub package and token, then construct the production adapter."""
    if _import_hf_hub() is None:
        raise HubUnavailableError(
            "The 'huggingface-hub' package is required for hydrate but is not "
            "installed. Install the optional extra: `uv sync --extra hub` "
            "(or `pip install 'daydream[hub]'`)."
        )
    token_present = token_present if token_present is not None else bool(os.environ.get("HF_TOKEN"))
    if not token_present:
        raise HubUnavailableError(
            "HF_TOKEN is not set; hydration requires a read token for the "
            "private Hub repo. Export HF_TOKEN (or pass --token-source) and retry."
        )
    return HfHubClient(repo_id)


class HfHubClient:
    """Production :class:`HubClient` adapter over a lazily imported ``huggingface_hub``."""

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
            info = self._api.repo_info(self._repo_id, revision=revision, repo_type="dataset")
        except Exception as exc:  # mapped, never swallowed
            raise HubDownloadError(f"repo_info failed for {self._repo_id}: {exc}") from exc
        return RepoInfo(sha=str(info.sha), private=bool(info.private))

    def list_repo_files(self, revision: str | None = None) -> list[str]:
        try:
            return list(self._api.list_repo_files(self._repo_id, revision=revision, repo_type="dataset"))
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

    def commit_files_atomic(
        self,
        mapping: dict[str | Path, Path],
        commit_message: str,
        *,
        parent_commit: str,
        branch: str,
    ) -> str:
        """Commit ``mapping`` as one guarded dataset tree update."""
        from huggingface_hub.errors import HfHubHTTPError  # noqa: PLC0415  # optional lazy dependency

        try:
            operations = [
                self._hf.CommitOperationAdd(
                    path_in_repo=str(path_in_repo),
                    path_or_fileobj=str(local_path),
                )
                for path_in_repo, local_path in sorted(mapping.items(), key=lambda item: str(item[0]))
            ]
        except Exception as exc:
            raise HydrationError(
                redact_text(f"atomic commit input failed for {self._repo_id} on {branch}: {exc}")
            ) from None
        try:
            result = self._api.create_commit(
                repo_id=self._repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
                revision=branch,
                create_pr=False,
                run_as_future=False,
                parent_commit=parent_commit,
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            if isinstance(exc, HfHubHTTPError) and getattr(response, "status_code", None) == 412:
                raise HubConcurrentUpdateError(
                    redact_text(f"atomic commit parent changed for {self._repo_id} on {branch}")
                ) from None
            raise HydrationError(
                redact_text(f"atomic commit failed for {self._repo_id} on {branch}: {exc}")
            ) from None

        oid = str(getattr(result, "oid", ""))
        if _FULL_SHA_RE.fullmatch(oid) is None:
            raise HydrationError(
                redact_text(f"atomic commit returned invalid commit OID for {self._repo_id}")
            )
        return oid


def _curation_manifest_doc(
    stage: Path, *, curation_id: str, source_commit: str, ledger: dict[str, Any]
) -> dict[str, Any]:
    """Render the portable curation manifest (schema v1) from the real ledger (M12/M18)."""
    batches: list[dict[str, Any]] = []
    for admitted, entries in ((True, ledger.get("imported", [])), (False, ledger.get("rejections", []))):
        for entry in entries:
            sid = str(entry["session_id"])
            if admitted:
                status = "admitted"
                code = None
                root = "excluded"
                collision = False
                relpath = f"batches/{sid}"
                digest = entry.get("content_digest") or ""
            else:
                code = entry.get("reason_code")
                status = "excluded" if code in hydrate_rules.EXCLUSION_CODES else "quarantined"
                root = "excluded" if status == "excluded" else "quarantine"
                collision = code == hydrate_rules.REASON_CODE_IDENTITY_COLLISION
                # Rejected paths describe staging. Hash unsafe IDs before path access;
                # identity collisions retain their .conflict suffix.
                segment = sid if _is_bare_segment(sid) else hashlib.sha256(sid.encode()).hexdigest()
                relpath = f"{root}/{segment}.conflict" if collision else f"{root}/{segment}"
                digest = entry.get("content_digest")
                if not isinstance(digest, str) or not digest:
                    # Prefer the rejected staging copy, then its downloaded source.
                    candidates = [stage / "quarantine" / f"{segment}.conflict"] if collision else []
                    candidates.extend((
                        stage / root / segment,
                        stage / "downloads" / str(ledger["pinned_revision"]) / "bundles" / segment,
                    ))
                    source_dir = next((path for path in candidates if path.is_dir()), None)
                    digest = sanitize._derivative_digest(source_dir) if source_dir is not None \
                        else hashlib.sha256(sid.encode()).hexdigest()
            repo_slug, license_evidence = _session_identity(
                stage, sid, str(ledger["pinned_revision"]), root=root, collision=collision
            )
            batches.append({
                "session_id": sid,
                "content_digest": str(digest),
                "status": status,
                "reason_code": code,
                "artifact_relpath": relpath,
                "manifest_relpath": f"{relpath}/manifest.json" if admitted else None,
                "repo_slug": repo_slug,
                "license_evidence": license_evidence,
            })
    return {
        "schema_version": hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        "source_hub_commit": str(source_commit),
        "curation_id": curation_id,
        "sanitizer_version": hydrate_rules.SANITIZER_VERSION,
        "hydration_index_schema_version": hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        "admission_policy_version": hydrate_rules.ADMISSION_POLICY_VERSION,
        "publication_prefix": f"curated/{curation_id}/",
        "batches": batches,
    }


def finalize(client: HubClient, stage: Path, *, curation_id: str, source_commit: str,
             binding: dict[str, Any]) -> str:
    """Publish the curation manifest, policy binding, and final checksums; return its SHA."""
    curated = stage / "curated" / curation_id
    ledger_path = curated / "import-ledger.json"
    if not ledger_path.is_file():
        raise HydrationError(redact_text(f"no import ledger under curated/{curation_id}; cannot finalize"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    doc = _curation_manifest_doc(stage, curation_id=curation_id, source_commit=source_commit, ledger=ledger)
    manifest_path = curated / "curation-manifest.json"
    atomic_write_json(manifest_path, doc)
    prefix = f"curated/{curation_id}/"
    # Issue #1094: pin the policy binding into the published prefix and fail
    # closed on any conflicting remote record before the manifest commit.
    binding_path = curated / "policy-binding.json"
    binding_path.write_text(_policy_binding_record(binding), encoding="utf-8")
    check_prefix_binding(client, curation_id=curation_id, binding=binding, allow_unbound_resume=True)
    # ``_SUCCESS`` is excluded: it does not exist at the verify commit (it is
    # published only after verification), so it must not enter the checksums.
    final_relpaths = [
        p.relative_to(curated).as_posix()
        for p in curated.rglob("*") if p.is_file() and p.name != "_SUCCESS"
    ]
    _write_sha256sums(curated, prefix, final_relpaths)
    _retry_upload(
        client,
        {
            f"{prefix}curation-manifest.json": manifest_path,
            f"{prefix}SHA256SUMS": curated / "SHA256SUMS",
            f"{prefix}policy-binding.json": binding_path,
        },
        f"daydream hydrate {curation_id}: curation manifest + checksums",
    )
    return str(client.repo_info().sha)


def _publish_success_marker(
    client: HubClient, stage: Path, *, curation_id: str, source_commit: str
) -> None:
    """Publish the terminal success commit only after clean-room verification passes."""
    curated = stage / "curated" / curation_id
    prefix = f"curated/{curation_id}/"
    success_path = curated / "_SUCCESS"
    success_path.write_text(
        json.dumps({"curation_id": curation_id, "source_hub_commit": str(source_commit), "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    _retry_upload(client, {f"{prefix}_SUCCESS": success_path},
                  f"daydream hydrate {curation_id}: success marker")


def verify_publication(
    client: HubClient,
    stage: Path,
    *,
    output_commit_sha: str,
    curation_id: str,
    dry_run_admitted: int,
    source_commit: str,
) -> int:
    """Verify pinned output in a fresh directory and return the admitted count."""
    prefix = f"curated/{curation_id}/"
    verify_dir = stage / "_verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    verify_dir.mkdir(parents=True)

    def _download(relpath: str) -> bytes:
        try:
            return client.download_file(f"{prefix}{relpath}", revision=output_commit_sha)
        except HydrationError as exc:
            raise VerificationError(redact_text(f"verify: published file {relpath!r} missing: {exc}")) from exc

    # 1. SHA256SUMS must match every published file byte-for-byte.
    sums_text = _download("SHA256SUMS").decode("utf-8")
    for line in sums_text.splitlines():
        if not line.strip():
            continue
        digest, _, relpath = line.partition("  ")
        relpath = relpath.removeprefix(prefix)
        actual = hashlib.sha256(_download(relpath)).hexdigest()
        if actual != digest:
            raise VerificationError(redact_text(f"verify: checksum mismatch for {relpath!r}"))

    # 2. Curation manifest: schema-valid and consistent with the pinned inputs.
    from jsonschema import Draft202012Validator  # noqa: PLC0415  # lazy: verify-time only

    schema_path = Path(__file__).parent.parent / "training" / "schema" / "curation-manifest.json"
    doc = json.loads(_download("curation-manifest.json").decode("utf-8"))
    errors = sorted(Draft202012Validator(json.loads(schema_path.read_text())).iter_errors(doc), key=str)
    if errors:
        raise VerificationError(redact_text(f"verify: curation manifest invalid: {errors[0].message}"))
    if doc["curation_id"] != curation_id or doc["source_hub_commit"] != str(source_commit):
        raise VerificationError(redact_text("verify: curation manifest identity mismatch"))
    # The _SUCCESS marker is *not* expected at this commit: it is published by
    # run_hydrate_hub only after this cycle passes (verify-before-success), so
    # a failed verification can never leave a published "complete" marker.

    # 3. Rescan every published batch (clean-room) and rebuild the scratch index.
    for batch in doc["batches"]:
        if batch["status"] != "admitted":
            continue
        sid = batch["session_id"]
        batch_dir = verify_dir / "batches" / sid
        for line in sums_text.splitlines():
            if not line.strip():
                continue
            _, _, relpath = line.partition("  ")
            relpath = relpath.removeprefix(prefix)
            if not relpath.startswith(f"batches/{sid}/"):
                continue
            target = batch_dir / relpath.removeprefix(f"batches/{sid}/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_download(relpath))
        scan = scan_run_dir(batch_dir)
        if scan.blocking:
            raise VerificationError(
                redact_text(
                    f"verify: published batch {sid!r} fails the secrets scan ({scan.summary()})"
                )
            )
        if scan.findings:
            # Advisory-only: a name/template shape, not a credential (#1170).
            # Reported value-free rather than failing an already-published
            # commit that a rule which cannot identify a secret objected to.
            _warn(
                f"verify: published batch {sid!r} carries advisory-only scan "
                f"findings ({scan.summary()})"
            )
        if sanitize._derivative_digest(batch_dir) != batch["content_digest"]:
            raise VerificationError(redact_text(f"verify: batch {sid!r} digest mismatch"))
        data = _read_manifest_dict(batch_dir)
        if data is None:
            raise VerificationError(redact_text(f"verify: batch {sid!r} has an unreadable manifest"))
        kwargs = manifest_index_fields(data)
        kwargs.pop("daydream", None)
        kwargs["archive_path"] = str(batch_dir)
        _has_url, slug, canonical = _manifest_remote_fields(data)
        kwargs["repo_slug"], kwargs["remote_url"] = slug, canonical
        upsert_run(verify_dir, kwargs)

    verify_admitted = len(query_runs(verify_dir))
    if verify_admitted != dry_run_admitted:
        raise VerificationError(
            redact_text(
                f"verify: candidate count mismatch — dry run admitted {dry_run_admitted}, "
                f"clean-room rebuild found {verify_admitted}"
            )
        )
    return verify_admitted


def prepare_hydration(config: HydrateHubConfig, source_client: HubClient) -> tuple[str, dict[str, Any] | None]:
    """Stage the exact candidate population shared by preview and publication."""
    source_commit = resolve_source_revision(
        source_client, config.source_revision, exploratory=config.exploratory,
    )
    download_snapshot(source_client, revision=source_commit, stage_dir=config.stage_dir / "downloads")
    ingest_bundles(config.stage_dir, revision=source_commit)
    dedupe_admitted(config.stage_dir, revision=source_commit)
    if config.license_policy_path is None:
        return source_commit, None
    from daydream.archive.license_enrich import (  # noqa: PLC0415  # avoid import cycle
        _make_license_resolver,
        enrich_license_evidence,
    )

    # Resolver drift changes identity; enriched bytes must become the baseline
    # before the gate moves rejected derivatives out of the admitted population.
    enrich_license_evidence(config.stage_dir, resolver=_make_license_resolver())
    restamp_admitted_digests(config.stage_dir, revision=source_commit)
    apply_license_gate(
        config.stage_dir, revision=source_commit, license_policy_path=config.license_policy_path,
        allow_copyleft=config.allow_copyleft,
    )
    binding = resolve_curation_identity(
        config.stage_dir, source_commit=source_commit, license_policy_path=config.license_policy_path,
        allow_copyleft=config.allow_copyleft,
    )
    return source_commit, binding


def run_hydrate_hub(config: HydrateHubConfig, client: HubClient | None = None) -> HydrateSummary:
    """Hydrate through admission, policy binding, publication, and clean-room verification."""
    # Direct callers must meet the same policy prerequisite as the CLI.
    if config.license_policy_path is None:
        raise HydrationError(
            "run_hydrate_hub requires license_policy_path for any publication "
            "path (fail-closed)"
        )
    # Two clients: the source repo guards the pinned snapshot, the destination
    # repo receives the published output. Tests inject one FakeHub for both.
    source_client = client if client is not None else _make_client(config.source_repo)
    dest_client = client if client is not None else _make_client(config.destination_repo)
    if not dest_client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; hydration "
            "publishes sanitized corpora only to private repos (M17)"
        )
    source_commit, binding = prepare_hydration(config, source_client)
    assert binding is not None  # the required policy was checked before client construction
    curation_id = str(binding["curation_id"])
    # The enrichment cache is copied into the *v2* curated prefix as the
    # pinned evidence record of this curation (audit + replay harnesses).
    from daydream.archive.license_enrich import publish_enrichment_cache  # noqa: PLC0415  # avoid import cycle

    publish_enrichment_cache(
        config.stage_dir, revision=source_commit,
        curated_dir=config.stage_dir / "curated" / curation_id,
    )
    ledger = build_import_ledger(
        config.stage_dir, revision=source_commit, source_commit=source_commit, binding=binding,
    )
    license_admission = license_admission_summary(ledger)
    checkpoint = resume_state(dest_client, curation_id=curation_id, stage_dir=config.stage_dir / "_resume")
    # Resume may precede the binding commit; the remote ledger identifies
    # that interrupted window. Conflicting or legacy-unbound prefixes fail.
    check_prefix_binding(
        dest_client, curation_id=curation_id, binding=binding,
        allow_unbound_resume=True,
    )
    publish_batches(
        dest_client, config.stage_dir, curation_id=curation_id, skip_sessions=checkpoint.completed_sessions
    )
    # finalize pins the verify commit (manifest + checksums); _SUCCESS is
    # uploaded only after the clean-room cycle passes (M18/M20).
    output_commit_sha = finalize(
        dest_client, config.stage_dir, curation_id=curation_id, source_commit=source_commit,
        binding=binding,
    )
    summary = HydrateSummary(
        source_commit=source_commit,
        curation_id=curation_id,
        output_commit_sha=output_commit_sha,
        dry_run_discovered=int(ledger["tallies"]["discovered"]),
        dry_run_admitted=int(ledger["tallies"]["imported"]),
        dry_run_rejected=int(ledger["tallies"]["rejections"]),
        dry_run_incomplete_manifests=tuple(ledger["tallies"]["incomplete_manifests"]),
        license_admission=license_admission,
    )
    summary.verify_admitted = verify_publication(
        dest_client,
        config.stage_dir,
        output_commit_sha=output_commit_sha,
        curation_id=curation_id,
        dry_run_admitted=summary.dry_run_admitted,
        source_commit=source_commit,
    )
    _publish_success_marker(
        dest_client, config.stage_dir, curation_id=curation_id, source_commit=source_commit
    )
    summary.output_commit_sha = str(dest_client.repo_info().sha)  # head after the _SUCCESS commit
    summary.verified = True  # only after the full clean-room cycle passes (M20)
    return summary
