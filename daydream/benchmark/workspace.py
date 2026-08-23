"""Benchmark workspace orchestration: ``init`` / ``status`` / ``validate``.

``workspace.py`` owns the three user-facing commands of the private benchmark
workspace. ``init_workspace`` builds the private layout + manifest through the
transaction journal under the workspace lock; ``workspace_status``
reads the derived state (recovery + read under the lock, so it serializes
safely against a concurrent writer); ``validate_workspace`` returns the
``0/2/1`` classification. Expected workspace errors are never
surfaced as bare tracebacks — ``InitError``/``WorkspaceCorrupt``/schema
failures map to the documented exit codes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import yaml
from pydantic import BaseModel

from daydream import git_ops
from daydream.benchmark import schema
from daydream.benchmark import snapshot as snapshot_mod
from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseDocument,
    CaseIndexEntry,
    ImportDocument,
    PreflightLedger,
    Privacy,
    PullRequestEntry,
    Source,
    _normalize_host_list,
    classify_validation,
    derive_workspace_state,
)
from daydream.benchmark.storage import (
    Transaction,
    WorkspaceCorrupt,
    WorkspaceLock,
    load_json_strict,
    load_yaml_strict,
    recover_startup,
    resolve_authoring_path,
    sha256_file,
)

_SUBDIRS = ("imports", "cases", "snapshots", "transactions", "runtime", "cache")

_PRIVACY_CLASSIFICATION = "confidential"


class InitError(Exception):
    """A workspace ``init`` refused or failed before the layout was complete."""


def _rfc3339_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_nonempty(root: Path) -> bool:
    """True if ``root`` holds real user content (ignore internal crash residue).

    The workspace lock file and empty managed scaffold dirs (``cases/``,
    ``transactions/``, and the other layout subdirs) are internal residue left
    by an interrupted ``init``; they must not block a clean re-init.
    """
    if not root.exists():
        return False
    for entry in root.iterdir():
        name = entry.name
        if name == ".benchmark.lock":
            continue
        if entry.is_dir() and name in _SUBDIRS and not any(entry.iterdir()):
            continue
        return True
    return False


def _normalize_all(hosts: list[str], what: str) -> list[str]:
    try:
        return _normalize_host_list(hosts, what)
    except ValueError as exc:
        raise InitError(str(exc)) from exc


def _manifest_bytes(privacy: Privacy, source: Source, benchmark_id: str) -> bytes:
    doc = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": _rfc3339_now(),
        "source": {
            "provider": source.provider,
            "hostname": source.hostname,
            "repository": source.repository,
            "repository_id": None,
            "visibility": "unresolved",
        },
        "privacy": {
            "classification": privacy.classification,
            "reviewer_data": privacy.reviewer_data,
            "reviewer_allowed_hosts": privacy.reviewer_allowed_hosts,
            "judge_data": privacy.judge_data,
            "judge_allowed_hosts": privacy.judge_allowed_hosts,
            "archive": privacy.archive,
            "uploads": privacy.uploads,
        },
        "pull_requests": [],
        "cases": [],
    }
    return yaml.safe_dump(doc, sort_keys=False).encode("utf-8")


def init_workspace(
    root: Path,
    repo: str,
    reviewer_hosts: list[str],
    judge_hosts: list[str],
) -> BenchmarkManifest:
    """Create a private benchmark workspace at ``root``.

    Refuses a pre-existing nonempty directory, builds the ``0700`` private
    layout + self-ignoring ``.gitignore``, and persists ``.gitignore`` +
    ``benchmark.yaml`` through the transaction journal under the workspace
    lock (``benchmark.yaml`` replaced last). A crash mid-init rolls back to an
    empty/absent workspace.
    """
    root = Path(root)
    # Heal any interrupted prior init journal so the rollback-to-empty/absent
    # guarantee holds for a re-init.
    try:
        recover_startup(root)
    except WorkspaceCorrupt as exc:
        raise InitError(f"refusing to init into an unrecoverable workspace: {exc}") from exc
    if _is_nonempty(root):
        raise InitError(f"refusing to init into a nonempty directory: {root}")

    normalized_reviewer = _normalize_all(reviewer_hosts, "reviewer_hosts")
    normalized_judge = _normalize_all(judge_hosts, "judge_hosts")
    if not repo or "/" not in repo or repo.startswith("/") or repo.endswith("/"):
        raise InitError(f"repository must be OWNER/REPO, got {repo!r}")

    source = Source.model_validate(
        {"provider": "github", "hostname": "github.com", "repository": repo}
    )
    privacy = Privacy.model_validate(
        {
            "classification": _PRIVACY_CLASSIFICATION,
            "reviewer_data": "source_snapshot",
            "reviewer_allowed_hosts": normalized_reviewer,
            "judge_data": "finding_text_and_location_only",
            "judge_allowed_hosts": normalized_judge,
            "archive": "disabled",
            "uploads": "disabled",
        }
    )
    benchmark_id = str(uuid4())

    gitignore_content = "*\n!.gitignore\n"

    with WorkspaceLock(root):
        with Transaction(root, op_id="init", kind="init") as tx:
            tx.stage(".gitignore", gitignore_content.encode("utf-8"))
            tx.stage("benchmark.yaml", _manifest_bytes(privacy, source, benchmark_id))
            # The 0700 layout subdirs are journaled too, so an interrupted init
            # rolls them back with the rest of the transaction (``transactions/``
            # itself is created by the journal and cleaned by recovery).
            for sub in _SUBDIRS:
                if sub != "transactions":
                    tx.create_dir(sub)
            tx.commit()

    return BenchmarkManifest.model_validate(load_yaml_strict(root / "benchmark.yaml"))


@dataclass
class Ledger:
    """The workspace's parsed ``pull_requests[]`` ledger."""

    pull_requests: list[PullRequestEntry]


@dataclass
class WorkspaceStatus:
    """Read-only derived status of a benchmark workspace."""

    workspace_state: str
    source: Source
    repository_identity_resolved: bool
    ledger: Ledger
    cases: list[CaseIndexEntry]
    case_snapshots: list[dict[str, str]] = field(default_factory=list)
    last_preflight_verified_at: str | None = None


def workspace_status(root: Path) -> WorkspaceStatus:
    """Return a read-only ``WorkspaceStatus`` for ``root``.

    Runs startup recovery, then reads + strictly validates ``benchmark.yaml``.
    Recovery mutates the tree (it rolls back interrupted journals), so it runs
    under the workspace lock; a status is safe to run concurrently because
    each call holds the lock only for the duration of its recovery+read.
    """
    root = Path(root)
    with WorkspaceLock(root):
        recover_startup(root)
        raw = load_yaml_strict(root / "benchmark.yaml")
        try:
            manifest = BenchmarkManifest.model_validate(raw)
        except Exception as exc:
            raise WorkspaceCorrupt(f"{root}: invalid benchmark.yaml: {exc}") from exc
        docs = load_case_documents(root, manifest)
        state, resolved = _derived_state(root, manifest, docs)
        case_snapshots = _case_snapshot_summaries(root, manifest, docs)
    return WorkspaceStatus(
        workspace_state=state,
        source=manifest.source,
        repository_identity_resolved=resolved,
        last_preflight_verified_at=_last_preflight_verified_at(root),
        ledger=Ledger(pull_requests=manifest.pull_requests),
        cases=manifest.cases,
        case_snapshots=case_snapshots,
    )


def _last_preflight_verified_at(root: Path) -> str | None:
    """Return the ledger timestamp of the last successful repository verification, or None.

    A missing or malformed ledger reads as absent (status is read-only and must
    never fail the workspace over the mode-0600 ``runtime/preflight.json``).
    """
    path = root / "runtime" / "preflight.json"
    if not path.exists():
        return None
    try:
        raw = load_json_strict(path)
        ledger = PreflightLedger.model_validate(raw)
        if not ledger.matched:
            return None
        return ledger.last_verified_at
    except Exception:
        return None


def validate_workspace(root: Path) -> tuple[int, str]:
    """Validate a workspace, returning a ``(exit_code, human_label)`` pair.

    ``0`` ready; ``2`` structurally valid but incomplete (e.g. unresolved
    repository identity); ``1`` corrupt (invalid/missing ``benchmark.yaml``,
    an orphan/missing indexed file, a checksum-mismatched import/case, or a
    corrupted/missing/checksum-mismatched ready-snapshot bundle). The exit
    code comes from :func:`classify_validation`, so the documented
    ``0``/``2``/``1`` classifier has a single source of truth. Expected
    workspace errors map to ``1`` + a label — a raw traceback is the wrong
    surface for a bench validation.
    """
    root = Path(root)
    with WorkspaceLock(root):
        try:
            recover_startup(root)
            raw = load_yaml_strict(root / "benchmark.yaml")
            manifest = BenchmarkManifest.model_validate(raw)
        except Exception as exc:  # schema/checksum/unreadable all map to corruption
            return (
                classify_validation(corrupt=True, ready=False, incomplete=False),
                f"corrupt: invalid benchmark.yaml ({exc})",
            )

        # Orphan + missing-indexed-file rule over the case/import/bundle set.
        try:
            docs = load_case_documents(root, manifest)
            recover_startup(
                root,
                indexed=_case_index_paths(manifest, docs),
                on_disk=_scan_authoring_files(root),
            )
        except WorkspaceCorrupt as exc:
            return (classify_validation(corrupt=True, ready=False, incomplete=False), f"corrupt: {exc}")

        # State + identity resolution, shared with status. Loading each case
        # strictly and verifying import checksums (below) surfaces a
        # present-but-corrupt case as ``1`` rather than silently ``draft``.
        try:
            state, resolved = _derived_state(root, manifest, docs)
        except WorkspaceCorrupt as exc:
            return (classify_validation(corrupt=True, ready=False, incomplete=False), f"corrupt: {exc}")

    ready = resolved and state == "ready"
    if ready:
        label = "ready"
    elif not resolved:
        label = "incomplete: repository identity unresolved"
    else:
        label = f"incomplete: workspace state {state}"
    return (classify_validation(ready=ready, incomplete=not ready, corrupt=False), label)


def _derived_state(
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument] | None = None
) -> tuple[str, bool]:
    """Shared (workspace state, identity-resolved) derivation for status+validate.

    Loading each indexed case document with the shared model-gated loader,
    resolving every indexed authoring file exactly once, and verifying each
    fetched import's on-disk sha256 plus each ``ready`` snapshot's bundle
    sha256 keeps the two read-only call paths on one rule, so a
    state/resolution rule can't diverge between them. An unreadable/invalid
    case or a checksum mismatch surfaces as :class:`WorkspaceCorrupt`.
    """
    if docs is None:
        docs = load_case_documents(root, manifest)
    pr_dicts = [{"import_state": pr.import_state} for pr in manifest.pull_requests]
    state = derive_workspace_state(
        pull_requests=pr_dicts,
        cases=_case_curation_states(root, manifest, docs),
    )
    # Model-validate every fetched import exactly once per call; the checksum
    # and cross-document verifiers share this set instead of each re-reading
    # and re-model-validating the same documents. The authoring index is
    # likewise resolved (and existence-checked) exactly once per call, shared
    # by the checksum + duplicate-inode verifiers instead of each verifier
    # re-resolving the same files.
    imports = _import_documents(root, manifest)
    paths = _resolved_authoring_paths(root, manifest, docs)
    _verify_import_checksums(root, manifest, paths)
    _verify_snapshot_checksums(root, manifest, docs, paths)
    _verify_cross_document(root, manifest, docs, imports=imports)
    _verify_duplicate_inodes(root, paths)
    resolved = manifest.source.repository_id is not None and manifest.source.visibility != "unresolved"
    return state, resolved


def load_case_documents(root: Path, manifest: BenchmarkManifest) -> dict[str, CaseDocument]:
    """Load every indexed case document as a strict ``CaseDocument`` model.

    Shared by the validate/status read path and the ``harbor`` compile path,
    keyed by ``case_file``. Each case file is resolved through
    :func:`resolve_authoring_path` (containment enforced) and validated with
    ``CaseDocument.model_validate`` after the persisted ``gold_mode`` audit
    field is stripped via :func:`schema._schema_ready` — a present-but-corrupt
    case raises :class:`WorkspaceCorrupt` naming the ``case_file``, never a
    defaulted/skipped read.
    """
    docs: dict[str, CaseDocument] = {}
    for case in manifest.cases:
        docs[case.case_file] = _load_authoring_document(
            root,
            case.case_file,
            what="case",
            loader=load_yaml_strict,
            model=CaseDocument,
            preprocess=schema._schema_ready,
        )
    return docs


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_authoring_document(
    root: Path,
    rel: str,
    *,
    what: str,
    loader: Callable[[Path], dict[str, Any]],
    model: type[_ModelT],
    preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> _ModelT:
    """Resolve one authoring document and model-gate it through the strict loader.

    The single definition of the resolve -> strict-load -> model-validate ->
    wrap-in-:class:`WorkspaceCorrupt` block shared by the case and import
    authoring loaders: resolution through :func:`resolve_authoring_path`, the
    strict ``loader``, then ``model.model_validate`` (with an optional
    ``preprocess`` of the raw dict, e.g. :func:`schema._schema_ready`). A
    present-but-invalid document raises :class:`WorkspaceCorrupt` naming only
    ``what`` + the authoring file -- the diagnostic never embeds the pydantic
    error, whose repr embeds the input document (PR bodies/evidence) that the
    CLI's no-disclosure contract keeps off stderr.
    """
    path = resolve_authoring_path(root, rel)
    try:
        raw = loader(path)
        if preprocess is not None:
            raw = preprocess(raw)
        return model.model_validate(raw)
    except Exception as exc:
        raise WorkspaceCorrupt(
            f"{root}: {what} {rel} is not a valid {what} document"
        ) from exc


def _verify_snapshot_checksums(
    root: Path,
    manifest: BenchmarkManifest,
    docs: dict[str, CaseDocument],
    paths: dict[str, Path],
) -> None:
    """Verify each indexed ``ready`` snapshot's bundle + sha256 digest.

    A missing ``bundle_file`` or a ``bundle_sha256`` mismatch for a committed
    ``ready`` case is :class:`WorkspaceCorrupt` — it is corruption surfacing,
    never curatable staleness, and never mutates the case document or ledger.

    A ``ready`` snapshot with no ``bundle_file``/``bundle_sha256`` is itself
    structurally invalid and reported corrupt. ``paths`` is the shared
    single-resolution pass (:func:`_resolved_authoring_paths`); a ``ready``
    bundle absent from it is a missing file.

    Beyond the checksum, every ``ready`` snapshot is validated with the
    authoritative offline-clone fidelity contract
    (:func:`daydream.benchmark.snapshot.validate_offline_clone`) on a
    disposable network-disabled clone under ``root/cache`` — exact refs, two
    synthetic reachable commits, root base, head-parented-on-base, tree IDs
    and the canonical diff digest. A fidelity failure is corruption (exit 1),
    never curatable staleness; only a checksum-restamped tampered bundle
    passes the sha256 gate yet still fails here.
    """
    for case in manifest.cases:
        snapshot = docs[case.case_file].snapshot
        if not isinstance(snapshot, schema.SnapshotReady):
            continue
        bundle_rel = snapshot.bundle_file
        expected = snapshot.bundle_sha256
        if not bundle_rel or not expected:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} ready snapshot missing "
                f"bundle_file/bundle_sha256"
            )
        bundle_path = paths.get(bundle_rel)
        if bundle_path is None:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot bundle missing: {bundle_rel}"
            )
        actual = sha256_file(bundle_path)
        if actual != expected:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot bundle checksum mismatch "
                f"(expected {expected}, got {actual})"
            )
        # Authoritative offline-clone fidelity: the recorded sha256 alone cannot
        # prove the bundle is the frozen base->head snapshot (a restamped
        # tampered bundle matches), so run the full fidelity contract (exact
        # refs, two synthetic commits, root base, head-parented-on-base, tree
        # IDs, canonical diff digest) on a disposable network-disabled clone of
        # the bundle under ``cache/``. A fidelity failure is corruption — never
        # swallowed, never a fallback.
        try:
            snapshot_mod.validate_offline_clone(
                bundle_path,
                snapshot.base_tree_sha,
                snapshot.head_tree_sha,
                snapshot.diff_sha256,
                workdir=root / "cache",
            )
        except (git_ops.GitError, OSError) as exc:
            # OSError covers a missing ``root/cache`` scratch dir surfacing as
            # FileNotFoundError from the probe's mkdtemp — mapped to corruption
            # (exit 1) like the sibling freeze path, never a bare traceback.
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot bundle fails offline-clone "
                f"fidelity: {exc}"
            ) from exc


def _load_import_document(root: Path, import_file: str) -> ImportDocument:
    """Model-gate one fetched import through the shared strict loader.

    The single definition of the import load+validate block shared by the
    checksum and cross-document verifiers (see :func:`_load_authoring_document`):
    resolution through :func:`resolve_authoring_path` plus
    ``ImportDocument.model_validate`` on the strict JSON read. A
    present-but-invalid import raises :class:`WorkspaceCorrupt` naming only
    the import file -- the diagnostic never embeds the document body (the
    CLI's no-disclosure contract).
    """
    return _load_authoring_document(
        root,
        import_file,
        what="import",
        loader=load_json_strict,
        model=ImportDocument,
    )


def _import_documents(root: Path, manifest: BenchmarkManifest) -> dict[str, ImportDocument]:
    """Load every fetched import once through the shared model gate.

    ``_derived_state`` precomputes the validated set so the checksum and
    cross-document verifiers consume the same models instead of each
    re-reading and re-model-validating every fetched import per call.
    """
    return {
        pr.import_file: _load_import_document(root, pr.import_file)
        for pr in manifest.pull_requests
        if pr.import_state == "fetched" and pr.import_file
    }


def _verify_import_checksums(
    root: Path,
    manifest: BenchmarkManifest,
    paths: dict[str, Path],
) -> None:
    """Verify each fetched import's on-disk sha256 against ``import_sha256``.

    A missing import file or a checksum mismatch is a :class:`WorkspaceCorrupt`
    failure — it is never folded into an incomplete/curating result. Each
    import is model-validated exactly once per call by :func:`_import_documents`
    before the verifiers run (see :func:`_derived_state`), and ``paths`` is
    the shared single-resolution pass (:func:`_resolved_authoring_paths`);
    an import absent from it is a missing file.
    """
    for pr in manifest.pull_requests:
        if pr.import_state != "fetched" or pr.import_file is None or pr.import_sha256 is None:
            continue
        path = paths.get(pr.import_file)
        if path is None:
            raise WorkspaceCorrupt(f"{root}: import {pr.import_file} is missing on disk")
        actual = sha256_file(path)
        if actual != pr.import_sha256:
            raise WorkspaceCorrupt(
                f"{root}: import {pr.import_file} checksum mismatch "
                f"(expected {pr.import_sha256}, got {actual})"
            )


def _verify_cross_document(
    root: Path,
    manifest: BenchmarkManifest,
    docs: dict[str, CaseDocument],
    imports: dict[str, ImportDocument] | None = None,
) -> None:
    """Verify every cross-document identity link and exact index membership.

    Each ``cases[]`` row must reference exactly ``cases/<case_id>.yaml`` and
    agree with its case document's ``pull_request.number``, and its PR must be
    present in the ``pull_requests[]`` ledger. Every case_id a ledger entry
    claims must be backed by an indexed ``cases[]`` row naming the same PR —
    the reverse (every indexed case covered by ``case_ids``) is not required,
    because a fetched->fetched narrower re-import (shrink) rewrites the
    ledger's ``case_ids`` to the newly requested heads while the previously
    imported case rows stay indexed, so the index legitimately outgrows the
    claim. Each fetched import document must name the same PR number and
    repository as its ledger entry. Any mismatch is
    :class:`WorkspaceCorrupt` — never a logged skip.
    """
    ledger = {pr.number: pr for pr in manifest.pull_requests}
    for case in manifest.cases:
        exact = f"cases/{case.case_id}.yaml"
        if case.case_file != exact:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} case_file {case.case_file!r} is not the "
                f"exact index path {exact!r}"
            )
        doc = docs[case.case_file]
        if doc.pull_request.number != case.pr_number:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} pull_request.number {doc.pull_request.number} "
                f"mismatches cases[] pr_number {case.pr_number}"
            )
        if case.pr_number not in ledger:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} PR {case.pr_number} is absent from "
                f"the pull_requests ledger"
            )
    indexed_cases = {case.case_id: case for case in manifest.cases}
    for pr in manifest.pull_requests:
        for case_id in pr.case_ids:
            row = indexed_cases.get(case_id)
            if row is None or row.pr_number != pr.number:
                raise WorkspaceCorrupt(
                    f"{root}: ledger PR {pr.number} case_ids {pr.case_ids!r} claims "
                    f"case {case_id} with no matching indexed cases[] row"
                )
        if pr.import_state != "fetched" or pr.import_file is None:
            continue
        imp = (
            imports[pr.import_file]
            if imports is not None and pr.import_file in imports
            else _load_import_document(root, pr.import_file)
        )
        if imp.pull_request.number != pr.number:
            raise WorkspaceCorrupt(
                f"{root}: import {pr.import_file} pull_request.number "
                f"{imp.pull_request.number} mismatches ledger PR {pr.number}"
            )
        if imp.repository.name_with_owner != manifest.source.repository:
            raise WorkspaceCorrupt(
                f"{root}: import {pr.import_file} repository "
                f"{imp.repository.name_with_owner!r} mismatches manifest source "
                f"{manifest.source.repository!r}"
            )


def _case_index_paths(manifest: BenchmarkManifest, docs: dict[str, CaseDocument]) -> set[str]:
    """Every authoring file the workspace index owns, across all three trees.

    The manifest index covers each indexed case document **plus** every
    ``fetched`` ledger import file **plus** every ``ready`` snapshot bundle
    referenced by the model-validated case docs. Orphan detection then spans
    ``cases/``, ``imports/``, and ``snapshots/``, so an unindexed import or
    bundle — and a referenced-but-missing one — surfaces as
    :class:`WorkspaceCorrupt` instead of being silently adopted or reported
    ``incomplete``.
    """
    paths = {c.case_file for c in manifest.cases}
    for pr in manifest.pull_requests:
        if pr.import_state == "fetched" and pr.import_file:
            paths.add(pr.import_file)
    for case in manifest.cases:
        doc = docs[case.case_file]
        if doc.snapshot.status == "ready" and doc.snapshot.bundle_file:
            paths.add(doc.snapshot.bundle_file)
    return paths


def _resolved_authoring_paths(
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument]
) -> dict[str, Path]:
    """Resolve every indexed authoring file exactly once, omitting missing ones.

    One :func:`resolve_authoring_path` + existence check per indexed rel,
    shared by the checksum and duplicate-inode verifiers so each
    validate/status call makes a single resolve+stat pass over the index
    instead of one per verifier. Missing files never enter the map — the
    checksum verifiers report them as corrupt (a missing import/bundle is
    :class:`WorkspaceCorrupt`), and the inode verifier only collides files
    that actually exist.
    """
    paths: dict[str, Path] = {}
    for rel in _case_index_paths(manifest, docs):
        path = resolve_authoring_path(root, rel)
        if path.exists():
            paths[rel] = path
    return paths


def _verify_duplicate_inodes(root: Path, paths: dict[str, Path]) -> None:
    """Reject two distinct indexed authoring files sharing one ``(st_dev, st_ino)``.

    A hard link (or any duplicate-inode surprise) between two differently-
    named indexed authoring files is corruption: ``Path.resolve()`` cannot
    distinguish the names (both resolve inside ``root``), so the batch inode
    cross-check across every resolved indexed authoring file is the enforcement
    point (Task 0 spike 4). ``paths`` is the shared single-resolution pass
    (:func:`_resolved_authoring_paths`): missing files are already omitted
    there (they are corruption via the orphan rule / checksum gates), so only
    files that actually exist are collided. Every check here is a hard
    failure — never a skip.
    """
    seen: dict[tuple[int, int], str] = {}
    for rel in sorted(paths):
        path = paths[rel]
        key = (path.stat().st_dev, path.stat().st_ino)
        if key in seen:
            raise WorkspaceCorrupt(
                f"{root}: indexed authoring files {seen[key]!r} and {rel!r} "
                f"share inode ({key[0]}, {key[1]})"
            )
        seen[key] = rel


def _case_curation_states(
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument] | None = None
) -> list[dict[str, str]]:
    """The ``curation.state`` per indexed case, for workspace-state derivation.

    ``derive_workspace_state`` needs the real curation states (its ``ready`` /
    ``stale`` / ``curating`` branches are driven by them); passing ``[]`` made
    those branches unreachable so a fully curated workspace could never report
    ``ready``. Each case document is loaded through the shared model-gated
    loader — an unreadable/invalid case surfaces as
    :class:`WorkspaceCorrupt` rather than being silently folded into ``draft``
    (storage's strict-loader invariant: a corrupt file is an error, never
    defaulted). A validated model always carries a concrete ``curation.state``.
    """
    if docs is None:
        docs = load_case_documents(root, manifest)
    states: list[dict[str, str]] = []
    for case in manifest.cases:
        states.append({"curation_state": docs[case.case_file].curation.state})
    return states


def _case_snapshot_summaries(
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument] | None = None
) -> list[dict[str, str]]:
    """Per-case snapshot summary for ``status``: snapshot state + frozen head.

    For each indexed case, loads the case through the shared model-gated
    loader and reports its snapshot ``status`` and the frozen head prefix
    (``original_head_sha[:12]``) when present. An unreadable/invalid case
    surfaces as :class:`WorkspaceCorrupt` (shared with the validate path).
    """
    if docs is None:
        docs = load_case_documents(root, manifest)
    summaries: list[dict[str, str]] = []
    for case in manifest.cases:
        doc = docs[case.case_file]
        status = doc.snapshot.status or "imported"
        head = doc.snapshot.original_head_sha or ""
        summaries.append(
            {
                "case_id": case.case_id,
                "snapshot_status": status,
                "head_prefix": head[:12],
            }
        )
    return summaries


def _scan_authoring_files(root: Path) -> set[Path]:
    """Every regular file under the authoring trees: ``cases/``, ``imports/``, ``snapshots/``.

    Runtime/cache/transaction residue is not authoring content, so it is never
    scanned — an unindexed authoring file in one of the three trees is
    orphan corruption, while internal state under ``runtime/``/``cache/``/
    ``transactions/`` stays out of the orphan rule.
    """
    found: set[Path] = set()
    for sub in ("cases", "imports", "snapshots"):
        tree = root / sub
        if not tree.exists():
            continue
        for entry in tree.rglob("*"):
            if entry.is_file():
                found.add(entry)
    return found
