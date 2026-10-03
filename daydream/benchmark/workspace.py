"""Initialize, recover, inspect, and validate private benchmark workspaces. Mutations and
recovery run under the workspace lock. Validation classifies ready/incomplete/corrupt as
0/2/1; expected failures become labeled results.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import yaml
from pydantic import BaseModel

from daydream import git_ops
from daydream.benchmark import schema, snapshot as snapshot_mod
from daydream.benchmark.manifest import load_benchmark_manifest
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
    WorkspaceCorrupt as WorkspaceCorrupt,
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


def _is_nonempty(root: Path) -> bool:
    """True if ``root`` holds real user content (ignore internal crash residue)."""
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
        "created_at": schema.rfc3339_now(),
        "source": source.model_dump(mode="json") | {"repository_id": None, "visibility": "unresolved"},
        "privacy": privacy.model_dump(mode="json"),
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
    """Create a private workspace, refusing a nonempty destination. Journal the 0700
    scaffold, self-ignoring .gitignore, and manifest under the lock, replacing the
    manifest last. Interrupted init rolls back to empty.
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
        try:
            manifest = load_benchmark_manifest(root)
        except WorkspaceCorrupt as exc:
            raise InitError(f"{root}: invalid benchmark.yaml") from exc

    return manifest


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
    """Recover and strictly read workspace state under one lock acquisition."""
    root = Path(root)
    with WorkspaceLock(root):
        recover_startup(root)
        manifest = load_benchmark_manifest(root)
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
    """Return the ledger timestamp of the last successful repository verification, or None."""
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
    """Return (exit code, label): ready=0, incomplete=2, corrupt=1. Schema, orphan,
    checksum, and ready-bundle failures are corruption; expected workspace errors are
    returned as labels.
    """
    root = Path(root)
    with WorkspaceLock(root):
        try:
            recover_startup(root)
        except Exception:  # recovery diagnostics must not disclose journal contents
            return (
                classify_validation(corrupt=True, ready=False),
                f"corrupt: {root}: workspace recovery failed",
            )

        try:
            manifest = load_benchmark_manifest(root)
        except Exception:  # schema/checksum/unreadable all map to corruption
            return (
                classify_validation(corrupt=True, ready=False),
                f"corrupt: {root}: invalid benchmark.yaml",
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
            return (classify_validation(corrupt=True, ready=False), f"corrupt: {exc}")

        # State + identity resolution, shared with status. Loading each case
        # strictly and verifying import checksums (below) surfaces a
        # present-but-corrupt case as ``1`` rather than silently ``draft``.
        try:
            state, resolved = _derived_state(root, manifest, docs)
        except WorkspaceCorrupt as exc:
            return (classify_validation(corrupt=True, ready=False), f"corrupt: {exc}")

    reasons = sorted(
        {
            doc.snapshot.error.reason
            for doc in docs.values()
            if isinstance(doc.snapshot, schema.SnapshotUnreplayable)
        }
    )
    ready = resolved and state == "ready"
    if ready:
        label = "ready"
    elif not resolved:
        label = "incomplete: repository identity unresolved"
    else:
        label = f"incomplete: workspace state {state}"
    if reasons:
        label += f"; unreplayable snapshot reasons: {', '.join(reasons)}"
    return (classify_validation(ready=ready, corrupt=False), label)


def _derived_state(
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument]
) -> tuple[str, bool]:
    """Derive state and identity resolution from one validated document/path inventory.
    Verify import bytes, ready bundle fidelity, and any retained source mirror.
    Unreadable documents or failed proofs raise WorkspaceCorrupt.
    """
    pr_dicts = [{"import_state": pr.import_state} for pr in manifest.pull_requests]
    state = derive_workspace_state(
        pull_requests=pr_dicts,
        cases=_case_curation_states(root, manifest, docs),
    )
    # Load/validate each import and resolve authoring paths once, sharing the results
    # across checksum, cross-document, and duplicate-inode checks.
    imports = _import_documents(root, manifest)
    paths = _resolved_authoring_paths(root, manifest, docs)
    _verify_import_checksums(root, manifest, paths)
    _verify_snapshot_checksums(root, manifest, docs, paths)
    _verify_snapshot_source_provenance(root, manifest, docs)
    _verify_cross_document(root, manifest, docs, imports=imports)
    _verify_duplicate_inodes(root, paths)
    resolved = manifest.source.repository_id is not None and manifest.source.visibility != "unresolved"
    return state, resolved


def load_case_documents(root: Path, manifest: BenchmarkManifest) -> dict[str, CaseDocument]:
    """Load indexed cases through containment, strict YAML, and CaseDocument validation.
    Strip persisted audit fields via _schema_ready; report corrupt cases by path.
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
    """Resolve, strictly load, optionally preprocess, and model-validate an authoring file.
    Wrap failures with only the document kind/path: Pydantic reprs can disclose private
    PR bodies and evidence and must never reach CLI diagnostics.
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
    """Verify ready bundle bytes and the portable offline-clone fidelity contract. Require
    bundle path/digest, exact refs, two synthetic commits, root base, head parentage,
    tree ids, and canonical diff. A restamped but tampered bundle is corruption, never
    curatable staleness. Source-mirror provenance is checked separately.
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
        # A restamped checksum cannot prove fidelity. Validate exact refs, synthetic
        # commit topology, trees, and canonical diff in a network-disabled scratch clone.
        # Any failure is corruption; never fall back.
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


def _verify_snapshot_source_provenance(
    root: Path,
    manifest: BenchmarkManifest,
    docs: dict[str, CaseDocument],
) -> None:
    """Repeat ready snapshot linkage proofs from a retained local mirror when available.
    After cache cleanup, the required provenance marker and self-contained
    bundle-fidelity check remain the portable contract.
    """
    mirror = snapshot_mod.mirror(root)
    if not mirror.exists():
        return
    for case in manifest.cases:
        snapshot = docs[case.case_file].snapshot
        if not isinstance(snapshot, schema.SnapshotReady):
            continue
        try:
            resolved_base = snapshot_mod.resolve_original_base(
                mirror,
                snapshot.requested_base_sha,
                snapshot.original_head_sha,
            )
            trees = snapshot_mod.resolve_trees(
                mirror,
                snapshot.original_base_sha,
                snapshot.original_head_sha,
            )
        except git_ops.GitError as exc:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot source provenance cannot be resolved "
                "from cache/repository.git; restore that mirror before retrying"
            ) from exc
        if resolved_base != snapshot.original_base_sha:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot source provenance merge-base mismatch"
            )
        if not isinstance(trees, tuple):
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot source provenance object is missing "
                "from cache/repository.git; restore that mirror before retrying"
            )
        base_tree, head_tree = trees
        if base_tree != snapshot.base_tree_sha or head_tree != snapshot.head_tree_sha:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} snapshot source provenance tree mismatch"
            )


def _import_documents(root: Path, manifest: BenchmarkManifest) -> dict[str, ImportDocument]:
    """Load fetched imports once through containment and strict model validation, without
    body disclosure.
    """
    return {
        pr.import_file: _load_authoring_document(
            root,
            pr.import_file,
            what="import",
            loader=load_json_strict,
            model=ImportDocument,
        )
        for pr in manifest.pull_requests
        if pr.import_state == "fetched" and pr.import_file
    }


def _verify_import_checksums(
    root: Path,
    manifest: BenchmarkManifest,
    paths: dict[str, Path],
) -> None:
    """Require every fetched import file and its recorded digest; mismatches are
    corruption.
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
    imports: dict[str, ImportDocument],
) -> None:
    """Verify PR/repository identity, source hashes, canonical case paths, and ledger
    claims. Every ledger-claimed case must be indexed for that PR. The index may contain
    additional cases retained after a narrower re-import. All other identity or
    membership disagreements are corruption.
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
        entry = ledger[case.pr_number]
        if (
            entry.import_state != "fetched"
            or entry.import_file is None
            or entry.import_sha256 is None
            or doc.source.import_file != entry.import_file
            or doc.source.import_sha256 != entry.import_sha256
        ):
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} import provenance mismatches its ledger entry"
            )
        imp = imports[entry.import_file]
        if doc.pull_request != imp.pull_request:
            raise WorkspaceCorrupt(
                f"{root}: case {case.case_id} import provenance PR metadata mismatch"
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
        imp = imports[pr.import_file]
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
    """Collect indexed cases, fetched imports, and ready bundles for complete orphan
    detection.
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
    """Resolve indexed authoring files once, omitting absent files. Checksum/orphan checks
    report missing files; inode checks consume present ones.
    """
    paths: dict[str, Path] = {}
    for rel in _case_index_paths(manifest, docs):
        path = resolve_authoring_path(root, rel)
        if path.exists():
            paths[rel] = path
    return paths


def _verify_duplicate_inodes(root: Path, paths: dict[str, Path]) -> None:
    """Reject distinct indexed authoring paths sharing a device/inode, including hard
    links.
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
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument]
) -> list[dict[str, str]]:
    """Read validated case curation states for workspace-state derivation."""
    states: list[dict[str, str]] = []
    for case in manifest.cases:
        doc = docs[case.case_file]
        state = doc.curation.state
        if state == "ready" and _task_spec_approval_state(doc) == "stale":
            state = "stale"
        states.append({"curation_state": state})
    return states


def _case_snapshot_summaries(
    root: Path, manifest: BenchmarkManifest, docs: dict[str, CaseDocument]
) -> list[dict[str, str]]:
    """Summarize validated snapshot state, frozen head prefix, and unreplayable reason."""
    summaries: list[dict[str, str]] = []
    for case in manifest.cases:
        doc = docs[case.case_file]
        status = doc.snapshot.status or "imported"
        head = doc.snapshot.original_head_sha or ""
        error_reason = (
            doc.snapshot.error.reason
            if isinstance(doc.snapshot, schema.SnapshotUnreplayable)
            else ""
        )
        summaries.append(
            {
                "case_id": case.case_id,
                "snapshot_status": status,
                "head_prefix": head[:12],
                "error_reason": error_reason,
                "task_spec_approval": _task_spec_approval_state(doc),
            }
        )
    return summaries


def _task_spec_approval_state(doc: CaseDocument) -> str:
    from daydream.benchmark.harbor.build import task_spec_approval

    return task_spec_approval(doc.model_dump(mode="json")).state


def _scan_authoring_files(root: Path) -> set[Path]:
    """Scan cases/imports/snapshots for orphans; exclude disposable runtime, cache, and
    transaction state.
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
