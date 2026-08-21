"""Benchmark workspace orchestration: ``init`` / ``status`` / ``validate``.

``workspace.py`` owns the three user-facing commands of the private benchmark
workspace. ``init_workspace`` builds the private layout + manifest through the
transaction journal under the workspace lock; ``workspace_status`` reads the
derived state read-only (no lock, safe concurrently); ``validate_workspace``
returns the ``0/2/1`` classification. Expected workspace errors are never
surfaced as bare tracebacks — ``InitError``/``WorkspaceCorrupt``/schema
failures map to the documented exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseIndexEntry,
    Privacy,
    PullRequestEntry,
    Source,
    classify_validation,
    derive_workspace_state,
    normalize_hostname,
)
from daydream.benchmark.storage import (
    Transaction,
    WorkspaceCorrupt,
    WorkspaceLock,
    ensure_private_dir,
    load_yaml_strict,
    recover_startup,
)

_SUBDIRS = ("imports", "cases", "snapshots", "transactions", "runtime", "cache", "harbor")

_PRIVACY_CLASSIFICATION = "confidential"


class InitError(Exception):
    """A workspace ``init`` refused or failed before the layout was complete."""


def _rfc3339_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_nonempty(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.iterdir())


def _normalize_all(hosts: list[str], what: str) -> list[str]:
    if not hosts:
        raise InitError(f"{what} must not be empty")
    try:
        return [normalize_hostname(h) for h in hosts]
    except ValueError as exc:
        raise InitError(f"{what}: {exc}") from exc


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

    for sub in _SUBDIRS:
        ensure_private_dir(root / sub)

    gitignore_content = "*\n!.gitignore\n"

    with WorkspaceLock(root):
        with Transaction(root, op_id="init", kind="init") as tx:
            tx.stage(".gitignore", gitignore_content.encode("utf-8"))
            tx.stage("benchmark.yaml", _manifest_bytes(privacy, source, benchmark_id))
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


def workspace_status(root: Path) -> WorkspaceStatus:
    """Return a read-only ``WorkspaceStatus`` for ``root``.

    Runs startup recovery first, then reads + strictly validates
    ``benchmark.yaml``. Takes **no** workspace lock so it is safe to run
    concurrently with another read-only command.
    """
    root = Path(root)
    recover_startup(root)
    raw = load_yaml_strict(root / "benchmark.yaml")
    try:
        manifest = BenchmarkManifest.model_validate(raw)
    except Exception as exc:
        raise WorkspaceCorrupt(f"{root}: invalid benchmark.yaml: {exc}") from exc

    pr_dicts = [{"import_state": pr.import_state} for pr in manifest.pull_requests]
    state = derive_workspace_state(pull_requests=pr_dicts, cases=[])
    resolved = manifest.source.repository_id is not None and manifest.source.visibility != "unresolved"
    return WorkspaceStatus(
        workspace_state=state,
        source=manifest.source,
        repository_identity_resolved=resolved,
        ledger=Ledger(pull_requests=manifest.pull_requests),
        cases=manifest.cases,
    )


def validate_workspace(root: Path) -> tuple[int, str]:
    """Validate a workspace, returning a ``(exit_code, human_label)`` pair.

    ``0`` ready; ``2`` structurally valid but incomplete (e.g. unresolved
    repository identity); ``1`` corrupt (invalid/missing ``benchmark.yaml``,
    an orphan/missing indexed file, or a checksum-mismatched import/case).
    Expected workspace errors map to ``1`` + a label — a raw traceback is the
    wrong surface for a bench validation.
    """
    root = Path(root)
    try:
        recover_startup(root)
        raw = load_yaml_strict(root / "benchmark.yaml")
        manifest = BenchmarkManifest.model_validate(raw)
    except Exception as exc:  # schema/checksum/unreadable all map to corruption
        return (1, f"corrupt: invalid benchmark.yaml ({exc})")

    # Orphan + missing-indexed-file rule over the case/import set.
    try:
        recover_startup(
            root,
            indexed=_case_index_paths(manifest),
            on_disk=_scan_case_files(root),
        )
    except WorkspaceCorrupt as exc:
        return (1, f"corrupt: {exc}")

    pr_dicts = [{"import_state": pr.import_state} for pr in manifest.pull_requests]
    state = derive_workspace_state(pull_requests=pr_dicts, cases=[])
    resolved = manifest.source.repository_id is not None and manifest.source.visibility != "unresolved"

    if not resolved:
        return (2, "incomplete: repository identity unresolved")
    ready = state == "ready"
    corrupt_like = state == "corrupt"
    code = classify_validation(ready=ready, incomplete=not ready, corrupt=corrupt_like)
    if code == 1:
        return (1, "corrupt: workspace derived state is corrupt")
    if ready or code == 0:
        return (0, "ready")
    return (code, f"incomplete: workspace state {state}")


def _case_index_paths(manifest: BenchmarkManifest) -> set[str]:
    return {c.case_file for c in manifest.cases}


def _scan_case_files(root: Path) -> set[Path]:
    cases_dir = root / "cases"
    if not cases_dir.exists():
        return set()
    found: set[Path] = set()
    for entry in cases_dir.rglob("*"):
        if entry.is_file():
            found.add(entry)
    return found
