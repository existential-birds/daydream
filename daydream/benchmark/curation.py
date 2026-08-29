"""UI-independent golden-review curation service for private benchmark workspaces.

This module is the issue-#5 browser/terminal seam: the fixed operation set a
curator (or the future interactive client) drives every gold-curation action
through. Every mutating operation (``accept_candidate``, ``add_finding``,
``add_findings``, ``add_edited_findings``, ``replace_findings``, ``exclude_evidence``, ``mark_ready``,
``attest_clean``, ``exclude_case``, ``reinclude_case``, ``apply_gold_fragment``)
runs its complete read -> validate -> mutate -> commit sequence under the
workspace lock:

1. acquires the blocking :class:`storage.WorkspaceLock` (process-reentrant per
   root), so concurrent curators/processes serialize and can never silently
   lose an update,
2. heals any prior interrupted journal via :func:`storage.recover_startup`
   under the lock, so a crashed earlier process's leftover ``committing``
   journal is rolled back before a new write,
3. loads the target case YAML strictly **after** acquiring the lock — never a
   stale pre-lock view,
4. derives ``finding_id`` / ``provenance.kind`` / ``gold_status`` /
   ``gold_mode`` / ``state`` — never caller-supplied — and enforces state
   transitions via :meth:`schema.validate_case_transition`,
5. re-validates the whole resulting :class:`schema.CaseDocument` plus
   validation (location-vs-head from a disposable clone of the frozen bundle,
   >50 gold cap, duplicate canonical finding, historical byte-match,
   exclusion/re-inclusion contract),
6. stages the rewritten case through the existing
   :class:`storage.Transaction` journal, or raises :class:`CurationError`
   naming the violated invariant **before** opening the Transaction.

Read-only paths (``list_cases``, ``get_case``, ``validate_case``, the pager)
take no lock and never mutate, so status/pager stay safe to run concurrently
with a writer and no nested-lock deadlock is possible.

The service imports no Rich/input/editor/HTTP code; it depends only on the
fixed schema, the mode-safe storage/journal layer, a disposable frozen-bundle
clone, and ``git_ops``."""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import yaml
from pydantic import ValidationError

from daydream import git_ops
from daydream.benchmark import schema, storage
from daydream.benchmark.schema import _schema_ready
from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict


class CurationError(Exception):
    """A curation operation violated an invariant and mutated nothing."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class StaleStateError(CurationError):
    """A curation operation ran against a stale attestation state.

    Raised when a mutation's precondition no longer holds against the freshly
    read on-disk state (e.g. a :func:`mark_ready` head SHA that no longer
    matches the snapshot) — the caller's view is stale and nothing was
    written.
    """


# Process-wide reuse cache for disposable frozen-bundle clones. Every located
# finding and every list_cases/validate_case/pager call used to open a fresh
# ``git clone --no-checkout`` of the same bundle (O(cases x findings) clone
# fan-out per operation); the cache collapses that to at most one clone per
# distinct bundle file per process. Keyed on the resolved bundle path + stat
# signature (mtime_ns, size), so a rewritten or re-targeted ``bundle_file``
# misses and is re-cloned. Clones live under ``root/cache`` (scratch, never
# part of the authoring index) for the process lifetime.
_CLONE_CACHE: dict[tuple[str, int, int], Path] = {}
_CLONE_CACHE_LOCK = threading.Lock()


def _clone_cache_key(bundle_path: Path) -> tuple[str, int, int] | None:
    """The reuse-cache key for a bundle, or None when the file vanished.

    ``(resolved path, mtime_ns, size)``: any rewrite of the bundle changes the
    signature, so a cached clone is never served for changed bytes.
    """
    try:
        st = bundle_path.stat()
    except OSError:
        return None
    return (str(bundle_path), st.st_mtime_ns, st.st_size)


@contextmanager
def _bundle_clone(root: Path, snapshot_doc: dict[str, Any]) -> Iterator[Path]:
    """A mirror-independent clone of the case's frozen bundle, reused per process.

    Clones ``snapshot.bundle_file`` (resolved via
    :func:`storage.resolve_authoring_path`) with ``--no-local --no-checkout``
    into a scratch dir under ``root/cache`` kept for the process lifetime, so
    every curation git read is served from the frozen bundle itself — the shared
    bare mirror can be deleted without making a case uncuratable. One clone per
    distinct bundle file is reused (keyed on the resolved path + stat
    signature, see :func:`_clone_cache_key`), so a case with N located findings
    costs one clone instead of N, and repeated ``list_cases``/``validate_case``
    calls reuse it instead of re-cloning per call. The clone exposes the two
    synthetic refs ``refs/remotes/origin/base`` and ``refs/remotes/origin/head``.
    Raises :class:`CurationError` when the snapshot carries no bundle or the
    bundle cannot be cloned.
    """
    bundle_rel = snapshot_doc.get("bundle_file")
    if not bundle_rel:
        raise CurationError("ready snapshot carries no bundle_file")
    root = Path(root)
    try:
        bundle_path = storage.resolve_authoring_path(root, bundle_rel)
    except storage.WorkspaceCorrupt as exc:
        # absolute / traversal bundle_file must surface as the curated
        # CurationError contract, never the storage family (the read-only
        # paths and TUI catch only CurationError).
        raise CurationError(f"invalid snapshot bundle path: {bundle_rel}") from exc
    if not bundle_path.exists():
        raise CurationError(f"snapshot bundle missing: {bundle_rel}")
    key = _clone_cache_key(bundle_path)
    if key is None:
        raise CurationError(f"snapshot bundle missing: {bundle_rel}")
    with _CLONE_CACHE_LOCK:
        cached = _CLONE_CACHE.get(key)
    if cached is not None and cached.exists():
        yield cached
        return
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    clone_dir = Path(tempfile.mkdtemp(prefix="curate-bundle-", dir=str(cache)))
    try:
        proc = git_ops._run_git(
            cache,
            ["clone", "--no-local", "--no-checkout", str(bundle_path), str(clone_dir)],
            retries=0,
            timeout=120,
        )
        if proc.returncode != 0:
            raise CurationError(
                f"bundle clone failed for {bundle_rel}: {proc.stderr.strip()}"
            )
        with _CLONE_CACHE_LOCK:
            _CLONE_CACHE[key] = clone_dir
        yield clone_dir
    finally:
        if _CLONE_CACHE.get(key) is not clone_dir:
            # a failed clone, or a clone superseded by a newer one of a
            # rewritten bundle, is not the cached entry: remove the scratch dir.
            shutil.rmtree(clone_dir, ignore_errors=True)


def _head_file_line_count(root: Path, snapshot_doc: dict[str, Any], path: str) -> int:
    """The line count of *path* in the frozen head tree (disposable bundle clone).

    Runs ``git cat-file blob refs/remotes/origin/head:<path>`` with cwd in a
    disposable ``--no-local --no-checkout`` clone of the case's frozen bundle
    under ``root/cache`` (removed on exit) — never the shared bare mirror. The
    bundle's synthetic head commit is addressed via ``refs/remotes/origin/head``
    because the original head SHA is NOT addressable inside the bundle. Raises
    :class:`CurationError` when the bundle clone cannot serve the path (the
    case cannot be a verified ``ready`` snapshot without a bundle that carried
    its head tree). A present file returns ``len(content.splitlines())``; an
    empty file has line count 0.
    """
    with _bundle_clone(root, snapshot_doc) as clone:
        proc = git_ops._run_git(
            clone, ["cat-file", "blob", f"refs/remotes/origin/head:{path}"], retries=0
        )
        if proc.returncode != 0:
            raise CurationError(
                f"location path {path!r} not present in the frozen head tree"
            )
        return len(proc.stdout.splitlines())


def _case_path(root: Path, case_id: str) -> Path:
    return Path(root) / "cases" / f"{case_id}.yaml"


def _load_case(root: Path, case_id: str) -> dict[str, Any]:
    """Load one case document strict, raising ``CurationError`` when absent."""
    path = _case_path(root, case_id)
    if not path.exists():
        raise CurationError(f"unknown case {case_id}")
    return load_yaml_strict(path)


def _with_case_lock(
    root: Path, case_id: str, op: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    """Run one curation mutation's whole read-validate-mutate-commit under the lock.

    Acquires the blocking :class:`storage.WorkspaceLock`, heals any prior
    interrupted journal via :func:`storage.recover_startup` (so a crashed
    earlier process's leftover ``committing`` journal is rolled back before a
    new write), loads the case **after** acquiring the lock (never a stale
    pre-lock view), runs ``mutate(raw)``, then stages the atomic rewrite
    through :class:`storage.Transaction`. ``mutate`` is called only with the
    lock held; its ``CurationError`` propagates and leaves the disk
    byte-unchanged.
    """
    with storage.WorkspaceLock(root):
        storage.recover_startup(root)
        raw = _load_case(root, case_id)
        mutate(raw)
        _stage_case(root, case_id, raw, op=op)


def _changed_file_stats(root: Path, case_id: str, snapshot_doc: dict[str, Any]) -> tuple[int, int]:
    """Change stats (files, lines) for a case snapshot's ``base..head`` diff.

    Only a snapshot whose ``status == "ready"`` is queried: runs ``git diff
    --numstat refs/remotes/origin/base refs/remotes/origin/head`` in a
    disposable ``--no-local --no-checkout`` clone of the case's frozen bundle
    (the synthetic base commit's tree is the true merge-base tree, so this is
    the correct merge-base diff base) and sums the per-file added/deleted
    counts. Returns ``(0, 0)`` for any non-ready snapshot. Raises
    :class:`CurationError` naming the case when the bundle clone read fails for
    a ready snapshot — counts are never fabricated.
    """
    if snapshot_doc.get("status") != "ready":
        return 0, 0
    try:
        with _bundle_clone(root, snapshot_doc) as clone:
            proc = git_ops._run_git(
                clone,
                ["diff", "--numstat", "refs/remotes/origin/base", "refs/remotes/origin/head"],
                retries=0,
            )
    except git_ops.GitError as exc:
        raise CurationError(f"case {case_id} bundle read failed: {exc}") from exc
    if proc.returncode != 0:
        raise CurationError(f"case {case_id} frozen bundle cannot serve its change diff")
    files = 0
    lines = 0
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            files += 1
            lines += int(parts[0]) + int(parts[1])
        except ValueError:
            continue
    return files, lines


def list_cases(root: Path) -> list[dict[str, Any]]:
    """Read-only index of the workspace's cases with derived curation state.

    Each row adds ``evidence_count``, ``changed_files``, and ``changed_lines``
    to the existing case_id/pr_number/state/gold_mode/gold_count/snapshot_status/
    head_prefix keys. ``evidence_count`` counts the case's full import evidence
    set (all kinds), falling back to the candidate count only when the import
    file is unreadable/missing so the resumable index never crashes.
    """
    manifest = load_yaml_strict(Path(root) / "benchmark.yaml")
    out: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        case_id = case.get("case_id")
        case_file = case.get("case_file")
        doc = load_yaml_strict(Path(root) / case_file) if case_file else {}
        raw_curation = doc.get("curation")
        curation = raw_curation if isinstance(raw_curation, dict) else {}
        findings = curation.get("findings") or []
        snapshot_doc = doc.get("snapshot") or {}
        snapshot_status = snapshot_doc.get("status", "imported")
        head_sha = snapshot_doc.get("original_head_sha") or ""
        changed_files, changed_lines = _changed_file_stats(root, case_id, snapshot_doc)
        try:
            gold_mode = schema.derive_gold_mode(_curation_model(curation))
        except ValidationError:
            gold_mode = None
        try:
            evidence_count = len(_evidence_list(root, doc))
        except WorkspaceCorrupt:
            # a missing/unreadable import file must not crash the resumable
            # index: fall back to the candidate count. Present-but-malformed
            # content (a non-list ``evidence``/``candidates`` value, a record
            # without a source_id) propagates as TypeError/KeyError instead of
            # being masked, matching get_case/validate_case — never fall back
            # for any other reason.
            evidence_count = len(doc.get("candidates") or [])
        out.append({
            "case_id": case_id,
            "pr_number": case.get("pr_number"),
            "state": curation.get("state"),
            "gold_mode": gold_mode,
            "gold_count": len(findings),
            "snapshot_status": snapshot_status,
            "head_prefix": head_sha[:12] if head_sha else "",
            "evidence_count": evidence_count,
            "changed_files": changed_files,
            "changed_lines": changed_lines,
        })
    return out


def _evidence_list(
    root: Path, raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load the import evidence once and return the ordered full record list.

    Returns every evidence record in persisted file order (all kinds — review,
    inline_comment, thread_comment, issue_comment), each augmented with
    ``candidate_index`` = index into ``raw["candidates"]`` by matching
    ``source_id``, or ``None`` when the record is not a candidate. A case that
    references an import file that is missing/unreadable raises the storage
    error — no list is fabricated.
    """
    source = raw.get("source") or {}
    import_file = source.get("import_file")
    if not import_file:
        return []
    import_data = storage.load_json_strict(Path(root) / import_file)
    candidate_index = {
        c.get("source_id"): i
        for i, c in enumerate(raw.get("candidates") or [])
    }
    records: list[dict[str, Any]] = []
    for ev in import_data.get("evidence") or []:
        record = dict(ev)
        record["candidate_index"] = candidate_index.get(record.get("source_id"))
        records.append(record)
    return records


def get_case(root: Path, case_id: str) -> dict[str, Any]:
    """Read-only view of one case document and its per-candidate evidence.

    Returns the raw case dict where each candidate gains an in-memory (never
    persisted) ``evidence`` sub-dict ``{kind, author, commit_id,
    authoring_commit_id, not_exact_reason, resolved, outdated}`` joined by
    ``source_id`` from the import file the case doc references; a candidate
    whose ``source_id`` matches no evidence record has no ``evidence`` key
    (absent, not ``None``). Additionally the case gains an in-memory
    ``evidence`` list of every import evidence record (all kinds), each
    augmented with a ``candidate_index``. A missing/unreadable import
    file for a case that references it propagates the storage error.
    """
    raw = _load_case(root, case_id)
    projection = _evidence_projection(root, raw)
    if projection:
        for cand in raw.get("candidates") or []:
            src = cand.get("source_id")
            if src in projection:
                cand["evidence"] = projection[src]
    raw["evidence"] = _evidence_list(root, raw)
    return raw


def _evidence_projection(
    root: Path, raw: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Read the import evidence once and join records by ``source_id``.

    Maps each evidence record's source_id to a read-only projection sub-dict
    (never persisted): the observed ``commit_id`` stays explanatory, the strict
    authoring commit comes from the record's ``authoring_anchor.commit_id``
    (None on a fail-closed/missing anchor — never the re-anchored id), and the
    candidate's ``not_exact_reason`` rides verbatim from the fixed closed set
    (None for an exact-acceptable candidate). A case that references an import
    file that is missing/unreadable raises the storage error — no projection
    is fabricated.
    """
    source = raw.get("source") or {}
    import_file = source.get("import_file")
    if not import_file:
        return {}
    import_data = storage.load_json_strict(Path(root) / import_file)
    candidate_reasons = {
        c.get("source_id"): c.get("not_exact_reason")
        for c in (raw.get("candidates") or [])
        if c.get("source_id")
    }
    projection: dict[str, dict[str, Any]] = {}
    for ev in import_data.get("evidence") or []:
        author = ev.get("author") or {}
        anchor = ev.get("authoring_anchor")
        anchor_commit = anchor.get("commit_id") if isinstance(anchor, dict) else None
        projection[ev["source_id"]] = {
            "kind": ev.get("kind"),
            "author": {
                "login": author.get("login"),
                "type": author.get("type"),
            },
            "commit_id": ev.get("commit_id"),
            "authoring_commit_id": anchor_commit,
            "not_exact_reason": candidate_reasons.get(ev["source_id"]),
            "resolved": ev.get("resolved", False),
            "outdated": ev.get("outdated", False),
        }
    return projection


# ---------------------------------------------------------------------------
# derivation + validation
# ---------------------------------------------------------------------------

MAX_GOLD_FINDINGS = 50


def _curation_model(curation: dict[str, Any]) -> schema.Curation:
    """Build the fixed Curation model from a raw dict.

    The persisted ``gold_mode`` audit field is not schema field, so it is
    dropped here (it is recomputed by ``derive_gold_mode`` when read); the
    ``task_spec_approved_at`` audit timestamp is stripped the same way (it
    never reaches the model, the compiled digest, or the recompile gate).
    """
    return schema.Curation(
        **{k: v for k, v in curation.items() if k not in ("gold_mode", "task_spec_approved_at")}
    )


def _snapshot_head(raw: dict[str, Any]) -> str | None:
    """The 40-hex head SHA the case's snapshot was frozen at, or None."""
    snapshot_doc = raw.get("snapshot") or {}
    return snapshot_doc.get("original_head_sha")


def _projection_matches(candidate: dict[str, Any], finding: dict[str, Any]) -> bool:
    """True when a finding is byte-identical to one candidate projection.

    Compares the canonical content projection (title/body/location) — the
    severity is always ``None`` on a candidate projection.
    """
    return (
        candidate.get("title") == finding.get("title")
        and candidate.get("body") == finding.get("body")
        and candidate.get("location") == finding.get("location")
    )


def _derive_content(raw: dict[str, Any]) -> None:
    """Derive (and persist) ``gold_status`` + ``gold_mode`` from parsed findings.

    Never caller-supplied — derived always from the resulting findings, so an
    ``--apply-gold`` fragment (or a caller) cannot forge them.
    """
    curation = raw["curation"]
    model = _curation_model(curation)
    curation["gold_status"] = schema.derive_gold_status(model)
    curation["gold_mode"] = schema.derive_gold_mode(model)


def _validate_location(root: Path, raw: dict[str, Any], finding: dict[str, Any]) -> None:
    """Location-vs-head: path present in the frozen head tree, ordered lines.

    The frozen head tree is read from a disposable clone of the case's frozen
    bundle (``refs/remotes/origin/head``), never the shared bare mirror — so a
    deleted mirror cannot make a case uncuratable.
    """
    location = finding.get("location")
    if location is None:
        return
    if not _snapshot_head(raw):
        raise CurationError("finding has a location but the snapshot carries no frozen head")
    path = location.get("path")
    start = location.get("start_line")
    end = location.get("end_line")
    if start is None or end is None:
        raise CurationError(
            f"finding location {path!r} is missing start_line and/or end_line"
        )
    line_count = _head_file_line_count(root, raw.get("snapshot") or {}, path)
    if start < 1:
        raise CurationError(f"finding location start_line {start} must be >= 1")
    if end > line_count:
        raise CurationError(
            f"finding location {path!r} end_line {end} exceeds the head file's "
            f"line count {line_count}"
        )


def _validate_raw(root: Path, case_id: str, raw: dict[str, Any]) -> None:
    """Revalidate an in-memory case doc through curation rules + the full schema.

    Raises :class:`CurationError` naming the first violated curation rule; the
    fixed-schema :class:`ValidationError` (a contract failure) propagates
    unchanged. Never writes.
    """
    curation = raw.get("curation") or {}
    findings = curation.get("findings") or []

    if len(findings) > MAX_GOLD_FINDINGS:
        raise CurationError(f"case {case_id} exceeds 50 gold findings")
    ids = [f.get("finding_id") for f in findings]
    for fid in set(ids):
        if ids.count(fid) > 1:
            raise CurationError(f"case {case_id} has duplicate finding {fid}")

    # ready => snapshot_attested and stale => not-attested are enforced by the
    # schema Curation._consistent validator.
    candidates = raw.get("candidates") or []
    for finding in findings:
        _validate_location(root, raw, finding)
        provenance = finding.get("provenance") or {}
        if provenance.get("kind") == "historical":
            srcs = provenance.get("source_ids") or []
            if len(srcs) != 1:
                raise CurationError(
                    f"case {case_id} historical finding must reference exactly one source"
                )
            src = srcs[0]
            cand = next((c for c in candidates if c.get("source_id") == src), None)
            if cand is None:
                raise CurationError(f"historical finding references unknown candidate {src}")
            if not _projection_matches(cand, finding):
                raise CurationError(
                    f"historical finding source {src} does not byte-match its candidate projection"
                )

    schema.CaseDocument(**_schema_ready(raw))


def validate_case(root: Path, case_id: str) -> None:
    """Re-validate one case through the fixed schema plus curation-service rules.

    Returns ``None`` on success; raises :class:`CurationError` on the first
    curation-rule violation and lets the fixed-schema
    :class:`ValidationError` propagate as a contract failure. Never writes.
    """
    raw = _load_case(root, case_id)
    _validate_raw(root, case_id, raw)
    return None


def _stage_case(root: Path, case_id: str, raw: dict[str, Any], *, op: str) -> None:
    """Validate in-memory then atomically rewrite the single case YAML.

    The full validity check (curation rules + :class:`schema.CaseDocument`) runs
    **before** the :class:`storage.Transaction` opens, so a rejected mutation
    leaves the on-disk case byte-unchanged. No manifest is staged — curation
    state lives only in the case YAML.
    """
    _validate_raw(root, case_id, raw)
    with storage.Transaction(root, op_id=f"curate-{case_id}", kind=f"curation:{op}") as tx:
        tx.stage(
            f"cases/{case_id}.yaml",
            yaml.safe_dump(raw, sort_keys=False).encode("utf-8"),
        )
        tx.commit()


def accept_candidate(root: Path, case_id: str, source_id: str) -> None:
    """Accept one exact-exceptable candidate as a byte-identical ``historical`` finding.

    The finding's title/body/severity/location are taken straight from the
    candidate projection, ``provenance.kind`` is ``historical`` (the only path
    that produces it), and ``finding_id`` is derived from the content.
    """

    def mutate(raw: dict[str, Any]) -> None:
        candidate = next(
            (c for c in (raw.get("candidates") or []) if c.get("source_id") == source_id),
            None,
        )
        if candidate is None:
            raise CurationError(f"no candidate {source_id} in case {case_id}")
        if not candidate.get("exact_acceptable"):
            raise CurationError(f"candidate {source_id} is not exact_acceptable")

        curation = raw.setdefault("curation", {})
        _reopen_for_mutation(curation)
        finding = {
            "title": candidate["title"],
            "body": candidate["body"],
            "severity": candidate.get("severity"),
            "location": candidate.get("location"),
            "provenance": {"kind": "historical", "source_ids": [source_id]},
        }
        finding["finding_id"] = schema.derive_finding_id(finding, case_id=case_id)
        raw.setdefault("curation", {}).setdefault("findings", []).append(finding)
        _derive_content(raw)

    _with_case_lock(root, case_id, "accept", mutate)


def _derive_provenance_kind(
    source_ids: list[str], *, authored: bool = False
) -> str:
    """Derive the provenance kind from source IDs + authoring intent.

    ``historical`` is produced ONLY by :func:`accept_candidate` and never here:
    additions/replacements are ``authored`` (no source, or explicit authoring)
    or ``edited`` (rewrites of one or more sources).
    """
    if authored:
        return "authored"
    if not source_ids:
        return "authored"
    return "edited"


def _evidence_source_ids(root: Path, raw: dict[str, Any]) -> set[str]:
    """The source_ids of every import evidence record the case references."""
    return set(_evidence_projection(root, raw))


def _check_evidence_sources(
    root: Path, raw: dict[str, Any], source_ids: list[str], case_id: str
) -> None:
    """Reject a caller-supplied source reference that no import evidence backs."""
    if not source_ids:
        return
    evidence_ids = _evidence_source_ids(root, raw)
    for src in source_ids:
        if src not in evidence_ids:
            raise CurationError(f"source {src} is not evidence of case {case_id}")


def _append_atoms_to_case(
    root: Path, raw: dict[str, Any], case_id: str, atoms: list[dict[str, Any]],
    *, authored: bool, require_sources: bool,
) -> None:
    """Shared mutate body of the three atomic-add siblings.

    Validates every atom's sources up front (all-or-nothing), reopens the case
    for mutation, appends each atom via :func:`_build_replacement` (the single
    finding builder, never duplicated inline), and re-derives gold status.
    With *require_sources* an atom carrying no ``source_ids`` is rejected
    naming its index — it would otherwise silently derive ``authored``.
    """
    for i, atom in enumerate(atoms):
        source_ids = list(atom.get("source_ids") or [])
        if require_sources and not source_ids:
            raise CurationError(f"edited-finding atom {i} carries no source_ids")
        _check_evidence_sources(root, raw, source_ids, case_id)
    curation = raw.setdefault("curation", {})
    _reopen_for_mutation(curation)
    for atom in atoms:
        curation.setdefault("findings", []).append(
            _build_replacement(root, raw, case_id, atom, authored=authored)
        )
    _derive_content(raw)


def add_finding(
    root: Path,
    case_id: str,
    *,
    title: str,
    body: str,
    severity: str | None = None,
    location: dict[str, Any] | None = None,
    source_ids: list[str] | None = None,
) -> None:
    """Add an authored (new) finding. provenance is ``authored`` with empty sources."""
    atom = {"title": title, "body": body, "severity": severity,
            "location": location, "source_ids": source_ids or []}

    def mutate(raw: dict[str, Any]) -> None:
        _append_atoms_to_case(
            root, raw, case_id, [atom], authored=True, require_sources=False
        )

    _with_case_lock(root, case_id, "add", mutate)


def add_findings(
    root: Path, case_id: str, *, findings: list[dict[str, Any]]
) -> None:
    """Atomically add a batch of authored findings in one transaction.

    Unlike a loop of :func:`add_finding` calls, a mid-batch invariant violation
    stages **nothing** — the whole batch validates (evidence sources checked
    up front), derives, and stages together, so the TUI's multi-atom ``[n]``
    add stays all-or-nothing and never leaves earlier atoms persisted on a
    failure.
    """

    def mutate(raw: dict[str, Any]) -> None:
        _append_atoms_to_case(
            root, raw, case_id, findings, authored=True, require_sources=False
        )

    _with_case_lock(root, case_id, "add", mutate)


def add_edited_findings(
    root: Path, case_id: str, *, atoms: list[dict[str, Any]]
) -> None:
    """Atomically add re-written (edited) findings in one transaction.

    The split/merge/author-from-evidence engine: N atoms sharing one source
    split it into N findings, one atom carrying M sources merges them into
    one finding. Every atom must carry a non-empty ``source_ids`` list whose
    members are import evidence of the case — an empty list is rejected with a
    :class:`CurationError` naming the atom index (it would otherwise silently
    derive ``authored``) — so the resulting findings are always ``edited``,
    never ``historical`` and never ``authored``. Finding IDs are derived, never
    caller-supplied. Like :func:`add_findings`, the whole batch validates
    (sources checked up front), derives, and stages together: a mid-batch
    invariant violation stages nothing.
    """

    def mutate(raw: dict[str, Any]) -> None:
        _append_atoms_to_case(
            root, raw, case_id, atoms, authored=False, require_sources=True
        )

    _with_case_lock(root, case_id, "add-edited", mutate)


def _build_replacement(
    root: Path, raw: dict[str, Any], case_id: str, replacement: dict[str, Any],
    *, authored: bool = False,
) -> dict[str, Any]:
    """Build one finding from a replacement atom; provenance kind driven by *authored*.

    The single finding builder shared by the atomic-adds (via
    :func:`_append_atoms_to_case`) and replacement paths; callers validate
    sources up front, so this is a pure dict construction.
    """
    source_ids = list(replacement.get("source_ids") or [])
    finding = {
        "title": replacement["title"],
        "body": replacement["body"],
        "severity": replacement.get("severity"),
        "location": replacement.get("location"),
        "provenance": {
            "kind": _derive_provenance_kind(source_ids, authored=authored),
            "source_ids": source_ids,
        },
    }
    finding["finding_id"] = schema.derive_finding_id(finding, case_id=case_id)
    return finding


def replace_findings(
    root: Path, case_id: str, finding_id: str, *, replacements: list[dict[str, Any]]
) -> None:
    """Replace one finding with an atomic set of re-written (edited) findings.

    Supports split (N replacements expand one finding) and merge (each
    replacement concatenates its own sources); the replacements are the atomic
    toehold set and are revalidated with the whole case.
    """

    def mutate(raw: dict[str, Any]) -> None:
        curation = raw.setdefault("curation", {})
        _reopen_for_mutation(curation)
        findings = curation.setdefault("findings", [])
        index = next(
            (i for i, f in enumerate(findings) if f.get("finding_id") == finding_id),
            None,
        )
        if index is None:
            raise CurationError(f"no finding {finding_id}")
        for r in replacements:
            _check_evidence_sources(root, raw, list(r.get("source_ids") or []), case_id)
        built = [_build_replacement(root, raw, case_id, r) for r in replacements]
        new_findings = list(findings[:index]) + built + list(findings[index + 1:])
        curation["findings"] = new_findings
        _derive_content(raw)

    _with_case_lock(root, case_id, "replace", mutate)


_EVIDENCE_REASONS = frozenset({
    "fixed_before_snapshot",
    "not_actionable",
    "incorrect",
    "duplicate",
    "style_only",
    "out_of_scope",
    "other",
})


def _set_clean(curation: dict[str, Any]) -> None:
    """Attest one case's gold set as reviewed-clean (deterministic derivation).

    Sets clean-attested and the clean gold status + mode, and clears any
    snapshot attestation (a clean-attested case is never ready-attested by this
    path). The state must already be reopened for mutation (:func:`_reopen_for_mutation`
    clears a ready/stale case's snapshot attestation); ``_set_clean`` re-clears it
    so the audit fields stay internally consistent. Shared by :func:`attest_clean`
    and the :func:`apply_gold_fragment` clean branch so the clean-attestation
    triple lives in one place.
    """
    curation["snapshot_attested"] = False
    curation["clean_attested"] = True
    curation["gold_status"] = "clean"
    curation["gold_mode"] = "clean"


def _append_evidence_exclusion(
    curation: dict[str, Any], source_id: str, reason: str, note: str | None
) -> None:
    """Append (or replace) one evidence-exclusion row (last-wins per source).

    Shared by :func:`exclude_evidence` and :func:`apply_gold_fragment` so the
    single-row insertion stays in one place.
    """
    exclusions = [
        e for e in curation.get("exclusions") or [] if e.get("source_id") != source_id
    ]
    exclusions.append({"source_id": source_id, "reason": reason, "note": note})
    curation["exclusions"] = exclusions


def exclude_evidence(
    root: Path, case_id: str, source_id: str, *, reason: str, note: str | None = None
) -> None:
    """Exclude one evidence source from gold with the fixed reason/note contract.

    Idempotent last-wins: re-excluding an already-excluded source replaces the
    existing row (a single entry per source). ``reason == "other"`` requires a
    non-blank note; a stray note on any other reason is rejected.
    """

    def mutate(raw: dict[str, Any]) -> None:
        _validate_evidence_exclusion_contract(reason, note)
        _check_evidence_sources(root, raw, [source_id], case_id)

        curation = raw.setdefault("curation", {})
        _reopen_for_mutation(curation)
        _append_evidence_exclusion(curation, source_id, reason, note)

    _with_case_lock(root, case_id, "exclude-evidence", mutate)


def exclude_evidence_batch(
    root: Path, case_id: str, source_ids: list[str], *, reason: str, note: str | None = None
) -> None:
    """Exclude several evidence sources from gold in one atomic transaction.

    The batch sibling of :func:`exclude_evidence`: validates the reason/note
    contract and every source up front (all-or-nothing), then appends the whole
    selection under one lock+transaction, so a mid-selection service failure
    stages nothing and cannot leave earlier sources committed. Matches the
    atomic ``add_findings``/``add_edited_findings`` mutation family the TUI
    uses for its multi-method actions.
    """

    def mutate(raw: dict[str, Any]) -> None:
        _validate_evidence_exclusion_contract(reason, note)
        for source_id in source_ids:
            _check_evidence_sources(root, raw, [source_id], case_id)
        curation = raw.setdefault("curation", {})
        _reopen_for_mutation(curation)
        for source_id in source_ids:
            _append_evidence_exclusion(curation, source_id, reason, note)

    _with_case_lock(root, case_id, "exclude-evidence", mutate)


def _validate_transition(frm: str | None, to: str) -> None:
    """Enforce one case state transition, exposing :class:`CurationError`.

    The schema helper raises :class:`schema.TransitionError`, but the service
    contract promises :class:`CurationError`; translate the message so library
    and CLI callers see a single exception family.
    """
    try:
        schema.validate_case_transition(cast("str", frm), to)
    except schema.TransitionError as exc:
        raise CurationError(str(exc)) from exc


def _invalidate_task_spec_approval(curation: dict[str, Any]) -> None:
    """Clear a persisted task-spec approval (approved digest + audit stamp).

    ``mark_ready`` persists the pair together and every gold/provenance/
    evidence mutation (or a ready demotion, case (re)import, or case
    exclusion) that makes a previously-approved spec no longer reflect the
    case must clear both together so they can never diverge. Shared by
    :func:`_demote_ready`, the stale branch of :func:`_reopen_for_mutation`,
    :func:`_apply_case_exclusion`, and the re-import path in ``github_import``
    — the exact four call sites that previously inlined the identical
    two-pop pair.
    """
    curation.pop("task_spec_sha256", None)
    curation.pop("task_spec_approved_at", None)


def _demote_ready(curation: dict[str, Any]) -> str | None:
    """Demote a ``ready`` case to ``draft``, clearing attestation.

    Returns the resulting state (``draft`` after a demotion, else the current
    state untouched) so callers that route onward to another terminal state
    (case-exclude, apply-gold) can drive the next transition.
    """
    state = curation.get("state")
    if state == "ready":
        _validate_transition("ready", "draft")
        curation["state"] = "draft"
        curation["snapshot_attested"] = False
        _invalidate_task_spec_approval(curation)
        return "draft"
    return state


def _reopen_for_mutation(curation: dict[str, Any]) -> dict[str, Any]:
    """Apply the plan §7 state discipline to every gold/provenance/evidence mutation.

    - a ``ready`` case first goes ``ready -> draft`` (clearing attestation);
    - a ``stale`` case stays ``stale`` but clears attestation;
    - a ``draft`` is already draft;
    - an ``excluded``/``unreplayable`` case rejects gold mutations (only
      re/include or case-exclude paths apply there).
    """
    state = curation.get("state")
    if state == "ready":
        _demote_ready(curation)
    elif state == "stale":
        curation["snapshot_attested"] = False
        _invalidate_task_spec_approval(curation)
    elif state in ("excluded", "unreplayable"):
        raise CurationError(
            f"gold mutations are rejected on a {state} case (re-include or case-exclude it first)"
        )
    return curation


def mark_ready(root: Path, case_id: str, *, head_sha: str, task_spec_sha256: str | None = None) -> None:
    """The final-attest operation: the one path that sets a case ready + attested.

    SHA-specific confirmation: *head_sha* must equal the snapshot's original
    head SHA, and the current state must move to ``ready`` (draft -> ready or
    stale -> ready). Sets ``snapshot_attested=True`` and ``state=ready`` after
    the full case revalidates. Records the human-approved Task.md digest
    (*task_spec_sha256*); when *task_spec_sha256* is omitted the digest is
    derived under the workspace lock from the exact case state that is about
    to be persisted, so the approved digest can never go stale and abort a
    later whole-workspace compile. Also records a persisted-but-stripped
    ``task_spec_approved_at`` audit timestamp -- there is no approval without
    a digest (R7).
    """

    def mutate(raw: dict[str, Any]) -> None:
        snapshot_doc = raw.get("snapshot") or {}
        original = snapshot_doc.get("original_head_sha")
        if head_sha != original:
            raise StaleStateError(
                f"attestation SHA mismatch: expected {original} got {head_sha}"
            )
        curation = raw.setdefault("curation", {})
        # Single-sourced empty-gold eligibility: derive_gold_status is None
        # exactly when the gold set is empty and never clean-attested -- the
        # same derived status harbor/build._is_compilable trusts.
        if schema.derive_gold_status(_curation_model(curation)) is None:
            raise CurationError(
                f"case {case_id} cannot be marked ready with an empty gold findings set "
                "and no clean attestation (clean-attest first)"
            )
        _validate_transition(curation.get("state"), "ready")
        stored_task_spec_sha256 = task_spec_sha256
        if stored_task_spec_sha256 is None:
            from daydream.benchmark.harbor import build

            stored_task_spec_sha256 = build.task_spec_digest(raw)
        curation["state"] = "ready"
        curation["snapshot_attested"] = True
        curation["task_spec_sha256"] = stored_task_spec_sha256
        curation["task_spec_approved_at"] = datetime.now(timezone.utc).isoformat()
        _derive_content(raw)

    _with_case_lock(root, case_id, "mark-ready", mutate)


def attest_clean(root: Path, case_id: str) -> None:
    """Attest a case reviewed-clean (only when its gold set is empty).

    Sets ``clean_attested=True`` and the clean gold status; never sets
    ``snapshot_attested`` and never marks ready — the final-attest operation
    remains required.
    """

    def mutate(raw: dict[str, Any]) -> None:
        curation = raw.setdefault("curation", {})
        if curation.get("findings"):
            raise CurationError(
                f"case {case_id} has gold findings; clean attestation requires an empty gold set"
            )
        # Route through the mutation discipline: a ready/stale case reopens to
        # draft (clearing snapshot attestation) instead of lingering as
        # ready-but-unattested, which the revalidation guard forbids. A draft case
        # stays draft -- the final-attest op remains required.
        _reopen_for_mutation(curation)
        _set_clean(curation)

    _with_case_lock(root, case_id, "attest-clean", mutate)


_CASE_EXCLUSION_REASONS = frozenset({"unreplayable", "not_suitable", "duplicate_case", "other"})


def _apply_case_exclusion(
    curation: dict[str, Any], *, reason: str, note: str | None
) -> None:
    """Route any case to ``excluded`` under the case reason/note contract.

    Demotes a ``ready`` case (``ready -> draft``, clearing attestation) then
    routes the resulting state to ``excluded``. Shared by both
    :func:`exclude_case` and :func:`apply_gold_fragment` so the transition
    block lives in one place.
    """
    _validate_case_exclusion_contract(reason, note)
    state = _demote_ready(curation)
    _validate_transition(state, "excluded")
    curation["state"] = "excluded"
    curation["snapshot_attested"] = False
    # A ready case's task-spec approval was already invalidated by
    # _demote_ready's ready->draft step; no non-ready state carries one.
    curation["case_exclusion"] = {"reason": reason, "note": note}


def exclude_case(
    root: Path, case_id: str, reason: str, *, note: str | None = None
) -> None:
    """Exclude an entire case from the dataset with the case reason/note contract.

    Routes through the fixed transition table: from ``draft``/``stale``/
    ``unreplayable`` directly to ``excluded``; from ``ready`` via the
    ``ready -> draft -> excluded`` double edge (clearing attestation on the
    ``ready -> draft`` step).
    """

    def mutate(raw: dict[str, Any]) -> None:
        curation = raw.setdefault("curation", {})
        _apply_case_exclusion(curation, reason=reason, note=note)

    _with_case_lock(root, case_id, "exclude-case", mutate)


def reinclude_case(root: Path, case_id: str) -> None:
    """Re-include an excluded case to the state its snapshot supports.

    A ready-snapshot case re-includes to ``draft``; an unreplayable-snapshot
    case to ``unreplayable``. Requires the case be currently ``excluded``.
    """

    def mutate(raw: dict[str, Any]) -> None:
        curation = raw.setdefault("curation", {})
        if curation.get("state") != "excluded":
            raise CurationError(f"case {case_id} is not excluded")
        snapshot_doc = raw.get("snapshot") or {}
        destination = "draft" if snapshot_doc.get("status") == "ready" else "unreplayable"
        _validate_transition("excluded", destination)
        curation["state"] = destination
        curation["snapshot_attested"] = False
        curation["case_exclusion"] = None

    _with_case_lock(root, case_id, "reinclude-case", mutate)


def _fragment_provenance(
    root: Path, raw: dict[str, Any], finding: dict[str, Any], source_ids: list[str],
    *,
    case_id: str,
) -> tuple[str, list[str]]:
    """Derive provenance kind from a fragment finding's source IDs + match.

    ``historical`` only when exactly one source whose projection byte-matches
    the finding; else ``edited`` (>=1 source) or ``authored`` (no source). The
    fragment's own kind is never trusted.
    """
    _check_evidence_sources(root, raw, source_ids, case_id)
    if len(source_ids) == 1:
        cand = next(
            (c for c in (raw.get("candidates") or []) if c.get("source_id") == source_ids[0]),
            None,
        )
        if cand is not None and _projection_matches(cand, finding):
            return "historical", source_ids
    return _derive_provenance_kind(source_ids, authored=False), source_ids


def apply_gold_fragment(root: Path, case_id: str, fragment: dict[str, Any]) -> None:
    """Apply a reviewed gold YAML fragment through the service derivation path.

    Strips the caller-supplied ``finding_id`` / ``provenance`` / ``state`` /
    ``gold_status`` / ``gold_mode`` from every finding before derivation, so a
    forged value in any of them is discarded and recomputed. Reuses the
    exclusion and case-exclusion reason/note contracts. Always leaves a
    ready-snapshot case ``draft`` and never sets ``snapshot_attested`` — it
    can never produce ready gold.
    """

    def mutate(raw: dict[str, Any]) -> None:
        curation = raw.setdefault("curation", {})
        _reopen_for_mutation(curation)

        findings: list[dict[str, Any]] = []
        for frag in fragment.get("findings") or []:
            finding = {
                "title": frag["title"],
                "body": frag["body"],
                "severity": frag.get("severity"),
                "location": frag.get("location"),
            }
            kind, source_ids = _fragment_provenance(
                root, raw, finding, list(frag.get("source_ids") or []), case_id=case_id
            )
            finding["provenance"] = {"kind": kind, "source_ids": source_ids}
            finding["finding_id"] = schema.derive_finding_id(finding, case_id=case_id)
            findings.append(finding)
        curation["findings"] = findings

        for exc in fragment.get("exclusions") or []:
            src = exc["source_id"]
            reason = exc["reason"]
            note = exc.get("note")
            _validate_evidence_exclusion_contract(reason, note)
            _check_evidence_sources(root, raw, [src], case_id)
            _append_evidence_exclusion(curation, src, reason, note)
        curation["exclusions"] = curation.get("exclusions") or []

        case_exclusion = fragment.get("case_exclusion")
        if case_exclusion is not None:
            _apply_case_exclusion(
                curation, reason=case_exclusion["reason"], note=case_exclusion.get("note")
            )

        clean = bool(fragment.get("clean"))
        if clean:
            if findings:
                raise CurationError(
                    f"case {case_id} clean fragment requires an empty gold findings set"
                )
            _set_clean(curation)
        else:
            _derive_content(raw)

    _with_case_lock(root, case_id, "apply-gold", mutate)


def _validate_exclusion_contract(
    reason: str, note: str | None, *, valid_reasons: frozenset[str], noun: str
) -> None:
    """Reason/note contract shared by the evidence- and case-exclusion paths."""
    if reason not in valid_reasons:
        raise CurationError(f"invalid {noun} exclusion reason {reason!r}")
    if reason == "other":
        if not note or not str(note).strip():
            raise CurationError(f"{noun} exclusion reason 'other' requires a note")
    elif note is not None:
        raise CurationError(f"{noun} exclusion note is only valid for reason 'other'")


def _validate_evidence_exclusion_contract(reason: str, note: str | None) -> None:
    """Evidence-level reason/note contract (shared by exclude and fragment)."""
    _validate_exclusion_contract(reason, note, valid_reasons=_EVIDENCE_REASONS, noun="evidence")


def _validate_case_exclusion_contract(reason: str, note: str | None) -> None:
    """Case-level reason/note contract (shared by exclude and apply-gold)."""
    _validate_exclusion_contract(reason, note, valid_reasons=_CASE_EXCLUSION_REASONS, noun="case")
