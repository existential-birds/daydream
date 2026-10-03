"""Golden-review curation over private, frozen benchmark workspaces.

Mutations acquire the workspace lock, recover interrupted transactions, then read
the latest case. The service derives ids, provenance, gold status, and transitions;
full schema, location, duplicate, cap, historical-match, and exclusion validation
precedes transactional staging. Read-only views take no lock and never write.

Frozen-bundle clones provide Git evidence independently of the shared mirror.
The service owns no terminal, browser, editor, or network interaction.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import yaml
from pydantic import ValidationError

from daydream import git_ops
from daydream.benchmark import schema, storage
from daydream.benchmark.schema import _schema_ready
from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict
from daydream.git_ops import process as git_process


class CurationError(Exception):
    """A curation operation violated an invariant and mutated nothing."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class StaleStateError(CurationError):
    """A mutation precondition no longer matches locked on-disk state; nothing was written."""


# Process-lifetime clones live under root/cache, outside the authoring index.
# Resolved path plus mtime/size invalidates reuse after bundle replacement.
_CLONE_CACHE: dict[tuple[str, int, int], Path] = {}
_CLONE_CACHE_LOCK = threading.Lock()


def _clone_cache_key(bundle_path: Path) -> tuple[str, int, int] | None:
    """Resolved bundle path, mtime_ns, and size; return None if the file vanished."""
    try:
        st = bundle_path.stat()
    except OSError:
        return None
    return (str(bundle_path), st.st_mtime_ns, st.st_size)


@contextmanager
def _bundle_clone(root: Path, snapshot_doc: dict[str, Any]) -> Iterator[Path]:
    """Read Git evidence from a cached frozen-bundle clone, independent of the mirror."""
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
        proc = git_process._run_git(
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
    """Count lines at refs/remotes/origin/head in the frozen-bundle clone."""
    with _bundle_clone(root, snapshot_doc) as clone:
        proc = git_process._run_git(
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


def _changed_file_stats(root: Path, case_id: str, snapshot_doc: dict[str, Any]) -> tuple[int, int]:
    """Sum base-to-head numstat files and lines from the frozen bundle."""
    if snapshot_doc.get("status") != "ready":
        return 0, 0
    try:
        with _bundle_clone(root, snapshot_doc) as clone:
            proc = git_process._run_git(
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
    """Read case state with evidence and frozen-diff counts."""
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
            # Only absent/unreadable imports fall back. Malformed loaded content
            # still raises TypeError/KeyError, as get_case and validate_case do.
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
    """Load all evidence in persisted order and attach candidate indices by source ID."""
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
    """Load one read-only case view with full evidence and candidate provenance."""
    raw = _load_case(root, case_id)
    records = _evidence_list(root, raw)
    projection = _evidence_projection(records, raw)
    if projection:
        for cand in raw.get("candidates") or []:
            src = cand.get("source_id")
            if src in projection:
                cand["evidence"] = projection[src]
    raw["evidence"] = records
    raw["prioritized_evidence"] = prioritized_evidence(raw)
    return raw


def _evidence_projection(
    records: list[dict[str, Any]], raw: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Project already-loaded records by source id for the candidate view."""
    candidate_reasons = {
        c.get("source_id"): c.get("not_exact_reason")
        for c in (raw.get("candidates") or [])
        if c.get("source_id")
    }
    projection: dict[str, dict[str, Any]] = {}
    for ev in records:
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


# prioritized evidence projection (issue #879)

# The seven bands in fixed display order (lowest rank first).
BAND_RANK: dict[str, int] = {
    "review_first": 0,
    "needs_judgment": 1,
    "possibly_actioned": 2,
    "likely_actioned": 3,
    "withdrawn": 4,
    "context": 5,
    "decided": 6,
}

# The closed reason-code set, in fixed display order (signal codes, then
# availability causes, then classification causes).
REASON_CODES: tuple[str, ...] = (
    "resolved",
    "outdated",
    "anchor-delta-changed",
    "anchor-delta-deleted",
    "anchor-delta-renamed",
    "anchor-delta-binary",
    "pr-author-reply",
    "commit-non-ancestor",
    "commit-unavailable",
    "anchor-unavailable",
    "facts-missing",
    "dismissed",
    "decided-by-finding",
    "decided-by-exclusion",
    "decided-by-conflict",
    "non-candidate",
)

_DELTA_SIGNAL_CODES = {
    "changed": "anchor-delta-changed",
    "deleted": "anchor-delta-deleted",
    "renamed": "anchor-delta-renamed",
    "binary": "anchor-delta-binary",
}


def _reply_parent_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index inline comments by REST database ID and GraphQL node ID."""
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("kind") != "inline_comment":
            continue
        db_id = rec.get("database_id")
        if db_id is not None:
            index.setdefault(str(db_id), rec)
            index.setdefault(db_id, rec)
        node_id = rec.get("node_id")
        if node_id:
            index.setdefault(node_id, rec)
    return index


def _thread_key(record: dict[str, Any], parents: dict[str, dict[str, Any]]) -> str | None:
    """Use GraphQL thread identity, or the top of the REST reply chain."""
    seen: set[Any] = set()
    cur = record
    while not cur.get("thread_id"):
        source_id = cur.get("source_id")
        if source_id is not None and source_id in seen:
            break
        if source_id is not None:
            seen.add(source_id)
        reply_to_id = cur.get("reply_to_id")
        parent = parents.get(reply_to_id) if reply_to_id else None
        if parent is None:
            break
        cur = parent
    return cur.get("thread_id") or cur.get("source_id")


def _reply_context(
    records: list[dict[str, Any]], pr_login: str | None
) -> tuple[dict[str, dict[str, Any]], dict[Any, str]]:
    """Index parent chains and each thread's latest PR-author reply once."""
    parents = _reply_parent_index(records)
    latest: dict[Any, str] = {}
    for record in records:
        created = record.get("created_at")
        if not pr_login or not created or not record.get("reply_to_id"):
            continue
        if (record.get("author") or {}).get("login") != pr_login:
            continue
        key = _thread_key(record, parents)
        latest[key] = max(latest.get(key, created), created)
    return parents, latest


def collect_signals(record: dict[str, Any], view_context: dict[str, Any]) -> frozenset[str]:
    """Return distinct resolved, outdated, anchor-delta, and later-PR-author-reply signals."""
    signals = {key for key in ("resolved", "outdated") if record.get(key)}
    source_id = record.get("source_id")
    facts = view_context.get("facts") or {}
    for group in ("candidates", "non_candidates"):
        entry = (facts.get(group) or {}).get(source_id) or {}
        code = _DELTA_SIGNAL_CODES.get(cast(str, entry.get("anchor_delta")))
        if code:
            signals.add(code)
    pr_login = view_context.get("pr_author_login")
    created = record.get("created_at")
    if pr_login and created:
        latest = view_context.get("latest_pr_author_reply")
        parents = view_context.get("reply_parents")
        if latest is None or parents is None:
            parents, latest = _reply_context(view_context.get("records") or [], pr_login)
        later = latest.get(_thread_key(record, parents))
        if later is not None and later > created:
            signals.add("pr-author-reply")
    return frozenset(signals)


def classify_evidence(
    *,
    refs: set[str],
    is_candidate: bool,
    dismissed: bool,
    signals: frozenset[str] | set[str],
    facts_present: bool,
    commit_relation: str,
    anchor_delta: str,
) -> tuple[str, str, list[str]]:
    """Classify first-match: decided, context, withdrawn, then signal counts."""
    decided = refs & {"finding", "exclusion"}
    if decided:
        disposition = next(iter(decided)) if len(decided) == 1 else "conflict"
        return "decided", disposition, [f"decided-by-{disposition}"]
    if not is_candidate:
        return "context", "n/a", ["non-candidate"]
    if dismissed:
        return "withdrawn", "undecided", ["dismissed"]
    if signals:
        band = "likely_actioned" if len(signals) >= 2 else "possibly_actioned"
        rank = {reason: index for index, reason in enumerate(REASON_CODES)}
        return band, "undecided", sorted(signals, key=rank.__getitem__)
    for needs_judgment, reason in (
        (commit_relation == "non_ancestor", "commit-non-ancestor"),
        (commit_relation == "unavailable", "commit-unavailable"),
        (anchor_delta == "unavailable", "anchor-unavailable"),
        (not facts_present, "facts-missing"),
    ):
        if needs_judgment:
            return "needs_judgment", "undecided", [reason]
    return "review_first", "undecided", []


def prioritized_evidence(raw: dict[str, Any]) -> dict[str, Any]:
    """Project loaded evidence into entries and by_source without Git, network, or writes."""
    records = raw.get("evidence") or []
    facts = raw.get("prioritization") or None
    curation = raw.get("curation") or {}
    not_exact = {
        c.get("source_id"): c.get("not_exact_reason")
        for c in (raw.get("candidates") or [])
        if c.get("source_id")
    }
    finding_refs = {
        sid
        for f in (curation.get("findings") or [])
        for sid in ((f.get("provenance") or {}).get("source_ids") or [])
    }
    exclusion_refs = {
        e.get("source_id") for e in (curation.get("exclusions") or []) if e.get("source_id")
    }
    pr_login = ((raw.get("pull_request") or {}).get("author") or {}).get("login")
    reply_parents, latest_pr_author_reply = _reply_context(records, pr_login)
    view_context = {
        "pr_author_login": pr_login,
        "records": records,
        "facts": facts,
        "latest_pr_author_reply": latest_pr_author_reply,
        "reply_parents": reply_parents,
    }
    thread_members: dict[str, list[str]] = {}
    for rec in records:
        tid = rec.get("thread_id")
        if tid is not None and rec.get("source_id"):
            thread_members.setdefault(tid, []).append(rec["source_id"])
    entries: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, Any]] = {}
    for position, rec in enumerate(records):
        sid = rec.get("source_id")
        group = "candidates" if sid in not_exact else "non_candidates"
        fentry = ((facts or {}).get(group) or {}).get(sid)
        facts_present = fentry is not None
        refs: set[str] = set()
        if sid in finding_refs:
            refs.add("finding")
        if sid in exclusion_refs:
            refs.add("exclusion")
        signals = collect_signals(rec, view_context)
        band, disposition, reasons = classify_evidence(
            refs=refs,
            is_candidate=sid in not_exact,
            dismissed=bool(rec.get("dismissed")),
            signals=signals,
            facts_present=facts_present,
            commit_relation=(fentry or {}).get("commit_relation") or "at_head",
            anchor_delta=(fentry or {}).get("anchor_delta") or "unchanged",
        )
        tid = rec.get("thread_id")
        entry = {
            "source_id": sid,
            "position": position,
            "band": band,
            "reasons": reasons,
            "not_exact_reason": not_exact.get(sid),
            "disposition": disposition,
            "thread_id": tid,
            "same_thread_ids": sorted(
                s for s in thread_members.get(tid, []) if s != sid
            )
            if tid is not None
            else [],
        }
        entries.append(entry)
        by_source[sid] = entry
    entries.sort(
        key=lambda e: (BAND_RANK[e["band"]], e["position"], e["source_id"])
    )
    return {"entries": entries, "by_source": by_source}


# derivation + validation

MAX_GOLD_FINDINGS = 50


def _curation_model(curation: dict[str, Any]) -> schema.Curation:
    """Parse curation without derived gold_mode or the task-spec approval audit timestamp."""
    return schema.Curation(
        **{k: v for k, v in curation.items() if k not in ("gold_mode", "task_spec_approved_at")}
    )


def _snapshot_head(raw: dict[str, Any]) -> str | None:
    """The 40-hex head SHA the case's snapshot was frozen at, or None."""
    snapshot_doc = raw.get("snapshot") or {}
    return snapshot_doc.get("original_head_sha")


def _projection_matches(candidate: dict[str, Any], finding: dict[str, Any]) -> bool:
    """Compare canonical title, body, and location; candidate severity is always None."""
    return (
        candidate.get("title") == finding.get("title")
        and candidate.get("body") == finding.get("body")
        and candidate.get("location") == finding.get("location")
    )


def _derive_content(raw: dict[str, Any]) -> None:
    """Derive stored gold status and mode from findings, ignoring caller claims."""
    curation = raw["curation"]
    model = _curation_model(curation)
    curation["gold_status"] = schema.derive_gold_status(model)
    curation["gold_mode"] = schema.derive_gold_mode(model)


def _derive_provenance_kind(
    source_ids: list[str]
) -> str:
    """No sources means authored; source rewrites mean edited, never historical."""
    return "authored" if not source_ids else "edited"


def _evidence_source_ids(root: Path, raw: dict[str, Any]) -> set[str]:
    """The source_ids of every import evidence record the case references."""
    return set(_evidence_projection(_evidence_list(root, raw), raw))


def _check_evidence_sources(
    root: Path, raw: dict[str, Any], source_ids: Iterable[str], case_id: str
) -> None:
    """Validate references in input order, loading their import projection only once."""
    evidence_ids = None
    for src in source_ids:
        if evidence_ids is None:
            evidence_ids = _evidence_source_ids(root, raw)
        if src not in evidence_ids:
            raise CurationError(f"source {src} is not evidence of case {case_id}")


def _build_finding(
    case_id: str, replacement: dict[str, Any],
    *, kind: str | None = None,
) -> dict[str, Any]:
    """Construct trusted finding fields, provenance, and a content-derived finding id."""
    source_ids = list(replacement.get("source_ids") or [])
    finding = {
        "title": replacement["title"],
        "body": replacement["body"],
        "severity": replacement.get("severity"),
        "location": replacement.get("location"),
        "provenance": {
            "kind": kind if kind is not None else _derive_provenance_kind(source_ids),
            "source_ids": source_ids,
        },
    }
    finding["finding_id"] = schema.derive_finding_id(finding, case_id=case_id)
    return finding


def _set_clean(curation: dict[str, Any]) -> None:
    """Set clean attestation and derived clean status while clearing snapshot attestation."""
    curation["snapshot_attested"] = False
    curation["clean_attested"] = True
    curation["gold_status"] = "clean"
    curation["gold_mode"] = "clean"


def _append_evidence_exclusion(
    curation: dict[str, Any], source_id: str, reason: str, note: str | None
) -> None:
    """Replace any prior exclusion for the source, then append the new row."""
    exclusions = [
        e for e in curation.get("exclusions") or [] if e.get("source_id") != source_id
    ]
    exclusions.append({"source_id": source_id, "reason": reason, "note": note})
    curation["exclusions"] = exclusions


def _validate_transition(frm: str | None, to: str) -> None:
    """Translate schema transition errors into the curation service exception family."""
    try:
        schema.validate_case_transition(cast("str", frm), to)
    except schema.TransitionError as exc:
        raise CurationError(str(exc)) from exc


def _invalidate_task_spec_approval(curation: dict[str, Any]) -> None:
    """Clear the approved digest and its audit timestamp together whenever case content invalidates them."""
    curation.pop("task_spec_sha256", None)
    curation.pop("task_spec_approved_at", None)


def _demote_ready(curation: dict[str, Any]) -> str | None:
    """Move ready to draft and clear attestation/approval; return the resulting state."""
    state = curation.get("state")
    if state == "ready":
        _validate_transition("ready", "draft")
        curation["state"] = "draft"
        curation["snapshot_attested"] = False
        _invalidate_task_spec_approval(curation)
        return "draft"
    return state


def _reopen_for_mutation(curation: dict[str, Any]) -> dict[str, Any]:
    """Demote ready to draft; leave stale stale but clear its attestation/approval."""
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


def _apply_case_exclusion(
    curation: dict[str, Any], *, reason: str, note: str | None
) -> None:
    """Validate the exclusion contract, demote ready if needed, then transition to excluded."""
    _validate_case_exclusion_contract(reason, note)
    state = _demote_ready(curation)
    _validate_transition(state, "excluded")
    curation["state"] = "excluded"
    curation["snapshot_attested"] = False
    # A ready case's task-spec approval was already invalidated by
    # _demote_ready's ready->draft step; no non-ready state carries one.
    curation["case_exclusion"] = {"reason": reason, "note": note}


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
    _validate_exclusion_contract(reason, note, valid_reasons=schema.EVIDENCE_REASONS, noun="evidence")


def _validate_case_exclusion_contract(reason: str, note: str | None) -> None:
    """Case-level reason/note contract (shared by exclude and apply-gold)."""
    _validate_exclusion_contract(reason, note, valid_reasons=schema.CASE_EXCLUSION_REASONS, noun="case")


class CaseEditor:
    """Own one case's locked mutations and complete transactional validation.

    Each operation reloads the latest case after recovery under the workspace
    lock; constructing the editor does not hold a lock across user interaction.
    """

    def __init__(self, root: Path, case_id: str) -> None:
        self.root = Path(root)
        self.case_id = case_id

    @contextmanager
    def _edit(self, op: str) -> Iterator[dict[str, Any]]:
        with storage.WorkspaceLock(self.root):
            storage.recover_startup(self.root)
            raw = _load_case(self.root, self.case_id)
            yield raw
            self._validate(raw)
            with storage.Transaction(self.root, op_id=f"curate-{self.case_id}", kind=f"curation:{op}") as tx:
                tx.stage(
                    f"cases/{self.case_id}.yaml",
                    yaml.safe_dump(raw, sort_keys=False).encode("utf-8"),
                )
                tx.commit()

    def _validate_location(self, raw: dict[str, Any], finding: dict[str, Any]) -> None:
        """Validate the finding range against the frozen bundle head, independently of the mirror."""
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
        line_count = _head_file_line_count(self.root, raw.get("snapshot") or {}, path)
        if start < 1:
            raise CurationError(f"finding location start_line {start} must be >= 1")
        if end > line_count:
            raise CurationError(
                f"finding location {path!r} end_line {end} exceeds the head file's "
                f"line count {line_count}"
            )


    def _validate(self, raw: dict[str, Any]) -> None:
        """Validate service rules before the full schema; preserve CurationError versus ValidationError."""
        curation = raw.get("curation") or {}
        findings = curation.get("findings") or []

        if len(findings) > MAX_GOLD_FINDINGS:
            raise CurationError(f"case {self.case_id} exceeds 50 gold findings")
        ids = [f.get("finding_id") for f in findings]
        for fid in set(ids):
            if ids.count(fid) > 1:
                raise CurationError(f"case {self.case_id} has duplicate finding {fid}")

        # ready => snapshot_attested and stale => not-attested are enforced by the
        # schema Curation._consistent validator.
        candidates = raw.get("candidates") or []
        for finding in findings:
            self._validate_location(raw, finding)
            provenance = finding.get("provenance") or {}
            if provenance.get("kind") == "historical":
                srcs = provenance.get("source_ids") or []
                if len(srcs) != 1:
                    raise CurationError(
                        f"case {self.case_id} historical finding must reference exactly one source"
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


    def validate(self) -> None:
        """Read and validate one case without writing; propagate service and schema errors unchanged."""
        raw = _load_case(self.root, self.case_id)
        self._validate(raw)
        return None


    def _fragment_provenance(
        self, raw: dict[str, Any], finding: dict[str, Any], source_ids: list[str],
    ) -> tuple[str, list[str]]:
        """Derive historical only for one source whose candidate content matches exactly."""
        _check_evidence_sources(self.root, raw, source_ids, self.case_id)
        if len(source_ids) == 1:
            cand = next(
                (c for c in (raw.get("candidates") or []) if c.get("source_id") == source_ids[0]),
                None,
            )
            if cand is not None and _projection_matches(cand, finding):
                return "historical", source_ids
        return _derive_provenance_kind(source_ids), source_ids


    def accept_candidate(self, source_id: str) -> None:
        """Accept an exact candidate with unchanged content, derived ID, and historical provenance."""

        with self._edit('accept') as raw:
            candidate = next(
                (c for c in (raw.get("candidates") or []) if c.get("source_id") == source_id),
                None,
            )
            if candidate is None:
                raise CurationError(f"no candidate {source_id} in case {self.case_id}")
            if not candidate.get("exact_acceptable"):
                raise CurationError(f"candidate {source_id} is not exact_acceptable")

            curation = raw.setdefault("curation", {})
            _reopen_for_mutation(curation)
            finding = _build_finding(self.case_id, {**candidate, "source_ids": [source_id]}, kind="historical")
            curation.setdefault("findings", []).append(finding)
            _derive_content(raw)


    def replace_findings(
        self, finding_id: str, *, replacements: list[dict[str, Any]]
    ) -> None:
        """Replace one finding with an atomic split/merge batch, validating the resulting whole case."""

        with self._edit('replace') as raw:
            curation = raw.setdefault("curation", {})
            _reopen_for_mutation(curation)
            findings = curation.setdefault("findings", [])
            index = next(
                (i for i, f in enumerate(findings) if f.get("finding_id") == finding_id),
                None,
            )
            if index is None:
                raise CurationError(f"no finding {finding_id}")
            sources = (source for atom in replacements for source in list(atom.get("source_ids") or []))
            _check_evidence_sources(self.root, raw, sources, self.case_id)
            built = [_build_finding(self.case_id, r) for r in replacements]
            new_findings = list(findings[:index]) + built + list(findings[index + 1:])
            curation["findings"] = new_findings
            _derive_content(raw)


    def exclude_evidence_batch(
        self, source_ids: list[str], *, reason: str, note: str | None = None
    ) -> None:
        """Validate reason, note, and all sources before atomically appending the exclusion batch."""

        with self._edit('exclude-evidence') as raw:
            _validate_evidence_exclusion_contract(reason, note)
            _check_evidence_sources(self.root, raw, source_ids, self.case_id)
            curation = raw.setdefault("curation", {})
            _reopen_for_mutation(curation)
            for source_id in source_ids:
                _append_evidence_exclusion(curation, source_id, reason, note)


    def mark_ready(self, *, head_sha: str, task_spec_sha256: str | None = None) -> None:
        """Attest the exact frozen head and approved task-spec digest, then mark ready."""

        with self._edit('mark-ready') as raw:
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
                    f"case {self.case_id} cannot be marked ready with an empty gold findings set "
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


    def attest_clean(self) -> None:
        """Attest an empty gold set as clean; reopen mutable state without marking the snapshot ready."""

        with self._edit('attest-clean') as raw:
            curation = raw.setdefault("curation", {})
            if curation.get("findings"):
                raise CurationError(
                    f"case {self.case_id} has gold findings; clean attestation requires an empty gold set"
                )
            # Reopen as draft; clean attestation cannot replace final snapshot approval.
            _reopen_for_mutation(curation)
            _set_clean(curation)


    def exclude_case(
        self, reason: str, *, note: str | None = None
    ) -> None:
        """Exclude a case through valid transitions; ready passes through draft and loses attestation."""

        with self._edit('exclude-case') as raw:
            curation = raw.setdefault("curation", {})
            _apply_case_exclusion(curation, reason=reason, note=note)


    def reinclude_case(self) -> None:
        """Re-include an excluded case to the state its snapshot supports."""

        with self._edit('reinclude-case') as raw:
            curation = raw.setdefault("curation", {})
            if curation.get("state") != "excluded":
                raise CurationError(f"case {self.case_id} is not excluded")
            snapshot_doc = raw.get("snapshot") or {}
            destination = "draft" if snapshot_doc.get("status") == "ready" else "unreplayable"
            _validate_transition("excluded", destination)
            curation["state"] = destination
            curation["snapshot_attested"] = False
            curation["case_exclusion"] = None


    def apply_gold_fragment(self, fragment: dict[str, Any]) -> None:
        """Derive and validate a reviewed gold fragment, ignoring forged IDs and status."""

        with self._edit('apply-gold') as raw:
            curation = raw.setdefault("curation", {})
            _reopen_for_mutation(curation)

            findings: list[dict[str, Any]] = []
            for frag in fragment.get("findings") or []:
                kind, _ = self._fragment_provenance(
                    raw, frag, list(frag.get("source_ids") or []),
                )
                findings.append(_build_finding(self.case_id, frag, kind=kind))
            curation["findings"] = findings

            for exc in fragment.get("exclusions") or []:
                src = exc["source_id"]
                reason = exc["reason"]
                note = exc.get("note")
                _validate_evidence_exclusion_contract(reason, note)
                _check_evidence_sources(self.root, raw, [src], self.case_id)
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
                        f"case {self.case_id} clean fragment requires an empty gold findings set"
                    )
                _set_clean(curation)
            else:
                _derive_content(raw)


    def add_findings(
        self, *, findings: list[dict[str, Any]], kind: Literal["authored", "edited"] = "authored",
    ) -> None:
        """Admit source references and derive one authored or edited batch atomically."""
        with self._edit("add" if kind == "authored" else "add-edited") as raw:
            if kind not in ("authored", "edited"):
                raise CurationError(f"unknown finding provenance {kind!r}")
            def sources() -> Iterator[str]:
                for i, atom in enumerate(findings):
                    source_ids = list(atom.get("source_ids") or [])
                    if kind == "edited" and not source_ids:
                        raise CurationError(f"edited-finding atom {i} carries no source_ids")
                    yield from source_ids

            _check_evidence_sources(self.root, raw, sources(), self.case_id)
            curation = raw.setdefault("curation", {})
            _reopen_for_mutation(curation)
            curation.setdefault("findings", []).extend(
                _build_finding(self.case_id, atom, kind=kind) for atom in findings
            )
            _derive_content(raw)
