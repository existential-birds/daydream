"""Import normalized evidence and frozen cases from explicit private GitHub PRs.

Preflight re-verifies access and immutable repository identity before mutations.
REST and GraphQL evidence, each requested head's snapshot/bundle, and the ledger
land in one crash-consistent transaction. Git/GitHub calls use git_ops; exhausted
bounded rate-limit retries record a fetch failure rather than dropping evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from daydream import git_ops
from daydream.benchmark import curation as cu, schema, snapshot, storage
from daydream.benchmark.github_anchors import (
    _anchor_fail_closed as _anchor_fail_closed,
    _derive_authoring_anchors,
    _derive_one_anchor as _derive_one_anchor,
    _extract_prioritization_facts as _extract_prioritization_facts,
    _reuse_prior_facts,
)
from daydream.benchmark.github_evidence import (
    _as_author,
    _evidence_from_inline,
    _evidence_from_review,
    _evidence_projection_hash as _evidence_projection_hash,
    _evidence_signature_from_doc as _evidence_signature_from_doc,
    _evidence_signature_from_raw as _evidence_signature_from_raw,
    _join_dismissal,
    _project_one as _project_one,
    _reconcile_inline_evidence,
    _rest_evidence as _rest_evidence,
    project_candidates as project_candidates,
)
from daydream.benchmark.github_preflight import (
    ImportTargetError as ImportTargetError,
    ImportTargets as ImportTargets,
    PreflightError as PreflightError,
    _run_gh_api_user as _run_gh_api_user,
    parse_import_targets as parse_import_targets,
    preflight as preflight,
)
from daydream.benchmark.github_transport import (
    _RATE_LIMIT_ATTEMPTS as _RATE_LIMIT_ATTEMPTS,
    _REVIEW_THREADS_QUERY as _REVIEW_THREADS_QUERY,
    _THREAD_COMMENTS_QUERY as _THREAD_COMMENTS_QUERY,
    _fetch_with_retry as _fetch_with_retry,
    _graphql_review_threads as _graphql_review_threads,
    _ImportRateLimitError as _ImportRateLimitError,
    _rest as _rest,
)


def _payload_sha256(import_doc: dict[str, Any]) -> str:
    """Hash the entire canonical import except its self-referential fetch record, including repository and PR intent."""
    canonical = json.dumps(import_doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _backfill_prior_anchors(doc: schema.ImportDocument, prior_raw: dict[str, Any]) -> None:
    """Restore persisted anchors onto fresh root inline comments before comparison."""
    prior_by_id: dict[int, schema.AuthoringAnchor] = {}
    for e in prior_raw.get("evidence", []):
        anchor_raw = e.get("authoring_anchor")
        if not isinstance(anchor_raw, dict):
            continue
        try:
            db_id = int(e["database_id"])
            if db_id not in prior_by_id:
                prior_by_id[db_id] = schema.AuthoringAnchor.model_validate(anchor_raw)
        except (KeyError, ValidationError) as exc:
            # Report corrupt prior state through the caller's ledger-failure path.
            raise storage.WorkspaceCorrupt(
                "prior import evidence record has an invalid authoring_anchor"
                " or a missing database_id"
            ) from exc
    for record in doc.evidence:
        if record.kind != "inline_comment" or record.reply_to_id is not None:
            continue
        prior = prior_by_id.get(record.database_id)
        if prior is not None and record.authoring_anchor is None:
            record.authoring_anchor = prior


def _task_input_signature_from_doc(doc: schema.ImportDocument) -> str:
    """Hash title, body, and base/head SHA+ref; timestamps, URLs, and merge metadata do not affect task input."""
    sig = _task_input_signature_from_raw(doc.model_dump(mode="json"))
    assert sig is not None  # a typed doc always carries body + head.ref
    return sig


def _task_input_signature_from_raw(raw: dict[str, Any]) -> str | None:
    """Return a comparable task-input hash only when the historical header is complete."""
    pr = raw.get("pull_request") or {}
    head = pr.get("head") or {}
    if "body" not in pr or "ref" not in head:
        return None
    base = pr.get("base") or {}
    payload = {
        "title": str(pr.get("title") or ""),
        "body": str(pr.get("body") or ""),
        "base_sha": base.get("sha"),
        "base_ref": base.get("ref"),
        "head_sha": head.get("sha"),
        "head_ref": head.get("ref"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _referenced_evidence_id(sid: str) -> int:
    """Parse a canonical source ID; corrupt curation references raise rather than bypass the stale gate."""
    if not schema._SOURCE_ID_RE.fullmatch(sid):
        raise storage.WorkspaceCorrupt(
            f"curation references non-canonical source_id {sid!r}"
        )
    return int(sid.rsplit(":", 1)[-1])


def _referenced_evidence_ids(curation: dict[str, Any]) -> set[int]:
    """Collect physical IDs from finding provenance and exclusions, rejecting noncanonical references."""
    ids: set[int] = set()
    for finding in curation.get("findings", []):
        if not isinstance(finding, dict):
            continue
        provenance = finding.get("provenance") or {}
        for sid in provenance.get("source_ids", []):
            ids.add(_referenced_evidence_id(str(sid)))
    for exclusion in curation.get("exclusions", []):
        if not isinstance(exclusion, dict):
            continue
        ids.add(_referenced_evidence_id(str(exclusion.get("source_id") or "")))
    return ids


def _referenced_projection_changed(
    prior: dict[str, Any],
    prior_case_candidates: dict[str, dict[str, Any]],
    fresh_candidates: list[schema.Candidate],
) -> bool:
    """Detect disappeared or relocated referenced candidates after projection."""
    referenced = _referenced_evidence_ids(prior)
    if not referenced:
        return False
    fresh_by_source: dict[str, dict[str, Any]] = {
        c.source_id: c.model_dump(mode="json") for c in fresh_candidates
    }
    for source_id, prior_candidate in prior_case_candidates.items():
        if _referenced_evidence_id(source_id) not in referenced:
            continue
        fresh_candidate = fresh_by_source.get(source_id)
        if fresh_candidate is None:
            return True
        if prior_candidate.get("location") != fresh_candidate.get("location"):
            return True
    return False


def _case_materialize(
    doc: schema.ImportDocument,
    number: int,
    requested_heads: list[str],
    import_file: str,
    *,
    root: Path | None = None,
    repo_slug: str = "",
    origin_url: str | None = None,
    prior: _PriorImport | None = None,
    changed_ids: set[int] | None = None,
    task_input_changed: bool = False,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[tuple[str, bytes]]]:
    """Freeze and project each distinct requested head, preserving prior curation."""
    pull_request = doc.pull_request
    base_sha = pull_request.base.sha
    out: list[tuple[str, str, dict[str, Any]]] = []
    bundle_drops: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for head_token in requested_heads:
        head_sha = head_token if head_token != "final" else None
        if head_token == "final":
            # head-immutable: an existing final_pr_head case resolves to its
            # pinned commit; only a first import (no pin) uses the live head.
            pinned = prior.pinned_head if prior is not None else None
            if pinned is not None:
                head_sha = pinned
            if head_sha is None:
                head_sha = pull_request.head.sha
        if not head_sha or head_sha in seen:
            continue
        seen.add(head_sha)
        case_id = schema.case_id_for(number, head_sha)
        curation: dict[str, Any] = {
            "state": "draft",
            "snapshot_attested": False,
            "clean_attested": False,
            "gold_status": None,
            "findings": [],
            "exclusions": [],
            "case_exclusion": None,
        }
        prior_case = prior.cases.get(case_id) if prior is not None else None
        previous = prior_case.curation if prior_case is not None else None
        if previous is not None:
            curation = dict(previous)
        if root is not None and origin_url is not None and base_sha and head_sha:
            policy = "final_pr_head" if head_token == "final" else "explicit_head"
            if policy == "explicit_head" and pull_request.changed_files is None:
                raise git_ops.GitError(
                    f"PR {number} explicit-head freeze requires a complete changed-files inventory"
                )
            snapshot_doc, bundle_bytes = snapshot.freeze_one(
                root,
                repo_slug,
                number,
                base_tip=base_sha,
                head_sha=head_sha,
                policy=policy,
                requested_head=head_token,
                pr_changed_files=frozenset(pull_request.changed_files or ()),
                origin_url=origin_url,
            )
            if snapshot_doc.get("status") == "ready" and bundle_bytes is not None:
                bundle_drops.append((snapshot_doc["bundle_file"], bundle_bytes))
            elif snapshot_doc.get("status") != "ready" and (previous or {}).get("state") in ("ready", "stale"):
                # An unreachable pinned head fails refresh. Preserve the curated
                # snapshot and indexed bundle rather than installing unreplayable state.
                error = snapshot_doc.get("error") or {}
                raise git_ops.GitError(
                    f"PR {number} freeze of curated case {case_id} is unreplayable "
                    f"({error.get('reason')}): {error.get('detail')}"
                )
            elif snapshot_doc.get("status") == "unreplayable" and curation.get("state") != "excluded":
                curation["state"] = "unreplayable"
                curation["snapshot_attested"] = False
                curation["clean_attested"] = False
                curation["gold_status"] = "findings" if curation.get("findings") else None
                cu._invalidate_task_spec_approval(curation)
            # Restore exact anchors from the populated mirror before projection.
            _derive_authoring_anchors(doc, snapshot.mirror(root), head_sha, changed_ids)
        else:
            # Imported status: no freeze, no mirror — anchors stay unset
            # (projection treats the missing anchor as not-exact, Task 5).
            snapshot_doc = {
                "status": "imported",
                "policy": "final_pr_head" if head_token == "final" else "explicit_head",
                "requested_head": head_token,
                # both base SHAs carry the PR base tip at import — the merge
                # base is not yet computed and diverges on imported -> ready.
                "original_base_sha": base_sha,
                "requested_base_sha": base_sha,
                "original_head_sha": head_sha,
                "error": None,
            }
        # Projection runs after the freeze branch so per-head candidates
        # consume the derived authoring anchors (or their absence, closed).
        candidates = project_candidates(doc, head_sha)
        facts: schema.PrioritizationFacts | None = None
        if root is not None and origin_url is not None and snapshot_doc["status"] == "ready":
            # Reuse only facts matching the freshly projected candidate split.
            facts = _reuse_prior_facts(
                prior_case.facts if prior_case is not None else None,
                head_sha,
                changed_ids,
                {c.source_id for c in candidates},
                {r.source_id for r in doc.evidence},
            )
            if facts is None:
                facts = _extract_prioritization_facts(
                    doc,
                    snapshot.mirror(root),
                    head_sha,
                    {c.source_id for c in candidates},
                )
        if previous is not None:
            # Derived anchors can move a referenced candidate even when raw evidence is unchanged.
            should_stale = task_input_changed or _referenced_projection_changed(
                previous, prior_case.candidates if prior_case is not None else {}, candidates
            ) or (
                changed_ids is not None
                and bool(_referenced_evidence_ids(previous) & changed_ids)
            )
            if should_stale and previous.get("state") in ("ready", "stale"):
                curation["state"] = "stale"
                curation["snapshot_attested"] = False
                cu._invalidate_task_spec_approval(curation)
        case_doc: dict[str, Any] = {
            "schema_version": 2,
            "case_id": case_id,
            "pull_request": pull_request.model_dump(mode="json"),
            "snapshot": snapshot_doc,
            "source": {"import_file": import_file},
            "curation": curation,
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
        if facts is not None:
            case_doc["prioritization"] = facts.model_dump(mode="json")
        out.append((case_id, f"cases/{case_id}.yaml", case_doc))
    return out, bundle_drops


def _retired_snapshot_bundles(
    root: Path,
    manifest: dict[str, Any],
    number: int,
    new_cases: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str]]:
    """Retire prior ready bundles only after a non-ready rewrite and only when no ready case still references them."""
    entry = _manifest_entry(manifest, number)
    if entry is None or entry.get("import_state") != "fetched":
        return []
    new_by_id = {case_id: case_doc for case_id, _, case_doc in new_cases}
    transitioned = {
        case_id
        for case_id, case_doc in new_by_id.items()
        if (case_doc.get("snapshot") or {}).get("status") != "ready"
    }
    if not transitioned:
        return []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest.get("cases", []):
        if isinstance(row, dict) and isinstance(row.get("case_id"), str):
            rows_by_id[row["case_id"]] = row
    old_docs: dict[str, schema.CaseDocument] = {}

    def old_doc(case_id: str) -> schema.CaseDocument:
        if case_id not in old_docs:
            row = rows_by_id.get(case_id)
            if not isinstance(row, dict) or not isinstance(row.get("case_file"), str):
                raise storage.WorkspaceCorrupt(
                    f"{root}: prior case {case_id} has no indexed case file"
                )
            raw = storage.load_yaml_strict(
                storage.resolve_authoring_path(root, row["case_file"])
            )
            prior_snapshot = raw.get("snapshot") if isinstance(raw, dict) else None
            if (
                isinstance(prior_snapshot, dict)
                and prior_snapshot.get("status") == "ready"
                and "base_resolution" not in prior_snapshot
            ):
                raise storage.WorkspaceCorrupt(
                    f"{root}: prior ready case {case_id} is missing snapshot.base_resolution; "
                    "run `daydream benchmark upgrade <workspace>` for this workspace "
                    "before refreshing"
                )
            try:
                old_docs[case_id] = schema.CaseDocument.model_validate(
                    schema._schema_ready(raw)
                )
            except Exception as exc:
                raise storage.WorkspaceCorrupt(
                    f"{root}: prior case {case_id} is not a valid case document"
                ) from exc
        return old_docs[case_id]

    candidates: dict[str, str] = {}
    for case_id in entry.get("case_ids", []):
        if case_id not in transitioned:
            continue
        prior_snapshot = old_doc(case_id).snapshot
        if not isinstance(prior_snapshot, schema.SnapshotReady):
            continue
        previous = candidates.get(prior_snapshot.bundle_file)
        if previous is not None and previous != prior_snapshot.bundle_sha256:
            raise storage.WorkspaceCorrupt(
                f"{root}: shared prior bundle has conflicting recorded digests"
            )
        candidates[prior_snapshot.bundle_file] = prior_snapshot.bundle_sha256

    if not candidates:
        return []
    retained_refs = {
        str((case_doc.get("snapshot") or {}).get("bundle_file"))
        for case_doc in new_by_id.values()
        if (case_doc.get("snapshot") or {}).get("status") == "ready"
        and (case_doc.get("snapshot") or {}).get("bundle_file")
    }
    for case_id in rows_by_id:
        if case_id in new_by_id:
            continue
        snapshot = old_doc(case_id).snapshot
        if isinstance(snapshot, schema.SnapshotReady):
            retained_refs.add(snapshot.bundle_file)
    return sorted(
        (bundle_file, digest)
        for bundle_file, digest in candidates.items()
        if bundle_file not in retained_refs
    )


def _ledger_replace(raw: dict[str, Any], entry: dict[str, Any]) -> None:
    """Replace (or append) one ``pull_requests[]`` entry, keeping stable order."""
    raw["pull_requests"] = [
        e for e in raw.get("pull_requests", []) if e.get("number") != entry["number"]
    ]
    raw["pull_requests"].append(entry)


def _stamp_fetched(
    raw: dict[str, Any],
    number: int,
    import_file: str,
    import_sha256: str,
    requested_heads: list[str],
    case_ids: list[str],
) -> None:
    schema.validate_pr_transition(
        _pending_pr_state(raw, number), "fetched"
    )
    _ledger_replace(
        raw,
        {
            "number": number,
            "import_state": "fetched",
            "import_file": import_file,
            "import_sha256": import_sha256,
            "error": None,
            "latest_error": None,  # a successful import/refresh clears the prior failed attempt
            "requested_heads": requested_heads,
            "case_ids": case_ids,
        },
    )
    for case_id in case_ids:
        # Replace any prior index row for this case_id so a re-import of the same
        # PR (incl. the fetched->fetched --refresh path) never leaves duplicate
        # cases[] rows. Mirrors _ledger_replace's replace-by-key semantics.
        raw["cases"] = [
            c for c in raw.get("cases", []) if c.get("case_id") != case_id
        ]
        raw["cases"].append(
            {"case_id": case_id, "pr_number": number, "case_file": f"cases/{case_id}.yaml"}
        )
    raw["cases"] = _sorted_cases(raw["cases"])


def _stages_failed(raw: dict[str, Any], number: int, code: str, message: str) -> None:
    prior_state = _pending_pr_state(raw, number)
    if prior_state == "fetched":
        # Keep last-good linkage so a failed refresh cannot orphan curated cases.
        entry = _manifest_entry(raw, number) or {}
        _ledger_replace(
            raw,
            {
                "number": number,
                "import_state": "fetched",
                "import_file": entry.get("import_file"),
                "import_sha256": entry.get("import_sha256"),
                "error": None,
                "latest_error": {"code": code, "message": message},
                "requested_heads": entry.get("requested_heads", []),
                "case_ids": entry.get("case_ids", []),
            },
        )
        return
    schema.validate_pr_transition(prior_state, "fetch_failed")
    _ledger_replace(
        raw,
        {
            "number": number,
            "import_state": "fetch_failed",
            "import_file": None,
            "import_sha256": None,
            "error": {"code": code, "message": message},
            "latest_error": None,
            "requested_heads": [],
            "case_ids": [],
        },
    )


def _pending_pr_state(raw: dict[str, Any], number: int) -> str:
    for entry in raw.get("pull_requests", []):
        if entry.get("number") == number:
            return str(entry.get("import_state", "pending"))
    return "pending"


def _manifest_entry(raw: dict[str, Any], number: int) -> dict[str, Any] | None:
    """The ledger entry for *number*, or None when not yet imported."""
    for entry in raw.get("pull_requests", []):
        if isinstance(entry, dict) and entry.get("number") == number:
            return entry
    return None


def _sorted_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(c: dict[str, Any]) -> tuple[int, str, str]:
        return (int(c["pr_number"]), schema.head_sha_from_case_id(c["case_id"]), c["case_id"])

    return sorted(cases, key=_key)


def _manifest_bytes(raw: dict[str, Any]) -> bytes:
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def _stage_fetch_failure(
    root: Path, raw: dict[str, Any], number: int, code: str, message: str
) -> None:
    """Persist a fetch failure in the ledger without staging imports or cases."""
    _stages_failed(raw, number, code, message)
    with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
        tx.stage("benchmark.yaml", _manifest_bytes(raw))
        tx.commit()


@dataclass(frozen=True)
class _PriorCase:
    curation: dict[str, Any] | None
    candidates: dict[str, dict[str, Any]]
    snapshot: dict[str, Any]
    facts: dict[str, Any] | None


@dataclass
class _PriorImport:
    """One consistent read of prior evidence and its frozen case documents."""

    import_file: str
    requested_heads: list[str]
    document: dict[str, Any] | None = None
    evidence_signature: frozenset[tuple[int, str]] | None = None
    task_signature: str | None = None
    cases: dict[str, _PriorCase] = field(default_factory=dict)

    @property
    def pinned_head(self) -> str | None:
        for case in self.cases.values():
            if case.snapshot.get("policy") == "final_pr_head" and case.snapshot.get("original_head_sha"):
                return cast(str, case.snapshot["original_head_sha"])
        return None


def _prior_import_state(root: Path, raw: dict[str, Any], number: int) -> _PriorImport:
    """Load prior state before fetching; present-but-corrupt documents fail closed."""
    existing = _manifest_entry(raw, number)
    prior = _PriorImport(
        f"imports/pr-{number:06d}.json", list(existing.get("requested_heads", [])) if existing else [],
    )
    if existing is None or existing.get("import_state") != "fetched":
        return prior
    prior_import_file = existing.get("import_file")
    if not prior_import_file:
        raise storage.WorkspaceCorrupt(f"{root}: fetched ledger entry for PR {number} is missing import_file")
    prior.document = storage.load_json_strict(storage.resolve_authoring_path(root, prior_import_file))
    prior.evidence_signature = _evidence_signature_from_raw(prior.document)
    prior.task_signature = _task_input_signature_from_raw(prior.document)
    for case_id in existing.get("case_ids", []):
        case = storage.load_yaml_strict(storage.resolve_authoring_path(root, f"cases/{case_id}.yaml"))
        curation = case.get("curation")
        curation = curation if isinstance(curation, dict) else None
        candidates = case.get("candidates")
        snapshot = case.get("snapshot") or {}
        if (curation is not None and curation.get("state") in ("ready", "stale")
                and not snapshot.get("original_head_sha")):
            raise storage.WorkspaceCorrupt(f"{root}: ready/stale case {case_id} is missing snapshot.original_head_sha")
        facts = case.get("prioritization")
        prior.cases[case_id] = _PriorCase(
            curation=curation,
            candidates={c["source_id"]: c for c in candidates if isinstance(c, dict) and c.get("source_id")}
            if isinstance(candidates, list) else {},
            snapshot=snapshot,
            facts=facts if isinstance(facts, dict) else None,
        )
    return prior


def _import_one_pr(
    root: Path,
    raw: dict[str, Any],
    verified: schema.PreflightLedger,
    number: int,
    requested_heads: list[str],
    *,
    refresh: bool,
    origin_url: str | None = None,
) -> int:
    """Fetch, materialize, and commit one PR atomically; stage failures without losing prior linkage."""
    prior = _prior_import_state(root, raw, number)
    import_file = prior.import_file
    try:
        # Refresh/re-import never orphans a previously pinned case. The same
        # union also decides whether a complete PR-file inventory is required:
        # a newly final-only refresh must still protect a retained explicit head.
        materialize_heads = requested_heads
        if prior.requested_heads:
            materialize_heads = list(dict.fromkeys([*prior.requested_heads, *requested_heads]))
        include_changed_files = any(head != "final" for head in materialize_heads)
        doc = fetch_and_normalize(
            root,
            verified,
            number,
            include_changed_files=include_changed_files,
        )
        # Existing final-head cases retain their frozen head even if the live PR advances.
        pinned = prior.pinned_head if prior is not None else None
        if pinned is not None:
            doc.pull_request.head.sha = pinned
        # Persisted anchors precede signature comparison; missing anchors are derived later.
        if prior.document is not None:
            _backfill_prior_anchors(doc, prior.document)
        # Task-input staleness is refresh-only; referenced evidence changes apply to every import.
        task_input_changed = (
            refresh
            and prior.task_signature is not None
            and prior.task_signature != _task_input_signature_from_doc(doc)
        )
        changed_ids: set[int] | None = None
        if prior.evidence_signature is not None:
            # A new (id, projection) pair changes that id; absent ids are deletions.
            # Prior duplicate projections are harmless when they cover every new pair.
            # Omitted legacy range fields are schema upgrades, not evidence edits.
            assert prior.document is not None
            legacy_without_start_line: set[int] = {
                int(e["database_id"])
                for e in prior.document.get("evidence", [])
                if "original_start_line" not in e
            }
            fresh = _evidence_signature_from_doc(doc, downgrade_start_line=legacy_without_start_line)
            prior_ids = {db_id for db_id, _ in prior.evidence_signature}
            fresh_ids = {db_id for db_id, _ in fresh}
            changed_ids = (prior_ids - fresh_ids) | {
                db_id for db_id, _ in fresh - prior.evidence_signature
            }
        # Refresh/re-import never orphans a previously pinned case: materialize
        # the union of the prior ledger heads and the newly-requested heads so
        # _stamp_fetched's cases[] rewrite keeps every curated case indexed.
        cases, bundle_rels = _case_materialize(
            doc, number, materialize_heads, import_file,
            root=root, repo_slug=verified.repository, origin_url=origin_url,
            prior=prior,
            changed_ids=changed_ids, task_input_changed=task_input_changed,
        )
        retired_bundles = _retired_snapshot_bundles(root, raw, number, cases)
        # Re-serialize the mutated doc (authoring anchors derived on the typed
        # evidence records during materialization), recompute the fetch payload
        # digest over the same blocks, and keep every digest in lockstep.
        final_doc = doc.model_dump(mode="json")
        final_doc["fetch"]["payload_sha256"] = _payload_sha256(
            {k: final_doc[k] for k in ("schema_version", "repository", "pull_request", "evidence")}
        )
        import_bytes = json.dumps(final_doc, indent=2).encode("utf-8")
        import_sha256 = hashlib.sha256(import_bytes).hexdigest()
        for _, _, case_doc in cases:
            case_doc["source"]["import_sha256"] = import_sha256
        with storage.Transaction(root, op_id=f"import-{number}", kind="import") as tx:
            for rel, digest in retired_bundles:
                tx.retire(rel, expected_sha256=digest)
            tx.stage(import_file, import_bytes)
            for rel, content in bundle_rels:
                tx.stage(rel, content)
            for _, case_path, case_doc in cases:
                tx.stage(case_path, yaml.safe_dump(case_doc, sort_keys=False).encode("utf-8"))
            _stamp_fetched(
                raw,
                number,
                import_file,
                import_sha256,
                materialize_heads,
                [c[0] for c in cases],
            )
            tx.stage("benchmark.yaml", _manifest_bytes(raw))
            tx.commit()
        return 0
    except _ImportRateLimitError as exc:
        _stage_fetch_failure(root, raw, number, "rate_limit", str(exc))
        return 1
    except (git_ops.GitError, schema.TransitionError, storage.WorkspaceError, PreflightError) as exc:
        _stage_fetch_failure(root, raw, number, "fetch", str(exc))
        return 1


_UNSET_ORIGIN = object()


def run_import_prs(
    root: Path,
    pr_numbers: list[int],
    heads: list[str] | None = None,
    pr_heads: dict[int, list[str]] | None = None,
    refresh: bool = False,
    origin_url: str | None | object = _UNSET_ORIGIN,
) -> int:
    """Recover, preflight, then atomically import each PR; any failed PR makes exit nonzero."""
    root = Path(root)
    flat_heads: list[str] = []
    seen_heads: set[str] = set()
    for head in ["final", *(heads or [])]:
        if head not in seen_heads:
            seen_heads.add(head)
            flat_heads.append(head)
    requested_by_pr: dict[int, list[str]] = {}
    for number in pr_numbers:
        if pr_heads is not None and pr_heads.get(number):
            requested_by_pr[number] = pr_heads[number]
        else:
            requested_by_pr[number] = list(flat_heads)
    exit_code = 0
    with storage.WorkspaceLock(root):
        storage.recover_startup(root)
        verified = preflight(root, len(pr_numbers))
        raw = storage.load_yaml_strict(root / "benchmark.yaml")
        repo = verified.repository
        if origin_url is _UNSET_ORIGIN:
            origin_url = f"https://github.com/{repo}.git" if repo else None
        effective_origin: str | None = (
            origin_url if isinstance(origin_url, str) or origin_url is None else None
        )
        for number in pr_numbers:
            if _import_one_pr(
                root, raw, verified, number, requested_by_pr[number], refresh=refresh, origin_url=effective_origin
            ):
                exit_code = 1
    return exit_code


def fetch_and_normalize(
    root: Path,
    verified: schema.PreflightLedger,
    number: int,
    *,
    include_changed_files: bool = False,
) -> schema.ImportDocument:
    """Fetch the full PR header and all REST/GraphQL review evidence."""
    owner_repo = verified.repository
    header = _fetch_with_retry(root, owner_repo, number)
    changed_files = None
    if include_changed_files:
        changed_files = _normalize_changed_files(
            header,
            _rest(root, f"repos/{owner_repo}/pulls/{number}/files"),
        )

    review_records = _rest(root, f"repos/{owner_repo}/pulls/{number}/reviews")
    inline_records = [_evidence_from_inline(raw) for raw in _rest(root, f"repos/{owner_repo}/pulls/{number}/comments")]
    threads = _graphql_review_threads(root, owner_repo, number)

    evidence: list[dict[str, Any]] = [_evidence_from_review(raw) for raw in review_records]
    evidence.extend(_join_dismissal(_reconcile_inline_evidence(inline_records, threads), review_records))
    for raw in _rest(root, f"repos/{owner_repo}/issues/{number}/comments"):
        evidence.append(_rest_evidence(raw, "issue_comment", "IC"))

    records = [schema.EvidenceRecord.model_validate(e) for e in evidence]
    # Canonical order: sort by (database_id, created_at) so persisted order and
    # payload_sha256 are independent of REST/GraphQL page boundaries.
    records.sort(key=lambda r: (r.database_id, r.created_at))
    record_dicts = [r.model_dump(mode="json") for r in records]
    base = header.get("base") or {}
    head = header.get("head") or {}
    title = header.get("title") or ""
    body = header.get("body") or ""          # null/empty -> "", Unicode/newlines preserved byte-for-byte
    pull_request = {
        "number": header["number"],          # KeyError propagates if absent — fail closed, never 0
        "url": header.get("url") or "",
        "html_url": header.get("html_url") or "",
        "title": title,
        "body": body,
        "state": header.get("state") or "",
        "title_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "base": {"sha": base.get("sha"), "ref": base.get("ref")},
        "head": {"sha": head.get("sha"), "ref": head.get("ref")},
        "created_at": header.get("created_at"),
        "updated_at": header.get("updated_at"),
        "merged_at": header.get("merged_at"),
        "closed_at": header.get("closed_at"),
        "author": _as_author(header),
        "changed_files": changed_files,
    }
    import_doc = {
        "schema_version": 1,
        "repository": {
            "id": verified.repository_id or "",
            "name_with_owner": owner_repo,
            "visibility": verified.visibility,
        },
        "pull_request": pull_request,
        "evidence": record_dicts,
    }
    return schema.ImportDocument.model_validate(
        {
            **import_doc,
            "fetch": {
                "fetched_at": schema.rfc3339_now(),
                "etag": None,
                "payload_sha256": _payload_sha256(import_doc),
            },
        }
    )


def _normalize_changed_files(header: dict[str, Any], rows: list[Any]) -> list[str]:
    """Return a complete canonical PR path inventory or fail closed."""
    valid_statuses = {
        "added",
        "removed",
        "modified",
        "renamed",
        "copied",
        "changed",
        "unchanged",
    }
    expected = header.get("changed_files")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        raise git_ops.GitError("PR changed_files count is missing or malformed")
    if expected > 3000:
        raise git_ops.GitError(
            f"PR changed_files count {expected} exceeds the 3000-file API inventory limit"
        )
    if len(rows) != expected:
        raise git_ops.GitError(
            f"PR changed_files inventory count mismatch: header={expected}, rows={len(rows)}"
        )

    current_names: set[str] = set()
    all_names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise git_ops.GitError(f"PR changed_files row {index} is not an object")
        try:
            current = schema.exact_git_tree_path(row.get("filename"))
        except ValueError as exc:
            raise git_ops.GitError(f"PR changed_files row {index} has invalid filename: {exc}") from exc
        if current in current_names:
            raise git_ops.GitError(f"PR changed_files inventory repeats filename {current!r}")
        current_names.add(current)
        all_names.add(current)

        status = row.get("status")
        if not isinstance(status, str) or status not in valid_statuses:
            raise git_ops.GitError(
                f"PR changed_files row {index} has missing or unsupported status"
            )
        previous = row.get("previous_filename")
        if status in ("renamed", "copied") and previous is None:
            raise git_ops.GitError(
                f"PR changed_files {status} row {index} is missing previous_filename"
            )
        if status not in ("renamed", "copied") and previous is not None:
            raise git_ops.GitError(
                f"PR changed_files row {index} has unexpected previous_filename"
            )
        if previous is not None:
            try:
                all_names.add(schema.exact_git_tree_path(previous))
            except ValueError as exc:
                raise git_ops.GitError(
                    f"PR changed_files row {index} has invalid previous_filename: {exc}"
                ) from exc
    return sorted(all_names)
