"""Canonical record stores for adjudication tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.record_identity import record_finding_id
from tests.harness.dataset import observation
from tests.harness.record_projection import projection_run


def record_store(root: Path, dispositions: tuple[str, ...] = ("unanswered", "ambiguous")) -> LocalRecordStore:
    store = LocalRecordStore(root)
    run = projection_run("run-1", dispositions=dispositions, profile="pr_review")
    store.commit_run(run)
    for item, disposition in zip(run["findings"]["value"]["items"], dispositions):
        evidence = [{"reply_id": item["item_uid"], "body": "context", "created_at": "2026-10-04T11:00:00Z"}]
        store.append_observation(
            observation(
                item["item_uid"],
                role="automatic",
                evidence_digest=reply_evidence_digest(evidence),
                evidence_digest_scheme="reply-evidence-v1",
                semantic_evidence=evidence,
                item_uid=item["item_uid"],
                payload={"type": "finding-judgment", "disposition": disposition, "rationale": "observed replies"},
            )
        )
    return store


def snapshot_id(
    store: LocalRecordStore, *, observed_before: str = "2100-01-01T00:00:00Z", valid_before: str | None = None
) -> str:
    return str(store.select_snapshot(observed_before=observed_before, valid_before=valid_before)["snapshot_id"])


def judgment(
    store: LocalRecordStore,
    disposition: str = "accepted",
    *,
    author: str = "alice",
    role: str = "rater",
    item_index: int = 0,
    observed_at: str = "2026-10-05T10:00:00Z",
    valid_at: str = "2026-10-04T11:00:00Z",
    **overrides: Any,
) -> dict[str, Any]:
    run = store.read_records()["runs"][0]
    item = run["findings"]["value"]["items"][item_index]
    automatic = next(o for o in store.read_records()["observations"] if o.get("item_uid") == item["item_uid"])
    return observation(
        f"{author}:{disposition}:{item['item_uid']}:{observed_at}",
        role=role,
        author=author,
        observed_at=observed_at,
        valid_at=valid_at,
        item_uid=item["item_uid"],
        semantic_evidence=automatic["semantic_evidence"],
        evidence_digest=automatic["evidence_digest"],
        evidence_digest_scheme="reply-evidence-v1",
        payload={
            "type": "finding-judgment",
            "disposition": disposition,
            "rationale": "reproduced",
            "record_id": record_finding_id(run["run_id"], run["run_id"], run["run_id"], item["item_uid"]),
        },
        **overrides,
    )
