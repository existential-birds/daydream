"""Canonical record stores for adjudication tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.record_identity import record_finding_id
from daydream.training.reply_classifier import classify_reply
from tests.harness.dataset import observation
from tests.harness.record_projection import projection_run


def reply_evidence(reply_id: str, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """A semantic source-hash projection and its separately retained producer text."""
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    semantic = {
        "reply_id": reply_id,
        "author": "alice",
        "author_association": "MEMBER",
        "created_at": "2026-10-04T11:00:00Z",
        "body_sha256": source_hash,
        "reason": "assoc:MEMBER",
        "classifier_label": classify_reply({"body": text, "user": {"login": "alice", "type": "User"},
                                             "author_association": "MEMBER"}),
    }
    capture = {
        "status": "available",
        "source_reply_id": reply_id,
        "body_sha256": source_hash,
        "text": text,
        "captured_sha256": source_hash,
    }
    return semantic, capture


def record_store(
    root: Path, dispositions: tuple[str, ...] = ("unanswered", "ambiguous"), *, retain_replies: bool = True
) -> LocalRecordStore:
    store = LocalRecordStore(root)
    run = projection_run("run-1", dispositions=dispositions, profile="pr_review")
    store.commit_run(run)
    for item, disposition in zip(run["findings"]["value"]["items"], dispositions):
        semantic, capture = reply_evidence(item["item_uid"], "context")
        evidence = [semantic]
        store.append_observation(
            observation(
                item["item_uid"],
                role="automatic",
                evidence_digest=reply_evidence_digest(evidence),
                evidence_digest_scheme="reply-evidence-v1",
                semantic_evidence=evidence,
                item_uid=item["item_uid"],
                payload={"type": "finding-judgment", "disposition": disposition, "rationale": "observed replies"},
                **({"reply_captures": [capture]} if retain_replies else {}),
            )
        )
    return store


def append_replies(
    store: LocalRecordStore,
    replies: tuple[tuple[str, str], ...] = (("9", "edited reply"),),
    *,
    observation_id: str = "edited",
    observed_at: str = "2026-10-06T12:00:00Z",
    retain_replies: bool = True,
) -> None:
    """Append an immutable automatic generation using the real source-hash projection."""
    original = store.read_records()["observations"][0]
    evidence, captures = zip(*(reply_evidence(reply_id, text) for reply_id, text in replies))
    value = {
        **original,
        "observation_id": observation_id,
        "observed_at": observed_at,
        "semantic_evidence": list(evidence),
        "evidence_digest": reply_evidence_digest(list(evidence)),
    }
    value.pop("reply_captures", None)
    if retain_replies:
        value["reply_captures"] = list(captures)
    store.append_observation(value)


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
    automatic = max(
        (o for o in store.read_records()["observations"]
         if o.get("item_uid") == item["item_uid"] and o["role"] == "automatic"),
        key=lambda o: (o["observed_at"], o["observation_id"]),
    )
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
