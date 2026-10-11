"""Canonical record harvest preserves external evidence, rewards and temporal priors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import pytest

from daydream import git_ops
from daydream.dataset import LocalRecordStore
from daydream.dataset_capture import capture_scoring
from daydream.json_utils import canonical_json
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.training import labeler_versions
from daydream.training.harvest import HarvestConfig, build_annotation, make_harvest_services
from daydream.training.harvest_types import HarvestEvidence, HarvestRow
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.reward import ScoringInputs
from daydream.training.rubric import Rubric
from tests.harness.adjudication import snapshot_id
from tests.harness.dataset import observation
from tests.harness.record_projection import projection_run
from tests.harness.scripts import cli_main


def store_run(
    root: Path,
    run_id: str = "run-1",
    *,
    number: int | None = 1,
    dispositions: tuple[str, ...] = ("unanswered",),
    repo: str = "owner/repo",
    source_path: Path | None = None,
    remote_url: str | None = None,
    head_sha: str | None = None,
    recommended_patch: str | None = None,
) -> LocalRecordStore:
    store = LocalRecordStore(root)
    run = projection_run(run_id, dispositions=dispositions, repo_slug=repo)
    run["original_task"]["value"]["repository"]["remote_url"] = remote_url
    if head_sha is not None:
        run["original_task"]["value"]["analyzed_revision"]["head_sha"] = head_sha
    if recommended_patch is not None:
        run["recommended_patch"] = {
            "status": "available",
            "value": {
                "patch": recommended_patch,
                "sha256": hashlib.sha256(recommended_patch.encode()).hexdigest(),
                "capture": None,
            },
        }
    run["original_task"]["value"]["pr"] = {"repo": repo if number else None, "number": number}
    run["scoring"]["value"] = capture_scoring(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}], format_valid=True, length=6), "review"
    )
    run["provenance"]["repository_context"] = {
        "branch": "feature",
        "base_branch": "main",
        "source_path": str(source_path) if source_path is not None else None,
    }
    store.commit_run(run)
    return store


def github(
    store: LocalRecordStore,
    replies: tuple[str | None, ...] = (None,),
    *,
    state: str = "merged",
    merged_at: str = "2026-10-04T13:00:00Z",
    run_id: str | None = None,
) -> Any:
    runs = store.read_records()["runs"]
    run = next(run for run in runs if run["run_id"] == run_id) if run_id is not None else runs[0]
    items = run["findings"]["value"]["items"]
    comments = []
    for index, (item, reply) in enumerate(zip(items, replies), start=1):
        comments.append(
            {
                "id": index,
                "user": {"login": "daydream-runner"},
                "body": f"Finding\n{finding_marker(item['fingerprint'])}\n{DAYDREAM_FOOTER}",
            }
        )
        if reply is not None:
            comments.append(
                {
                    "id": 100 + index,
                    "in_reply_to_id": index,
                    "user": {"login": "alice"},
                    "author_association": "OWNER",
                    "body": reply,
                    "created_at": "2026-10-04T11:00:00Z",
                    "updated_at": "2026-10-05T12:00:00Z",
                }
            )

    def response(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if "/license" in endpoint:
            raise AssertionError(f"harvest requested removed license endpoint: {endpoint}")
        if endpoint.endswith("/comments"):
            return comments
        if endpoint.endswith("/reviews"):
            return [{"user": {"login": "alice"}}]
        if "/commits/" in endpoint:
            return []
        return {"merged": state == "merged", "state": state if state != "merged" else "closed", "merged_at": merged_at}

    return response


@pytest.mark.parametrize(
    ("reply", "expected"), [("applied", "accepted"), ("not applicable", "rejected"), (None, "unanswered")]
)
def test_cli_harvest_captures_record_judgments_and_rewards_without_license_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str | None, expected: str
) -> None:
    store = store_run(tmp_path / "records")
    before = store.read_records()["runs"]
    monkeypatch.setattr(git_ops, "gh_api", github(store, (reply,)))
    assert harvest_cli(store, tmp_path / "dry-run-cache", "--dry-run") == 0
    assert not store.read_records()["observations"]
    assert not (tmp_path / "dry-run-cache").exists()
    assert (
        cli_main(
            [
                "corpus",
                "harvest",
                "--store",
                str(store.root),
                "--snapshot-id",
                snapshot_id(store),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--gh-spacing-sec",
                "0",
            ]
        )
        == 0
    )
    history = store.read_records()["observations"]
    annotation = next(o for o in history if o["payload"]["type"] == "harvest-annotation")
    judgment = next(o for o in history if o["payload"]["type"] == "finding-judgment")
    assert judgment["payload"]["disposition"] == expected
    assert judgment["item_uid"] == "item:0"
    captures = judgment["reply_captures"]
    assert len(captures) == (1 if reply is not None else 0)
    if reply is not None:
        assert captures[0]["text"] == reply
        assert captures[0]["source_reply_id"] == "101"
        assert captures[0]["body_sha256"] == hashlib.sha256(reply.encode()).hexdigest()
        assert captures[0]["captured_sha256"] == hashlib.sha256(reply.encode()).hexdigest()
        assert captures[0]["status"] == "available"
        assert captures[0]["in_reply_to_id"] == "1"
    assert annotation["payload"]["annotation"]["composite_reward"] is not None
    assert annotation["valid_at"] == ("2026-10-04T11:00:00Z" if reply else "2026-10-04T13:00:00Z")
    assert all(o["payload"].get("kind") != "license" for o in history)
    assert store.read_records()["runs"] == before
    assert not (store.root / "index.db").exists()


def test_cli_harvest_retains_every_scoped_reply_and_explicit_content_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_run(tmp_path / "records", dispositions=("unanswered", "unanswered", "unanswered"))
    before = store.read_records()["runs"]
    response = github(store, (None, None, None))
    comments = response("owner/repo", "repos/owner/repo/pulls/1/comments")
    comments.extend([
        {"id": 101, "in_reply_to_id": 1, "user": {"login": "alice"}, "author_association": "OWNER",
         "body": "Applied\nπ 👩🏽‍💻\n", "created_at": "2026-10-04T11:00:00Z",
         "updated_at": "2026-10-05T12:00:00Z", "html_url": "https://github.com/owner/repo/pull/1#discussion_r101"},
        {"id": 102, "in_reply_to_id": 1, "user": {"login": "alice"}, "author_association": "OWNER",
         "body": "False positive"},
        {"id": 103, "in_reply_to_id": 1, "user": {"login": "visitor[bot]"}, "author_association": "NONE",
         "body": "Applied by visitor\n"},
        {"id": 201, "in_reply_to_id": 2, "user": {"login": "alice"}, "author_association": "OWNER", "body": ""},
        {"id": 202, "in_reply_to_id": 2, "user": {"login": "alice"}, "author_association": "OWNER", "body": None},
        {"id": 203, "in_reply_to_id": 2, "user": {"login": "alice"}, "author_association": "OWNER", "body": {"bad": 1}},
        {"id": 204, "in_reply_to_id": 2, "user": {"login": "alice"}, "author_association": "OWNER"},
        {"id": 301, "in_reply_to_id": 3, "user": {"login": "visitor[bot]"},
         "author_association": "NONE", "body": "Applied"},
        {"id": 4, "user": {"login": "daydream-runner"}, "body": f"{finding_marker('f' * 64)}\n{DAYDREAM_FOOTER}"},
        {"id": 401, "in_reply_to_id": 4, "user": {"login": "alice"},
         "author_association": "OWNER", "body": "Outside run"},
    ])
    monkeypatch.setattr(git_ops, "gh_api", response)
    assert harvest_cli(store, tmp_path / "cache") == 0
    judgments = {o["item_uid"]: o for o in store.read_records()["observations"]
                 if o["payload"]["type"] == "finding-judgment"}
    assert [judgments[f"item:{i}"]["payload"]["disposition"] for i in range(3)] == [
        "ambiguous", "ambiguous", "unanswered",
    ]
    captures = [capture for judgment in judgments.values() for capture in judgment["reply_captures"]]
    assert {capture["source_reply_id"] for capture in captures} == {
        "101", "102", "103", "201", "202", "203", "204", "301",
    }
    by_id = {capture["source_reply_id"]: capture for capture in captures}
    assert by_id["101"]["text"] == "Applied\nπ 👩🏽‍💻\n"
    assert by_id["101"]["created_at"] == "2026-10-04T11:00:00Z"
    assert judgments["item:0"]["valid_at"] == "2026-10-04T13:00:00Z"
    assert by_id["101"]["updated_at"] == "2026-10-05T12:00:00Z"
    assert by_id["101"]["html_url"] == "https://github.com/owner/repo/pull/1#discussion_r101"
    assert by_id["103"]["text"] == "Applied by visitor\n"
    assert by_id["201"]["status"] == "available" and by_id["201"]["text"] == ""
    assert by_id["201"]["captured_sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    for reply_id, status, reason in [("202", "unproduced", "reply_body_missing"),
                                      ("203", "unavailable", "reply_body_malformed"),
                                      ("204", "unproduced", "reply_body_missing")]:
        assert by_id[reply_id]["status"] == status and by_id[reply_id]["reason"] == reason
        assert by_id[reply_id]["text"] is None and by_id[reply_id]["captured_sha256"] is None
    for judgment in judgments.values():
        for capture in judgment["reply_captures"]:
            source = next(reply for reply in judgment["semantic_evidence"]
                          if str(reply["reply_id"]) == capture["source_reply_id"])
            assert source["body_sha256"] == capture["body_sha256"]
            assert "text" not in source and "updated_at" not in source
    annotations = [o for o in store.read_records()["observations"] if o["payload"]["type"] == "harvest-annotation"]
    assert "reply_captures" not in annotations[0]["semantic_evidence"]["rubric_json"]
    frozen = store.read_records()
    comments.reverse()
    assert harvest_cli(store, tmp_path / "fresh-cache") == 0 and store.read_records() == frozen
    assert store.read_records()["runs"] == before


def test_cli_capture_only_enrichment_preserves_semantic_history_and_human_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.training.record_evidence import sessions_from_snapshot

    source = store_run(tmp_path / "source")
    monkeypatch.setattr(git_ops, "gh_api", github(source, ("applied",)))
    assert harvest_cli(source, tmp_path / "source-cache") == 0
    history = source.read_records()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(history["runs"][0])
    for original in history["observations"]:
        prior = dict(original)
        if prior["payload"]["type"] == "finding-judgment":
            prior.pop("reply_captures", None)
            prior["observation_id"] = "hash-only"
        store.append_observation(prior)
    old_snapshot = snapshot_id(store)
    original_judgment = next(o for o in history["observations"] if o["payload"]["type"] == "finding-judgment")
    store.append_observation({
        **original_judgment, "observation_id": "human", "role": "rater", "author": "alice",
        "payload": {"type": "finding-judgment", "disposition": "rejected", "rationale": "human confirmed"},
    })
    responder = github(store, ("applied",))
    monkeypatch.setattr(git_ops, "gh_api", responder)
    assert harvest_cli(store, tmp_path / "enrich-cache") == 0
    observations = store.read_records()["observations"]
    automatic = [o for o in observations if o["payload"]["type"] == "finding-judgment" and o["role"] == "automatic"]
    assert len(automatic) == 2 and {o["evidence_digest"] for o in automatic} == {original_judgment["evidence_digest"]}
    assert sum(o["payload"]["type"] == "harvest-annotation" for o in observations) == 1
    enriched = next(o for o in automatic if o["observation_id"] != "hash-only")
    assert enriched["reply_captures"][0]["text"] == "applied"
    effective = sessions_from_snapshot(store.read_snapshot(snapshot_id(store)))[0]["resolutions"][0]
    assert effective["disposition"] == "rejected"
    frozen = store.read_records()
    assert harvest_cli(store, tmp_path / "unchanged-cache") == 0 and store.read_records() == frozen
    # Edited provenance changes retained representation without changing the source
    # body, semantic generation, or creation-time pin.
    comments = responder("owner/repo", "repos/owner/repo/pulls/1/comments")
    comments[1]["updated_at"] = "2026-10-06T12:00:00Z"
    assert harvest_cli(store, tmp_path / "metadata-cache") == 0
    automatic = [o for o in store.read_records()["observations"]
                 if o["payload"]["type"] == "finding-judgment" and o["role"] == "automatic"]
    assert len(automatic) == 3 and {o["evidence_digest"] for o in automatic} == {original_judgment["evidence_digest"]}
    assert all(o["valid_at"] == "2026-10-04T11:00:00Z" for o in automatic)
    assert sum(o["payload"]["type"] == "harvest-annotation" for o in store.read_records()["observations"]) == 1
    earlier = store.read_snapshot(old_snapshot)
    earlier_judgments = [o for o in earlier.observations if o["payload"]["type"] == "finding-judgment"]
    assert [o["observation_id"] for o in earlier_judgments] == ["hash-only"]
    assert all("reply_captures" not in o for o in earlier.observations if o["payload"]["type"] == "finding-judgment")
    # A real source-body edit creates a new evidence generation even when its
    # classifier vote remains accepted, so the old human judgment stops applying.
    metadata_snapshot = snapshot_id(store)
    comments[1]["body"] = "applied\nAdditional context."
    assert harvest_cli(store, tmp_path / "source-edit-cache") == 0
    automatic = [o for o in store.read_records()["observations"]
                 if o["payload"]["type"] == "finding-judgment" and o["role"] == "automatic"]
    edited = next(o for o in automatic if o["evidence_digest"] != original_judgment["evidence_digest"])
    assert edited["payload"]["disposition"] == "accepted"
    original_label = original_judgment["semantic_evidence"][0]["classifier_label"]
    assert edited["semantic_evidence"][0]["classifier_label"] == original_label
    assert edited["reply_captures"][0]["body_sha256"] != original_judgment["reply_captures"][0]["body_sha256"]
    assert edited["reply_captures"][0]["text"] == "applied\nAdditional context."
    effective = sessions_from_snapshot(store.read_snapshot(snapshot_id(store)))[0]["resolutions"][0]
    previous = sessions_from_snapshot(store.read_snapshot(metadata_snapshot))[0]["resolutions"][0]
    assert effective["disposition"] == "accepted" and previous["disposition"] == "rejected"


@pytest.mark.parametrize("reply_id", [None, "101", 0, True])
def test_cli_invalid_reply_binding_is_row_isolated_without_fabricating_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply_id: Any
) -> None:
    store = store_run(tmp_path / "records", "one")
    store_run(store.root, "two", number=2)
    original = store.read_records()["runs"]
    first = github(store, ("applied",), run_id="one")
    second = github(store, ("applied",), run_id="two")

    def request(repo: str, endpoint: str, **kwargs: Any) -> Any:
        result = (first if "/pulls/1" in endpoint else second)(repo, endpoint, **kwargs)
        if endpoint.endswith("/1/comments"):
            return [result[0], {**result[1], "id": reply_id}]
        return result

    monkeypatch.setattr(git_ops, "gh_api", request)
    assert harvest_cli(store, tmp_path / "cache") == 1
    judgments = [o for o in store.read_records()["observations"] if o["payload"]["type"] == "finding-judgment"]
    assert len(judgments) == 1 and judgments[0]["run_id"] == "two"
    assert judgments[0]["reply_captures"][0]["source_reply_id"] == "101"
    assert store.read_records()["runs"] == original


def test_cli_interruption_keeps_captured_replies_and_resumes_remaining_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_run(tmp_path / "records", "one")
    store_run(store.root, "two", number=2)
    original = store.read_records()["runs"]
    first = github(store, ("applied\nFirst reply π",), run_id="one")
    second = github(store, ("not applicable\nSecond reply",), run_id="two")

    def interrupted(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if "/pulls/2" in endpoint:
            raise KeyboardInterrupt
        return first(repo, endpoint, **kwargs)

    cache = tmp_path / "cache"
    monkeypatch.setattr(git_ops, "gh_api", interrupted)
    assert harvest_cli(store, cache) == 130
    completed = [o for o in store.read_records()["observations"] if o["payload"]["type"] == "finding-judgment"]
    assert len(completed) == 1 and completed[0]["run_id"] == "one"
    assert completed[0]["reply_captures"][0]["text"] == "applied\nFirst reply π"
    progress = [json.loads(line) for line in (cache / "progress.jsonl").read_text().splitlines()]
    assert [entry["session_id"] for entry in progress] == ["one"]
    assert all(set(entry) == {"session_id", "labeler_policy_version", "completed_at"} for entry in progress)
    monkeypatch.setattr(git_ops, "gh_api", second)
    assert harvest_cli(store, cache) == 0
    judgments = [o for o in store.read_records()["observations"] if o["payload"]["type"] == "finding-judgment"]
    assert len(judgments) == 2 and completed[0] in judgments
    later = next(o for o in judgments if o["run_id"] == "two")
    assert later["reply_captures"][0]["text"] == "not applicable\nSecond reply"
    frozen = store.read_records()
    assert harvest_cli(store, cache) == 0 and store.read_records() == frozen
    assert store.read_records()["runs"] == original


@pytest.mark.parametrize("state", ["open", "closed", "merged"])
def test_pr_state_never_infers_acceptance_from_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    store = store_run(tmp_path / "records")
    monkeypatch.setattr(git_ops, "gh_api", github(store, state=state))
    assert harvest_cli(store, tmp_path / "cache") == 0
    annotations = [
        o["payload"]["annotation"]
        for o in store.read_records()["observations"]
        if o["payload"]["type"] == "harvest-annotation"
    ]
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation["labels"] == [] and annotation["pr_state"] == state


def test_harvest_record_idempotence_policy_bump_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    store = store_run(tmp_path / "records")
    monkeypatch.setattr(git_ops, "gh_api", github(store, ("applied",)))
    before = store.read_records()
    assert harvest_cli(store, tmp_path / "dry-run-cache", "--dry-run") == 0
    assert "'would_annotate': 1" in capsys.readouterr().out
    assert store.read_records() == before
    assert not (tmp_path / "dry-run-cache").exists()
    assert harvest_cli(store, tmp_path / "first-cache") == 0
    assert "'annotated': 1" in capsys.readouterr().out
    before = store.read_records()
    assert len([o for o in before["observations"] if o["payload"]["type"] == "harvest-annotation"]) == 1
    assert harvest_cli(store, tmp_path / "unchanged-cache") == 0
    assert "'skipped': 1" in capsys.readouterr().out
    assert store.read_records() == before
    monkeypatch.setattr(labeler_versions, "LABELER_POLICY_VERSION", "new-policy")
    assert harvest_cli(store, tmp_path / "new-policy-cache") == 0
    assert "'annotated': 1" in capsys.readouterr().out
    assert len([o for o in store.read_records()["observations"] if o["payload"]["type"] == "harvest-annotation"]) == 2


@pytest.mark.parametrize("field_name", ["changed_files", "findings_fingerprints"])
@pytest.mark.parametrize("value", ['["file.py"]', None, [1]])
def test_harvest_row_refuses_nonrecord_collection_shapes(field_name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field_name):
        HarvestRow.from_mapping({"session_id": "run", field_name: value}, row_number=1)


def pure_annotation(
    disposition: Literal["accepted", "rejected", "ambiguous", "unanswered", "missing"] = "rejected",
    *,
    prior: float | None = None,
    n: int = 0,
) -> Any:
    resolution = PerFindingResolution(
        fingerprint="a" * 64,
        comment_id=1,
        disposition=disposition,
        evidence=(
            {
                "reply_id": 2,
                "body": "maintainer reply",
                "reason": "qualifying:owner",
                "classifier_label": disposition,
                "created_at": "2026-10-04T11:00:00Z",
            },
        ),
        evidence_digest="b" * 64,
    )
    rubric = Rubric(
        PRMergeSignal(merged=True, merged_at="2026-10-04T13:00:00Z"),
        FixAppliedSignal("unknown", 0, 0, []),
        CommentResolutionSignal(1, 1, 0),
        None,
        "pr_review",
        (resolution,),
    )
    return build_annotation(
        HarvestRow("run-1", head_sha="2" * 40, pr_repo="owner/repo", pr_number=1),
        HarvestEvidence(
            ScoringInputs(verifier_verdicts=({"issue_id": 1, "verdict": "consistent"},), length=10, format_valid=True),
            rubric,
            reviewer_logins=("alice",),
            pooled_prior=prior,
            prior_n=n,
        ),
    )


@pytest.mark.parametrize(("prior", "n", "expected_prior"), [(None, 0, None), (0.8, 9, None), (0.8, 10, 0.8)])
def test_annotation_preserves_intrinsic_and_posterior_separation(
    prior: float | None, n: int, expected_prior: float | None
) -> None:
    annotation = pure_annotation(prior=prior, n=n)
    reward = json.loads(annotation.reward_json)
    assert annotation.composite_reward == reward["composite"]
    assert reward["outcome_prior"] == expected_prior
    assert reward["outcome_prior_n"] == n
    assert reward["posterior_cost"] == pytest.approx(1 - expected_prior if expected_prior is not None else 0.5)
    assert annotation.valid_at == "2026-10-04T11:00:00Z" and annotation.has_posterior


def test_record_reviewer_prior_uses_strict_valid_cutoff_latest_shared_reviewers_and_repo_scope(tmp_path: Path) -> None:
    store = store_run(tmp_path / "records", "current")
    for run_id, repo, valid, label, stamp in [
        ("past", "owner/repo", "2026-10-03T10:00:00Z", "rejected", "2026-10-04T10:00:00Z"),
        ("past", "owner/repo", "2026-10-03T10:00:00Z", "accepted", "2026-10-04T11:00:00Z"),
        ("edge", "owner/repo", "2026-10-04T11:00:00Z", "rejected", "2026-10-04T12:00:00Z"),
        ("other", "owner/other", "2026-10-03T10:00:00Z", "rejected", "2026-10-04T12:00:00Z"),
    ]:
        if not any(r["run_id"] == run_id for r in store.read_records()["runs"]):
            store.commit_run(projection_run(run_id, repo_slug=repo))
        annotation = asdict(pure_annotation())
        annotation.update(labels=[label], valid_at=valid)
        store.append_observation(
            observation(
                f"{run_id}:{stamp}",
                schema_version="daydream.observation.v3",
                run_id=run_id,
                item_uid=None,
                role="automatic",
                valid_at=valid,
                observed_at=stamp,
                semantic_evidence=annotation,
                evidence_digest=hashlib.sha256(canonical_json(annotation).encode()).hexdigest(),
                payload={"type": "harvest-annotation", "annotation": annotation, "labeler_policy_version": "test"},
            )
        )
    services = make_harvest_services(HarvestConfig(store.root, snapshot_id(store)))
    scoped = services.reviewer_prior(
        ("alice",), before_valid_at="2026-10-04T11:00:00+00:00", exclude_session="current", repo_slug="owner/repo"
    )
    unscoped = services.reviewer_prior(
        ("alice",), before_valid_at="2026-10-04T11:00:00Z", exclude_session="current", repo_slug=None
    )
    assert scoped == (0.0, 1) and unscoped == (0.5, 2)


def test_harvest_rate_limit_preserves_successful_record_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_run(tmp_path / "records", "one")
    second = projection_run("two")
    second["original_task"]["value"]["pr"]["number"] = 2
    store.commit_run(second)
    original = store.read_records()["runs"]
    successful = github(store, ("applied\nCaptured before rate limit",), run_id="one")

    def request(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if "/pulls/2" in endpoint:
            raise git_ops.RateLimitError("rate_limit")
        return successful(repo, endpoint, **kwargs)

    monkeypatch.setattr(git_ops, "gh_api", request)
    monkeypatch.setattr("daydream.training.harvest.time.sleep", lambda seconds: None)
    config = HarvestConfig(store.root, snapshot_id(store), cache_dir=tmp_path / "cache", gh_request_spacing_sec=0)
    assert harvest_cli(store, tmp_path / "cache") == 1
    assert make_harvest_services(config).completed_sessions() == {"one"}
    judgments = [o for o in store.read_records()["observations"] if o["payload"]["type"] == "finding-judgment"]
    assert len(judgments) == 1 and judgments[0]["run_id"] == "one"
    assert judgments[0]["reply_captures"][0]["text"] == "applied\nCaptured before rate limit"
    assert store.read_records()["runs"] == original


def test_run_label_human_precedence_and_temporal_eligibility_preserve_intrinsic_reward(tmp_path: Path) -> None:
    from daydream.training.record_evidence import latest_annotation

    store = store_run(tmp_path / "records")
    annotation = asdict(pure_annotation())

    def append_auto(identity: str, observed: str) -> None:
        store.append_observation(
            observation(
                identity,
                schema_version="daydream.observation.v3",
                item_uid=None,
                role="automatic",
                observed_at=observed,
                semantic_evidence=annotation,
                evidence_digest=hashlib.sha256(canonical_json(annotation).encode()).hexdigest(),
                payload={"type": "harvest-annotation", "annotation": annotation, "labeler_policy_version": "test"},
            )
        )

    append_auto("auto-first", "2026-10-04T12:00:00Z")
    human = {"label": "accepted"}
    store.append_observation(
        observation(
            "human",
            item_uid=None,
            valid_at="2026-10-05T10:00:00Z",
            observed_at="2026-10-05T12:00:00Z",
            semantic_evidence=human,
            evidence_digest=hashlib.sha256(canonical_json(human).encode()).hexdigest(),
            payload={"type": "run-label", **human},
        )
    )
    append_auto("auto-later", "2026-10-06T12:00:00Z")
    selected = store.read_snapshot(snapshot_id(store))
    effective = latest_annotation(selected, "run-1")
    assert effective is not None and effective["labels"] == ["accepted"]
    assert effective["composite_reward"] == annotation["composite_reward"]
    assert effective["reward_json"] == annotation["reward_json"]
    historical = latest_annotation(
        store.read_snapshot(snapshot_id(store, valid_before="2026-10-04T23:59:59Z")), "run-1"
    )
    assert historical is not None and historical["labels"] == ["rejected"]


def harvest_cli(store: LocalRecordStore, cache: Path, *extra: str) -> int:
    return cli_main(
        [
            "corpus",
            "harvest",
            "--store",
            str(store.root),
            "--snapshot-id",
            snapshot_id(store),
            "--cache-dir",
            str(cache),
            "--gh-spacing-sec",
            "0",
            *extra,
        ]
    )


def test_cli_orphan_relink_is_enrichment_without_rewriting_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_run(tmp_path / "records", number=None)
    head = store.read_records()["runs"][0]["original_task"]["value"]["analyzed_revision"]["head_sha"]
    original = store.read_records()["runs"]
    responder = github(store, ("applied",))

    def request(repo: str, endpoint: str, **kwargs: Any) -> Any:
        return (
            [{"number": 7, "head": {"sha": head}}] if "/commits/" in endpoint else responder(repo, endpoint, **kwargs)
        )

    monkeypatch.setattr(git_ops, "gh_api", request)
    assert harvest_cli(store, tmp_path / "cache") == 0
    history = store.read_records()["observations"]
    pr = next(o for o in history if o["payload"]["type"] == "enrichment" and o["payload"]["kind"] == "pr")
    assert pr["payload"]["evidence"]["value"] == {"number": 7, "repo": "owner/repo"}
    assert next(
        o["payload"]["annotation"]["labels"] for o in history if o["payload"]["type"] == "harvest-annotation"
    ) == ["accepted"]
    assert store.read_records()["runs"] == original


@pytest.mark.parametrize(("status", "expected_exit"), [(404, 0), (422, 0), (401, 1), (500, 1)])
def test_cli_pr_absence_degrades_unknown_but_auth_and_outage_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, expected_exit: int
) -> None:
    store = store_run(tmp_path / "records")

    def request(repo: str, endpoint: str, **kwargs: Any) -> Any:
        raise git_ops.GitError(f"HTTP {status}")

    monkeypatch.setattr(git_ops, "gh_api", request)
    assert harvest_cli(store, tmp_path / "cache") == expected_exit
    annotations = [
        o["payload"]["annotation"]
        for o in store.read_records()["observations"]
        if o["payload"]["type"] == "harvest-annotation"
    ]
    if expected_exit:
        assert annotations == []
    else:
        assert len(annotations) == 1 and annotations[0]["labels"] == [] and not annotations[0]["has_posterior"]


def test_cli_reviewer_lookup_error_keeps_decisive_reply_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_run(tmp_path / "records")
    responder = github(store, ("applied",))

    def request(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/reviews"):
            raise git_ops.GitError("HTTP 503")
        return responder(repo, endpoint, **kwargs)

    monkeypatch.setattr(git_ops, "gh_api", request)
    assert harvest_cli(store, tmp_path / "cache") == 0
    annotation = next(
        o["payload"]["annotation"]
        for o in store.read_records()["observations"]
        if o["payload"]["type"] == "harvest-annotation"
    )
    assert annotation["labels"] == ["accepted"] and annotation["has_posterior"]


@pytest.mark.parametrize("landed", [True, False])
def test_cli_local_commit_signal_is_not_pr_posterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, landed: bool
) -> None:
    from tests.conftest import _make_repo_with_main
    from tests.harness.git_helpers import commit, git
    from tests.harness.trajectory import diff_adding

    repo = _make_repo_with_main(tmp_path)
    (repo / "app.py").write_text("existing\n")
    git(repo, "add", "app.py")
    head = commit(repo, "app baseline")
    git(repo, "checkout", "-b", "feature")
    if landed:
        (repo / "app.py").write_text("existing\nguarded = True\n")
        git(repo, "add", "app.py")
        commit(repo, "apply suggested guard")
    store = store_run(
        tmp_path / "records",
        number=None,
        source_path=repo,
        head_sha=head,
        recommended_patch=diff_adding("guarded = True"),
    )
    monkeypatch.setattr(git_ops, "gh_api", lambda repo, endpoint, **kwargs: [])
    assert harvest_cli(store, tmp_path / "cache") == 0
    annotation = next(
        o["payload"]["annotation"]
        for o in store.read_records()["observations"]
        if o["payload"]["type"] == "harvest-annotation"
    )
    assert annotation["labels"] == (["accepted"] if landed else ["rejected"])
    assert not annotation["has_posterior"] and json.loads(annotation["reward_json"]).get("posterior_cost") is None


@pytest.mark.parametrize(
    "remote", ["https://token-value@github.com/owner/repo.git", "https://evil.example/owner/repo.git"]
)
def test_cli_clone_uses_allowed_credential_free_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: str
) -> None:
    import shutil

    from tests.conftest import _make_repo_with_main

    repo = _make_repo_with_main(tmp_path)
    store = store_run(tmp_path / "records", remote_url=remote)
    destinations = []

    def clone(url: str, target: Path, token: str | None, **kwargs: Any) -> None:
        assert url == "https://github.com/owner/repo" and token == "out-of-band-token"
        destinations.append(target)
        shutil.copytree(repo, target)

    monkeypatch.setenv("DAYDREAM_GIT_TOKEN", "out-of-band-token")
    monkeypatch.setattr(git_ops, "clone_with_token", clone)
    monkeypatch.setattr(git_ops, "gh_api", github(store, ("applied",)))
    assert harvest_cli(store, tmp_path / "cache") == 0
    assert bool(destinations) == ("github.com" in remote)
    if "evil.example" in remote:
        assert not (tmp_path / "cache" / "repos" / "owner" / "repo").exists()


def test_cli_edited_automatic_evidence_becomes_new_generation_and_requeues_old_human_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.training.record_evidence import sessions_from_snapshot

    store = store_run(tmp_path / "records")
    monkeypatch.setattr(git_ops, "gh_api", github(store, ("Needs investigation.",)))
    assert harvest_cli(store, tmp_path / "first-cache") == 0
    state = tmp_path / "state"
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "build",
                "--store",
                str(store.root),
                "--snapshot-id",
                snapshot_id(store),
                "--state-dir",
                str(state),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "label",
                "--state-dir",
                str(state),
                "--batch",
                "1",
                "--disposition",
                "accepted",
                "--rationale",
                "human confirmed",
                "--labeler",
                "alice",
            ]
        )
        == 0
    )
    monkeypatch.setattr(git_ops, "gh_api", github(store, ("applied",)))
    assert harvest_cli(store, tmp_path / "accepted-cache") == 0
    old_snapshot = snapshot_id(store)
    monkeypatch.setattr(git_ops, "gh_api", github(store, ("not applicable",)))
    assert harvest_cli(store, tmp_path / "fresh-cache") == 0
    selected = store.read_snapshot(snapshot_id(store))
    resolution = sessions_from_snapshot(selected)[0]["resolutions"][0]
    assert resolution["disposition"] == "rejected" and not resolution["conflict"]
    assert sessions_from_snapshot(store.read_snapshot(old_snapshot))[0]["resolutions"][0]["disposition"] == "accepted"
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "build",
                "--store",
                str(store.root),
                "--snapshot-id",
                snapshot_id(store),
                "--state-dir",
                str(state),
            ]
        )
        == 0
    )
    queue = json.loads((state / "queue.json").read_text())
    assert len(queue) == 1 and queue[0]["status"] == "reopened" and queue[0]["prior_disposition"] == "accepted"


def test_record_reviewer_prior_orders_fractional_utc_spelling_chronologically(tmp_path: Path) -> None:
    store = store_run(tmp_path / "records", "current")
    store.commit_run(projection_run("past", repo_slug="owner/repo"))
    generations: list[tuple[str, Literal["accepted", "rejected"], str]] = [
        ("old", "rejected", "2026-10-04T12:00:00Z"),
        ("new", "accepted", "2026-10-04T12:00:00.000001+00:00"),
    ]
    for identity, disposition, observed_at in generations:
        annotation = asdict(pure_annotation(disposition))
        store.append_observation(
            observation(
                identity,
                schema_version="daydream.observation.v3",
                run_id="past",
                item_uid=None,
                role="automatic",
                valid_at="2026-10-03T10:00:00Z",
                observed_at=observed_at,
                semantic_evidence=annotation,
                evidence_digest=hashlib.sha256(canonical_json(annotation).encode()).hexdigest(),
                payload={"type": "harvest-annotation", "annotation": annotation, "labeler_policy_version": "test"},
            )
        )
    provider = make_harvest_services(HarvestConfig(store.root, snapshot_id(store)))
    assert provider.reviewer_prior(
        ("alice",), before_valid_at="2026-10-05T00:00:00Z", exclude_session="current", repo_slug="owner/repo"
    ) == (0.0, 1)
