"""Fresh bronze reaches adjudication without a preliminary canonical harvest."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream import cli, git_ops
from daydream.archive import hydrate, license_enrich
from daydream.archive.hydrate_client import FakeHub
from daydream.archive.index import append_label_observation, label_observation_history, upsert_run
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.training.adjudication.observations import load_observations
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION, REPLY_CLASSIFIER_VERSION
from tests.fixtures.training.build_archive import _MINIMAL_TRAJECTORY
from tests.fixtures.training.build_hub_snapshot import PINNED_POLICY_FIXTURE
from tests.harness.git_helpers import git
from tests.harness.trajectory import make_manifest


def _cli(args: list[str]) -> int:
    with pytest.raises(SystemExit) as result:
        cli.main(["corpus", *args])
    return int(result.value.code or 0)


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def fresh_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[dict[str, Any]]]:
    root = tmp_path / "hydrated"
    run = root / "runs" / "semantic"
    manifest = make_manifest(
        session_id="semantic", archive_path="/archive/semantic", repo_slug="org/repo",
        head_sha="b" * 40, pr_repo="org/repo", pr_number=7,
        pipeline_status="succeeded", remote_url="https://github.com/org/repo",
        profile_schema_version=2, profile_name="pr_review", profile_source_kind="builtin",
        profile_digest="f" * 64,
    )
    fingerprints = [str(i) * 64 for i in range(1, 6)]
    source = FakeHub(repo_id="org/bronze", private=True, files={
        "semantic/manifest.json": json.dumps(manifest.to_dict()).encode(),
        "semantic/trajectory.json": json.dumps(_MINIMAL_TRAJECTORY).encode(),
        "semantic/findings.json": json.dumps({
            "findings": [{"fingerprint": fp, "stack": "python"} for fp in fingerprints],
        }).encode(),
    })
    source.commit_revision("a" * 40)

    def hub(_repo_id: str, **_kwargs: Any) -> FakeHub:
        return source

    class LicenseResolver:
        def resolve(self, repo_slug: str, repo_commit: str | None) -> license_enrich.EnrichedEvidence:
            return license_enrich.EnrichedEvidence("MIT", "github:org/repo", repo_commit or "b" * 40)

    monkeypatch.setattr(hydrate, "_make_client", hub)
    monkeypatch.setattr(license_enrich, "_make_license_resolver", LicenseResolver)
    monkeypatch.setenv("HF_TOKEN", "offline-fixture-token")
    monkeypatch.setenv("GITHUB_TOKEN", "offline-fixture-token")
    assert _cli([
        "hydrate-hub", "--source-repo", source.repo_id, "--source-revision", "a" * 40,
        "--destination-repo", source.repo_id, "--stage-dir", str(root),
        "--license-policy", str(PINNED_POLICY_FIXTURE),
    ]) == 0
    assert "resolutions" not in json.loads((run / "trajectory.json").read_text())
    assert label_observation_history(root, "semantic") == []
    comments: list[dict[str, Any]] = []
    for number, fp in enumerate(fingerprints[:4], start=1):
        comments.append({
            "id": number, "user": {"login": "daydream-runner"},
            "body": f"finding\n{finding_marker(fp)}\n{DAYDREAM_FOOTER}",
        })
    for number, parent, body in (
        (10, 1, "Good catch, applied."),
        (11, 2, "False positive, this is intentional."),
        (12, 3, "Good catch, applied."),
        (13, 3, "False positive, this is intentional."),
    ):
        comments.append({
            "id": number, "in_reply_to_id": parent, "body": body,
            "user": {"login": "reviewer"}, "author_association": "OWNER",
            "created_at": "2026-09-10T10:00:00Z",
        })

    def github(_repo: Path, endpoint: str, **_kwargs: Any) -> Any:
        if endpoint == "repos/org/repo/pulls/7":
            return {"merged": True, "merged_at": "2026-09-10T11:00:00Z", "user": {"login": "author"}}
        if endpoint.endswith("/reviews"):
            return [{"user": {"login": "reviewer"}}]
        if endpoint.endswith("/comments"):
            return comments
        pytest.fail(f"Unexpected GitHub request: {endpoint}")

    monkeypatch.setattr(git_ops, "gh_api", github)
    return root, comments


def _materialize(root: Path, out: Path) -> int:
    curation, = (root / "curated").glob("*/curation-manifest.json")
    return _cli([
        "adjudicate", "materialize", "--index-root", str(root), "--out-dir", str(out),
        "--curation-id", curation.parent.name, "--sanitized-hub-commit", "a" * 40,
        "--source-hub-commit", "a" * 40,
        "--archive-index-digest", hashlib.sha256((root / "index.db").read_bytes()).hexdigest(),
        "--evidence-observed-at", "2026-09-11T00:00:00+00:00",
    ])


def test_fresh_semantic_preview_is_complete_and_readonly(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path, git_repo: Path,
) -> None:
    root, _comments = fresh_archive
    before = _tree(root)
    out = tmp_path / "preview"
    assert _materialize(root, out) == 0
    assert _tree(root) == before
    assert label_observation_history(root, "semantic") == []
    records = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
    assert len(records) == len({r["record_id"] for r in records}) == 5
    assert {r["fingerprint"]: r["disposition"] for r in records} == {
        "1" * 64: "accepted", "2" * 64: "rejected", "3" * 64: "ambiguous",
        "4" * 64: "unanswered", "5" * 64: "missing",
    }
    assert all(r["evidence_digest"] for r in records)
    assert all(r["profile"]["profile_name"] == "pr_review" and r["stack"] == "python" for r in records)
    assert _materialize(root, tmp_path / "repeat") == 0
    assert (out / "sessions.jsonl").read_bytes() == (tmp_path / "repeat" / "sessions.jsonl").read_bytes()
    # The actual semantic harvester emits the same per-finding evidence.
    canonical = tmp_path / "canonical"
    shutil.copytree(root, canonical)
    upsert_run(canonical, make_manifest(
        session_id="semantic", archive_path=str(canonical / "runs" / "semantic"),
        source_path=str(git_repo), repo_slug="org/repo", head_sha="b" * 40,
        pr_repo="org/repo", pr_number=7,
    ))
    assert _cli([
        "harvest", "--archive-dir", str(canonical), "--cache-dir", str(tmp_path / "cache"),
        "--gh-spacing-sec", "0",
    ]) == 0
    annotation, = label_observation_history(canonical, "semantic")
    semantic = json.loads(annotation["rubric_json"])["per_finding_resolutions"]
    assert {(r["fingerprint"], r["disposition"], r["evidence_digest"]) for r in semantic} == {
        (r["fingerprint"], r["disposition"], r["evidence_digest"]) for r in records
    }


@pytest.mark.parametrize("history_kind", ["rater", "conflict", "legacy"])
@pytest.mark.parametrize("target_fingerprint", ["1", "3"])
def test_imported_finding_reaches_queue_and_canonical_harvest(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path,
    history_kind: str,
    target_fingerprint: str,
) -> None:
    root, comments = fresh_archive
    preview, state, backup = (tmp_path / name for name in ("preview", "state", "backup"))
    assert _materialize(root, preview) == 0
    records = [json.loads(line) for line in (preview / "sessions.jsonl").read_text().splitlines()]
    finding = next(row for row in records if row["fingerprint"] == target_fingerprint * 64)
    shutil.copytree(root, backup)
    append_label_observation(
        backup, "semantic", labels=["accepted"], pr_state=None,
        labeler_version=ADJUDICATION_LABELER_VERSION, evidence_sha="b" * 40,
        rubric_json=json.dumps({"per_finding_resolutions": [{
            **finding, "disposition": "accepted", "human_labeler": "alice", "human_role": "rater",
        }]}), source="human", observed_at="2026-09-11T00:00:00+00:00",
        reply_classifier_version="legacy" if history_kind == "legacy" else REPLY_CLASSIFIER_VERSION,
    )
    if history_kind == "conflict":
        append_label_observation(
            backup, "semantic", labels=["rejected"], pr_state=None,
            labeler_version=ADJUDICATION_LABELER_VERSION, evidence_sha="b" * 40,
            rubric_json=json.dumps({"per_finding_resolutions": [{
                **finding, "disposition": "rejected", "human_labeler": "bob", "human_role": "rater",
            }]}), source="human", observed_at="2026-09-11T01:00:00+00:00",
            reply_classifier_version=REPLY_CLASSIFIER_VERSION,
        )
    source = _tree(backup)
    args = [
        "adjudicate", "import-local-observations", "--archive-root", str(backup),
        "--index-root", str(root), "--archive-dir", str(root), "--state-dir", str(state),
    ]
    assert _cli([*args, "--dry-run"]) == 0
    assert not state.exists()
    assert _cli(args) == 0
    assert _tree(backup) == source
    observations = load_observations(state / "observations.jsonl")
    assert len(observations) == (2 if history_kind == "conflict" else 1)
    assert observations[0]["record_id"] == finding["record_id"]
    assert observations[0]["labeler"] == "alice"
    assert _cli(args) == 0
    assert load_observations(state / "observations.jsonl") == observations
    assert _cli(["adjudicate", "build", "--index-root", str(preview), "--state-dir", str(state)]) == 0
    queue = json.loads((state / "queue.json").read_text())
    assert {item["fingerprint"] for item in queue} == {str(i) * 64 for i in (3, 4, 5)} | {target_fingerprint * 64}
    assert _cli([
        "adjudicate", "export", "--index-root", str(preview), "--state-dir", str(state), "--dry-run",
    ]) == 0
    harvest = [
        "adjudicate", "harvest-snapshot", "--index-root", str(root),
        "--materialize-dir", str(preview), "--archive-dir", str(root), "--state-dir", str(state),
    ]
    # Editing even a decisive finding's external evidence aborts before any write.
    before = _tree(root)
    original = comments[4]["body"]
    comments[4]["body"] = "False positive, this is intentional."
    assert _cli(harvest) == 1
    assert _tree(root) == before
    assert not (preview / "annotations.jsonl").exists()
    comments[4]["body"] = original
    assert _cli(harvest) == 0
    annotations = [json.loads(line) for line in (preview / "annotations.jsonl").read_text().splitlines()]
    assert len(annotations) == len({row["record_id"] for row in annotations}) == 5
    assert {(r["record_id"], r["evidence_digest"]) for r in records} == {
        (r["record_id"], r["evidence_digest"]) for r in annotations
    }
    resolved = next(row for row in annotations if row["fingerprint"] == target_fingerprint * 64)
    if history_kind == "rater":
        assert resolved["disposition"] == "accepted"
        assert resolved["human_labeler"] == "alice"
        assert resolved["resolutions"][0]["disposition"] == "accepted"
    else:
        assert resolved["disposition"] == "ambiguous"
    history = label_observation_history(root, "semantic")
    assert len(history) == (3 if history_kind == "conflict" else 2)
    assert _cli(harvest) == 0
    assert label_observation_history(root, "semantic") == history
    curation, = (root / "curated").glob("*/curation-manifest.json")
    assert _cli([
        "adjudicate", "publish-final", "--index-root", str(root), "--materialize-dir", str(preview),
        "--archive-dir", str(root), "--curation-bundle-dir", str(curation.parent),
        "--state-dir", str(state), "--hub-repo", "org/annotations", "--dry-run",
    ]) == 0
    immutable_history = [
        json.loads(line) for line in (preview / "final-bundle" / "label-observations.jsonl").read_text().splitlines()
    ]
    assert len(immutable_history) == len(history)
    assert any(row["source"] == "human" and "alice" in row["rubric_json"] for row in immutable_history)
    assert (preview / "final-bundle" / "sessions.jsonl").read_bytes() == (preview / "annotations.jsonl").read_bytes()
    if history_kind in {"legacy", "conflict"}:
        assert _cli([
            "adjudicate", "label", "--state-dir", str(state), "--record-id", finding["record_id"],
            "--disposition", "accepted", "--rationale", "verified against current evidence",
            "--labeler", "carol", "--role", "adjudicator",
        ]) == 0
        assert _cli(harvest) == 0
        rows = [json.loads(line) for line in (preview / "annotations.jsonl").read_text().splitlines()]
        decision = next(row for row in rows if row["record_id"] == finding["record_id"])
        assert decision["disposition"] == "accepted"
        assert decision["human_labeler"] == "carol"
        assert len(load_observations(state / "observations.jsonl")) == len(observations) + 1


def test_import_does_not_link_a_backup_to_itself_outside_the_pinned_curation(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path,
) -> None:
    root, _comments = fresh_archive
    backup, state = tmp_path / "backup", tmp_path / "state"
    upsert_run(backup, make_manifest(
        session_id="outside", repo_slug="org/repo", base_sha="c" * 40, head_sha="d" * 40,
    ))
    append_label_observation(
        backup, "outside", labels=["accepted"], pr_state=None,
        labeler_version=ADJUDICATION_LABELER_VERSION, evidence_sha="d" * 40,
    )
    assert _cli([
        "adjudicate", "import-local-observations", "--archive-root", str(backup),
        "--index-root", str(root), "--archive-dir", str(root), "--state-dir", str(state),
    ]) == 0
    report = json.loads((state / "import-report.json").read_text())
    assert report["identity_summary"]["outside"]["validation_outcome"] == "unmatched"
    assert report["merge"]["appended"] == 0
    assert label_observation_history(root, "outside") == []


def test_harvest_dry_run_does_not_prepare_bronze_or_resume_state(
    fresh_archive: tuple[Path, list[dict[str, Any]]], git_repo: Path,
) -> None:
    root, _comments = fresh_archive
    run = root / "runs" / "semantic"
    manifest = make_manifest(
        session_id="semantic", archive_path=str(run), source_path=str(git_repo),
        repo_slug="org/repo", pr_repo="org/repo", pr_number=7,
        head_sha=git(git_repo, "rev-parse", "HEAD"), base_branch="main", branch="main",
    )
    (run / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    upsert_run(root, manifest)
    before, repo_before = _tree(root), _tree(git_repo)
    assert _cli([
        "harvest", "--archive-dir", str(root), "--cache-dir", str(root / "cache"), "--dry-run",
    ]) == 0
    assert _tree(root) == before
    assert _tree(git_repo) == repo_before
    assert label_observation_history(root, "semantic") == []


def test_preview_keeps_findings_when_the_pr_is_unavailable(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _comments = fresh_archive

    def missing_pr(*_args: Any, **_kwargs: Any) -> Any:
        raise git_ops.GitError("PR unavailable (HTTP 404)")

    monkeypatch.setattr(git_ops, "gh_api", missing_pr)
    assert _materialize(root, tmp_path / "preview") == 0
    records = [json.loads(line) for line in (tmp_path / "preview" / "sessions.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert {r["disposition"] for r in records} == {"missing"}


def test_preview_accepts_an_explicitly_empty_finding_inventory(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path,
) -> None:
    root, _comments = fresh_archive
    (root / "runs" / "semantic" / "findings.json").write_text('{"findings": []}')
    before = _tree(root)
    preview = tmp_path / "preview"
    assert _materialize(root, preview) == 0
    assert (preview / "sessions.jsonl").read_bytes() == b""
    assert _tree(root) == before


def test_preview_rate_limit_aborts_without_any_output_or_input_writes(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _comments = fresh_archive

    def rate_limit(*_args: Any, **_kwargs: Any) -> Any:
        raise git_ops.RateLimitError("rate limit", retry_after=0)

    monkeypatch.setattr(git_ops, "gh_api", rate_limit)
    before = _tree(root)
    assert _materialize(root, tmp_path / "preview") == 1
    assert not (tmp_path / "preview").exists()
    assert _tree(root) == before


def test_canonical_harvest_rejects_new_findings_before_writes(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path,
) -> None:
    root, _comments = fresh_archive
    preview = tmp_path / "preview"
    assert _materialize(root, preview) == 0
    path = root / "runs" / "semantic" / "findings.json"
    findings = json.loads(path.read_text())
    findings["findings"].append({"fingerprint": "6" * 64})
    path.write_text(json.dumps(findings))
    before = _tree(root)
    assert _cli([
        "adjudicate", "harvest-snapshot", "--index-root", str(root),
        "--materialize-dir", str(preview), "--archive-dir", str(root), "--state-dir", str(tmp_path / "state"),
    ]) == 1
    assert _tree(root) == before
    assert not (preview / "annotations.jsonl").exists()


@pytest.mark.parametrize("reason", [
    "unknown_fingerprint", "record_identity_mismatch", "evidence_digest_mismatch", "ambiguous_fingerprint",
])
def test_invalid_imported_finding_mapping_stays_reviewable_without_becoming_a_judgment(
    fresh_archive: tuple[Path, list[dict[str, Any]]], tmp_path: Path, reason: str,
) -> None:
    root, _comments = fresh_archive
    preview, backup, state = (tmp_path / name for name in ("preview", "backup", "state"))
    assert _materialize(root, preview) == 0
    finding = json.loads((preview / "sessions.jsonl").read_text().splitlines()[0])
    resolution = {**finding, "disposition": "accepted", "human_labeler": "alice"}
    if reason == "unknown_fingerprint":
        resolution["fingerprint"] = "f" * 64
    elif reason == "record_identity_mismatch":
        resolution["record_id"] = "f" * 64
    elif reason == "evidence_digest_mismatch":
        resolution["evidence_digest"] = "f" * 64
    resolutions = [resolution, resolution] if reason == "ambiguous_fingerprint" else [resolution]
    shutil.copytree(root, backup)
    append_label_observation(
        backup, "semantic", labels=["accepted"], pr_state=None, source="human",
        labeler_version=ADJUDICATION_LABELER_VERSION, evidence_sha="b" * 40,
        rubric_json=json.dumps({"per_finding_resolutions": resolutions}),
        reply_classifier_version=REPLY_CLASSIFIER_VERSION,
    )
    assert _cli([
        "adjudicate", "import-local-observations", "--archive-root", str(backup),
        "--index-root", str(root), "--archive-dir", str(root), "--state-dir", str(state),
    ]) == 0
    assert load_observations(state / "observations.jsonl") == []
    report = json.loads((state / "import-report.json").read_text())
    assert {decision["reason"] for decision in report["finding_decisions"]} == {reason}
    assert len(report["finding_decisions"]) == len(resolutions)
    assert len(label_observation_history(root, "semantic")) == 1
