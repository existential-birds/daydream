"""Real-path tests for the corpus adjudicate sub-verbs (style: test_cli_label.py)."""
import json
from pathlib import Path

import pytest

from daydream.commands.corpus import _handle_corpus_command
from daydream.training.adjudication import cli as adjudication_cli
from daydream.training.adjudication.cli import handle_adjudicate
from daydream.training.adjudication.final_bundle import final_snapshot_id
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.observations import append_observation
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.publish import (
    AnnotationHubClient,
    FinalAnnotationBundle,
    publish_final_annotation_bundle,
)
from daydream.training.labeler_versions import (
    ADJUDICATION_LABELER_VERSION,
    REPLY_CLASSIFIER_VERSION,
    RUBRIC_SCHEMA_VERSION,
)
from tests.fixtures.training.build_hub_snapshot import AnnotationsHub
from tests.harness.adjudication import (
    accepted_observation,
    write_checkpoint_inputs,
    write_sessions_index,
    write_sessions_jsonl,
)
from tests.test_training_adjudication_publish import _final_bundle


def _install_annotation_hub(monkeypatch: pytest.MonkeyPatch, hub: AnnotationHubClient,) -> None:
    """Route only the external Hub boundary at an in-memory implementation."""

    monkeypatch.setattr(adjudication_cli, "_make_client", lambda _repo_id: hub)


def _wired_hub(monkeypatch: pytest.MonkeyPatch, *, repo_id: str = "org/private-annotations", private: bool = True,
) -> "AnnotationsHub":
    """Build the revision-aware fixture and route the CLI at it in one step."""

    hub = AnnotationsHub(repo_id=repo_id, private=private)
    _install_annotation_hub(monkeypatch, hub)
    return hub


def _publish_final(
    index_root: Path, materialize_dir: Path, archive_dir: Path, state_dir: Path, *, dry_run: bool = False,
) -> int:
    argv = ["publish-final", "--index-root", str(index_root),
        "--materialize-dir", str(materialize_dir), "--archive-dir", str(archive_dir),
        "--curation-bundle-dir", str(index_root), "--state-dir", str(state_dir), "--hub-repo", "org/private-ds",
    ]
    if dry_run:
        argv.append("--dry-run")
    return handle_adjudicate(argv)


def _publish_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> tuple[Path, Path, "AnnotationsHub"]:
    """Publish the standard ``cur-1`` checkpoint and return the wired hub."""
    state, manifest = write_checkpoint_inputs(tmp_path)
    hub = _wired_hub(monkeypatch)
    assert handle_adjudicate([
        "publish-state", "--state-dir", str(state), "--manifest", str(manifest), "--hub-repo", hub.repo_id,
    ]) == 0
    return state, manifest, hub


def _console_text(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    return "".join((captured.out + captured.err).split()).replace("║", "")

def test_adjudicate_label_records_human_observation(tmp_path: Path) -> None:
    _built_queue(tmp_path)
    queue = json.loads((tmp_path / "adj" / "queue.json").read_text())
    record_id = str(queue[0]["record_id"])
    rc = _handle_corpus_command(["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--record-id", record_id, "--disposition", "accepted",
         "--rationale", "reply confirms fix", "--labeler", "kevin"]
    )
    assert rc == 0
    lines = (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()
    obs = [json.loads(line) for line in lines]
    assert obs[-1]["disposition"] == "accepted" and obs[-1]["role"] == "rater"
    assert obs[-1]["record_id"] == record_id

def test_adjudicate_label_unknown_record_id_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _built_queue(tmp_path)
    rc = _handle_corpus_command(["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--record-id", "a" * 64, "--disposition", "accepted",
         "--rationale", "reply confirms fix", "--labeler", "kevin"]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "a" * 64 in captured.out + captured.err

def test_adjudicate_label_batch_n_processes_unresolved_in_order(tmp_path: Path) -> None:
    _built_queue(tmp_path)
    rc = _handle_corpus_command(["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--batch", "1", "--disposition", "rejected", "--rationale", "stale finding",
         "--labeler", "kevin"]
    )
    assert rc == 0
    obs = [json.loads(line) for line in (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()]
    assert len(obs) == 1  # one observation per item; re-run advances, never duplicates
    rc2 = _handle_corpus_command(["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--batch", "1", "--disposition", "rejected", "--rationale", "stale finding",
         "--labeler", "kevin"]
    )
    assert rc2 == 0
    obs2 = [json.loads(line) for line in (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()]
    assert len(obs2) == 2 and obs2[0]["record_id"] != obs2[1]["record_id"]  # resume point advanced

def test_adjudicate_show_lists_queue_and_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _built_queue(tmp_path)
    rc = _handle_corpus_command(["adjudicate", "show", "--state-dir", str(tmp_path / "adj")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ambiguous" in out and "unanswered" in out


def _built_queue(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    root = tmp_path
    state = tmp_path / "adj"
    write_sessions_index(root)
    assert _handle_corpus_command(["adjudicate", "build", "--index-root", str(root), "--state-dir", str(state)]
    ) == 0
    queue = json.loads((state / "queue.json").read_text())
    return root, state, queue


def _seed_adjudicated(tmp_path: Path) -> tuple[Path, Path]:
    """Hydrated index + state dir with a built queue, one human decision, and
    a digest-pinned preview ledger (export harvest input)."""

    root, state, queue = _built_queue(tmp_path)
    record_id = str(queue[0]["record_id"])
    assert _handle_corpus_command(["adjudicate", "label", "--state-dir", str(state), "--record-id", record_id,
         "--disposition", "accepted", "--rationale", "reply confirms fix", "--labeler", "kevin"]
    ) == 0
    run_preview(root, state / "preview-ledger.json")
    return root, state


def _seed_with_conflict(tmp_path: Path) -> tuple[Path, Path]:
    """State dir where two records each have two disagreeing human raters;
    the first record's conflict is older than the second's."""

    root, state, queue = _built_queue(tmp_path)
    obs_path = state / "observations.jsonl"
    older_record, newer_record = queue[0], queue[1]
    common = {"rationale": "independent pass", "role": "rater",
        "valid_at": "2024-01-01T00:00:00+00:00", "rubric_version": "984-adjudicate-r1",
    }
    append_observation(obs_path, {"record_id": str(older_record["record_id"]), "disposition": "accepted",
        "evidence_digest": str(older_record["evidence_digest"]),
        "evidence": older_record["evidence"], "labeler": "rater-one",
        "observed_at": "2024-01-01T00:00:00+00:00", **common,
    })
    append_observation(obs_path, {"record_id": str(older_record["record_id"]), "disposition": "rejected",
        "evidence_digest": str(older_record["evidence_digest"]),
        "evidence": older_record["evidence"], "labeler": "old-conflict",
        "observed_at": "2024-01-02T00:00:00+00:00", **common,
    })
    append_observation(obs_path, {"record_id": str(newer_record["record_id"]), "disposition": "rejected",
        "evidence_digest": str(newer_record["evidence_digest"]),
        "evidence": newer_record["evidence"], "labeler": "rater-two",
        "observed_at": "2024-02-01T00:00:00+00:00", **common,
    })
    append_observation(obs_path, {"record_id": str(newer_record["record_id"]), "disposition": "accepted",
        "evidence_digest": str(newer_record["evidence_digest"]),
        "evidence": newer_record["evidence"], "labeler": "new-conflict",
        "observed_at": "2024-02-02T00:00:00+00:00", **common,
    })
    return root, state

def test_export_writes_projector_shape_and_dry_run_validates_only(tmp_path: Path) -> None:
    root, state = _seed_adjudicated(tmp_path)
    rc = _handle_corpus_command(["adjudicate", "export", "--index-root", str(root), "--state-dir", str(state),
         "--out", str(tmp_path / "export.jsonl"), "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "export.jsonl").exists()  # dry-run validates without writing
    rc = _handle_corpus_command(["adjudicate", "export", "--index-root", str(root), "--state-dir", str(state),
         "--out", str(tmp_path / "export.jsonl")])
    assert rc == 0
    rows = [json.loads(line) for line in (tmp_path / "export.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:  # loadable by the projector consumer without manual JSON editing
        assert set(row) >= {"record_id", "fingerprint", "disposition", "evidence",
                            "evidence_digest", "exclusion_reason"}

def test_export_requires_out_without_dry_run(tmp_path: Path) -> None:
    root, state = _seed_adjudicated(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _handle_corpus_command(["adjudicate", "export", "--index-root", str(root), "--state-dir", str(state)])
    assert excinfo.value.code == 2

def test_report_subverb_prints_coverage_and_strata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, state = _seed_adjudicated(tmp_path)
    rc = _handle_corpus_command(["adjudicate", "report", "--index-root", str(root), "--state-dir", str(state)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "outcome-bearing" in out and "silver/task-only" in out
    assert "inter-rater" in out.lower()
    assert "adjudicated 1 / 1" in out
    assert "80% gate PASS" in out

def test_conflict_review_lists_disagreeing_raters_oldest_first(tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root, state = _seed_with_conflict(tmp_path)
    rc = _handle_corpus_command(
        ["adjudicate", "report", "--index-root", str(root), "--state-dir", str(state), "--conflicts"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.index("old-conflict") < out.index("new-conflict")  # oldest-first ordering

def test_adjudicate_unknown_subverb_exits_2() -> None:
    with pytest.raises(SystemExit):
        _handle_corpus_command(["adjudicate", "bogus"])

def test_adjudicate_bare_invocation_exits_2() -> None:
    with pytest.raises(SystemExit):
        _handle_corpus_command(["adjudicate"])

def test_adjudicate_build_preserves_observations_and_is_idempotent(tmp_path: Path) -> None:
    write_sessions_index(tmp_path)
    for _ in range(2):
        rc = _handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                         "--state-dir", str(tmp_path / "adj")])
        assert rc == 0
    queue = json.loads((tmp_path / "adj" / "queue.json").read_text())
    assert len(queue) == 2


_PIN_ARGS = ["--curation-id", "cur-1", "--sanitized-hub-commit", "a" * 40, "--source-hub-commit", "b" * 40,
    "--evidence-observed-at", "2026-01-01T00:00:00+00:00", "--as-of", "2026-02-01T00:00:00+00:00",
]


def _cli_index(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    sessions = [{"session_id": "s1", "trajectory_id": "t", "segment_id": "g",
        "resolutions": [{
            "fingerprint": "fp", "disposition": "unanswered", "evidence": [{"reply_id": 1, "body_sha256": "x"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
        }],
    }]
    write_sessions_jsonl(root, sessions)
    return root

def test_cli_materialize_writes_sessions_and_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    code = handle_adjudicate(["materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(out), "--archive-index-digest", "c" * 64, *_PIN_ARGS,
    ])
    assert code == 0
    assert (out / "sessions.jsonl").is_file()
    manifest = json.loads((out / "preview-manifest.json").read_text())
    assert manifest["curation_id"] == "cur-1"
    printed = capsys.readouterr().out
    assert "snapshot" in printed and "record" in printed  # S2 operator summary

def test_cli_materialize_matches_run_materialize_output(tmp_path: Path) -> None:
    out_cli = tmp_path / "out-cli"
    code = handle_adjudicate(["materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(out_cli), "--archive-index-digest", "c" * 64, *_PIN_ARGS,
    ])
    assert code == 0
    out_direct = tmp_path / "out-direct"
    run_materialize(_cli_index(tmp_path / "direct"), out_direct, pin={
        "curation_id": "cur-1", "sanitized_hub_commit": "a" * 40,
        "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
        "evidence_observed_at": "2026-01-01T00:00:00+00:00", "as_of": "2026-02-01T00:00:00+00:00",
        "labeler_version": ADJUDICATION_LABELER_VERSION, "rubric_version": RUBRIC_SCHEMA_VERSION,
        "classifier_version": REPLY_CLASSIFIER_VERSION,
    })
    assert (out_cli / "sessions.jsonl").read_bytes() == (out_direct / "sessions.jsonl").read_bytes()
    assert (out_cli / "preview-manifest.json").read_bytes() == (out_direct / "preview-manifest.json").read_bytes()

def test_cli_materialize_missing_index_exits_1(tmp_path: Path) -> None:
    assert handle_adjudicate(["materialize", "--index-root", str(tmp_path / "nope"),
        "--out-dir", str(tmp_path / "out"), "--archive-index-digest", "c" * 64, *_PIN_ARGS,
    ]) == 1

def test_cli_materialize_missing_pin_component_exits_1(tmp_path: Path) -> None:
    assert handle_adjudicate(["materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(tmp_path / "out"), "--archive-index-digest", "c" * 64,
        "--curation-id", "cur-1",  # missing the other pin flags
    ]) == 1

def test_cli_materialize_without_as_of_is_unpinned_edge(tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = handle_adjudicate(["materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(out), "--archive-index-digest", "c" * 64, "--curation-id", "cur-1",
        "--sanitized-hub-commit", "a" * 40, "--source-hub-commit", "b" * 40,
        "--evidence-observed-at", "2026-01-01T00:00:00+00:00",
    ])
    assert code == 0
    manifest = json.loads((out / "preview-manifest.json").read_text())
    assert manifest["as_of"] == ""
    records = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
    assert records and all(str(r.get("as_of", "missing")) == "" for r in records)

def test_cli_materialize_malformed_invocation_exits_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        handle_adjudicate(["materialize", "--index-root", "/tmp"])  # missing required pin flags
    assert excinfo.value.code == 2

def test_cli_publish_state_missing_state_file_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeHub:
        @property
        def repo_private(self) -> bool:
            return True
    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: _FakeHub())
    manifest = tmp_path / "preview-manifest.json"
    manifest.write_text(json.dumps({"curation_id": "cur-1", "snapshot_id": "e" * 64}), encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()  # queue.json / observations.jsonl / preview-ledger.json absent

    assert handle_adjudicate(["publish-state", "--state-dir", str(state_dir), "--manifest", str(manifest)]) == 1
    captured = capsys.readouterr()
    assert "publish-state failed" in captured.out + captured.err
    assert "queue.json" in captured.out + captured.err
    assert "required regular file is missing" in captured.out + captured.err

def test_cli_publish_state_checkpoint_reports_batch_and_actual_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _state, _manifest, hub = _publish_checkpoint(tmp_path, monkeypatch)
    revision = hub.repo_info("main").sha
    pointer_path = "annotations/cur-1/checkpoints/batch-latest.json"
    pointer = json.loads(hub.download_file(pointer_path, revision))
    rendered = _console_text(capsys)
    assert pointer["batch_id"] in rendered
    assert revision in rendered

def test_cli_resume_state_bootstraps_from_curation_without_local_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, hub = _publish_checkpoint(tmp_path, monkeypatch)
    capsys.readouterr()
    revision = hub.repo_info("main").sha
    destination = tmp_path / "restored"

    assert handle_adjudicate([
        "resume-state", "--curation-id", "cur-1", "--destination", str(destination), "--hub-repo", hub.repo_id,
    ]) == 0

    assert (destination / "preview-manifest.json").read_bytes() == manifest.read_bytes()
    assert revision in _console_text(capsys)

def test_cli_resume_state_manifest_compatibility_enforces_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, hub = _publish_checkpoint(tmp_path, monkeypatch)

    assert handle_adjudicate(["resume-state", "--manifest", str(manifest), "--destination", str(tmp_path / "restored"),
        "--hub-repo", hub.repo_id,
    ]) == 0
    mismatched = tmp_path / "mismatched-manifest.json"
    mismatched.write_text(json.dumps({"curation_id": "cur-1", "snapshot_id": "f" * 64}) + "\n", encoding="utf-8",)
    assert handle_adjudicate([
        "resume-state", "--manifest", str(mismatched), "--destination", str(tmp_path / "not-restored"),
        "--hub-repo", hub.repo_id,
    ]) == 1
    assert not (tmp_path / "not-restored").exists()

@pytest.mark.parametrize("remote_state", ["missing", "corrupt", "public"])
def test_cli_resume_state_missing_unavailable_or_corrupt_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_state: str,
) -> None:
    hub = _wired_hub(monkeypatch, repo_id="org/annotations", private=remote_state != "public")
    if remote_state == "corrupt":
        hub.seed_remote_files({"annotations/cur-1/checkpoints/batch-latest.json": b"not JSON"})
    destination = tmp_path / "restored"

    assert handle_adjudicate([
        "resume-state", "--curation-id", "cur-1", "--destination", str(destination), "--hub-repo", hub.repo_id,
    ]) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".restored.*"))

@pytest.mark.parametrize("destination_kind", ["file", "directory", "symlink"])
def test_cli_resume_state_rejects_existing_destination_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str,
) -> None:
    state, _manifest, hub = _publish_checkpoint(tmp_path, monkeypatch)
    hub.downloaded_revision_log.clear()
    destination = tmp_path / "restored"
    if destination_kind == "file":
        destination.write_text("occupied", encoding="utf-8")
    elif destination_kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(state, target_is_directory=True)

    assert handle_adjudicate([
        "resume-state", "--curation-id", "cur-1", "--destination", str(destination), "--hub-repo", hub.repo_id,
    ]) == 1
    assert hub.downloaded_revision_log == []
    assert not list(tmp_path.glob(".restored.*"))


def _download_final_argv(curation_id: str, snapshot_id: str, revision: str, destination: Path, hub_repo: str
) -> list[str]:
    return ["download-final", "--curation-id", curation_id, "--snapshot-id", snapshot_id, "--revision", revision,
        "--destination", str(destination), "--hub-repo", hub_repo,
    ]

def test_cli_download_final_installs_exact_success_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:

    hub = _wired_hub(monkeypatch)
    bundle, curation_id = _final_bundle(tmp_path)
    published = publish_final_annotation_bundle(
        hub, FinalAnnotationBundle.read(bundle),
    )
    destination = tmp_path / "downloaded"

    assert handle_adjudicate(_download_final_argv(
        curation_id, published["final_snapshot_id"], published["hub_commit_sha"], destination, hub.repo_id,
    )) == 0

    assert sorted(path.name for path in destination.iterdir()) == published["files"]
    assert (destination / "_SUCCESS").is_file()
    rendered = _console_text(capsys)
    assert published["final_snapshot_id"] in rendered
    assert published["hub_commit_sha"] in rendered

@pytest.mark.parametrize("destination_kind", ["file", "directory", "symlink"])
def test_cli_download_final_rejects_existing_destination_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str,
) -> None:

    hub = _wired_hub(monkeypatch)
    bundle, curation_id = _final_bundle(tmp_path)
    published = publish_final_annotation_bundle(
        hub, FinalAnnotationBundle.read(bundle),
    )
    hub.downloaded_revision_log.clear()
    destination = tmp_path / "downloaded"
    if destination_kind == "file":
        destination.write_text("occupied", encoding="utf-8")
    elif destination_kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(bundle, target_is_directory=True)

    assert handle_adjudicate(_download_final_argv(
        curation_id, published["final_snapshot_id"], published["hub_commit_sha"], destination, hub.repo_id,
    )) == 1
    assert hub.downloaded_revision_log == []
    assert not list(tmp_path.glob(".downloaded.*"))

def test_cli_download_final_hub_failure_exits_1_without_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = _wired_hub(monkeypatch)
    destination = tmp_path / "downloaded"

    assert handle_adjudicate(_download_final_argv("cur-1", "e" * 64, "f" * 40, destination, hub.repo_id,)) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".downloaded.*"))


# ---- final publish verb (issue #1078, task 6 / M4-M6) ----

from daydream.training.adjudication.canonical import run_canonical_harvest  # noqa: E402
from tests.fixtures.training.build_hub_snapshot import build_snapshot  # noqa: E402
from tests.test_training_adjudication_final_bundle import seed_final_bundle_state  # noqa: E402


@pytest.mark.parametrize("legacy_stage", [False, True])
def test_publish_final_dry_run_validates_and_publishes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch, legacy_stage: bool) -> None:

    # Replace only the external Hub; retain the real CLI dry run and OID state.
    hub = _wired_hub(monkeypatch)
    index_root, mat, archive_dir, pin = seed_final_bundle_state(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    # Coverage requires a matching human observation.
    (state / "observations.jsonl").write_text(json.dumps(accepted_observation()) + "\n", encoding="utf-8")
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=state / "observations.jsonl")
    scratch = mat / "final-bundle" / ".publish-stage"
    scratch_bytes = b"hf_legacyPrivateScratchMustNeverBeUploaded\n"
    if legacy_stage:
        scratch.mkdir(parents=True)
        (scratch / "annotations.jsonl").write_bytes(scratch_bytes)
    rc = _publish_final(index_root, mat, archive_dir, state, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "annotations.jsonl" in out and "record" in out.lower()

    final_id, _digests = final_snapshot_id(mat / "final-bundle")
    assert final_id in "".join(out.split()).replace("║", "")
    assert not any(k.startswith("annotations/") and "/final/" in k for k in hub.files)

    # Publish through the same Hub fake as a positive control for the dry-run no-publication
    # assertion.
    assert _publish_final(index_root, mat, archive_dir, state) == 0
    published_output = _console_text(capsys)
    assert final_id in published_output
    assert hub.repo_info("main").sha in published_output
    assert any(k.startswith("annotations/") and "/final/" in k for k in hub.files)
    if legacy_stage:
        assert (scratch / "annotations.jsonl").read_bytes() == scratch_bytes
        assert not any(".publish-stage" in name for name in hub.files)
        assert not any(scratch_bytes in data for data in hub.files.values())

def test_publish_final_refuses_when_admission_gate_not_met(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = build_snapshot()
    hub.commit_revision("a" * 40)
    _install_annotation_hub(monkeypatch, hub)
    index_root, mat, archive_dir, pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    rc = _publish_final(index_root, mat, archive_dir, tmp_path / "state")
    assert rc == 1
    assert not any(k.startswith("annotations/") and "/final/" in k for k in hub.files)

def test_publish_final_missing_artifact_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_root, mat, archive_dir, pin = seed_final_bundle_state(tmp_path)
    ann = mat / "annotations.jsonl"
    if ann.exists():
        ann.unlink()  # the canonical-harvest output when present; either way the artifact is missing
    rc = _publish_final(index_root, mat, archive_dir, tmp_path / "state", dry_run=True)
    assert rc == 1
    captured = capsys.readouterr()
    # Remove Rich borders and wrapping before comparing the long path.
    flattened = "".join(captured.out.split()).replace("║", "")
    assert "annotations.jsonl" in flattened + captured.err


