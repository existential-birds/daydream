"""Exercise corpus dispatch through cli.main, including exit codes and real build output.

Bare corpus prints help and exits 2. Removed top-level data verbs fall through
to review and fail target validation; only handler/backend boundaries are mocked.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.commands import corpus as cli_corpus
from daydream.training.adjudication import cli as adjudication_cli
from daydream.training.adjudication.publish import publish_final_annotation_bundle
from tests.fixtures.training.build_hub_snapshot import AnnotationsHub
from tests.harness.adjudication import write_checkpoint_inputs
from tests.harness.scripts import cli_main
from tests.test_corpus_projection import _write_annotations_snapshot, _write_bundle
from tests.test_training_adjudication_publish import _final_bundle


@pytest.mark.parametrize("summary, expected", [
    ({"considered": 3, "annotated": 1, "skipped": 0, "errors": 0, "aborted": 1}, 1),
    ({"considered": 3, "annotated": 1, "skipped": 0, "errors": 2, "aborted": 0}, 1),
    # Unresolved findings in the data are not process failure (spec KD).
    ({"considered": 3, "annotated": 1, "skipped": 2, "errors": 0, "aborted": 0}, 0),
])
def test_corpus_harvest_exit_code_maps_summary(monkeypatch: pytest.MonkeyPatch, summary: dict[str, Any], expected: int
) -> None:
    async def _fake_run_harvest(_pass: Any, **_: Any) -> dict[str, Any]:
        return summary
    monkeypatch.setattr("daydream.training.harvest.HarvestPass.run", _fake_run_harvest)
    assert cli_main(["corpus", "harvest", "--dry-run"]) == expected

def test_corpus_harvest_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}
    async def _fake_run_harvest(_pass: Any, **_: Any) -> dict[str, Any]:
        called["hit"] = True
        return {"errors": 0, "annotated": 0, "skipped": 0, "total": 0}
    monkeypatch.setattr("daydream.training.harvest.HarvestPass.run", _fake_run_harvest)
    assert cli_main(["corpus", "harvest", "--dry-run"]) == 0
    assert called["hit"]

def test_corpus_label_route(monkeypatch: pytest.MonkeyPatch) -> None:
    label_called = {}
    def _fake_label(argv: list[str]) -> int:
        label_called["argv"] = argv
        return 0

    # The dispatcher stores function references; patch its mapping rather than the source module.
    monkeypatch.setitem(cli_corpus._CORPUS_SUBVERBS, "label", _fake_label)
    assert cli_main(["corpus", "label", "sess-0001", "--outcome", "accepted"]) == 0
    assert label_called["argv"] == ["sess-0001", "--outcome", "accepted"]

def test_bare_corpus_prints_help_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["corpus"]) == 2
    captured = capsys.readouterr()
    # Assert help tokens independently of 80-column wrapping.
    assert "calibrate-reward" in captured.out
    assert "build" in captured.out
    assert "hydrate-hub" in captured.out
    assert "harvest" in captured.out
    assert "label" in captured.out

def test_adjudicate_publication_commands_run_through_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    state, manifest = write_checkpoint_inputs(tmp_path, curation_id="cur-main")
    hub = AnnotationsHub(repo_id="org/private-annotations")
    monkeypatch.setattr(adjudication_cli, "_make_client", lambda _repo_id: hub)

    assert cli_main(["corpus", "adjudicate", "publish-state", "--state-dir", str(state), "--manifest", str(manifest),
        "--hub-repo", hub.repo_id,
    ]) == 0
    revision = hub.repo_info("main").sha
    assert "annotations/cur-main/checkpoints/batch-latest.json" in hub.list_repo_files(revision)

    destination = tmp_path / "restored"
    assert cli_main([
        "corpus", "adjudicate", "resume-state", "--curation-id", "cur-main", "--destination", str(destination),
        "--hub-repo", hub.repo_id,
    ]) == 0
    assert (destination / "preview-manifest.json").read_bytes() == manifest.read_bytes()

    bundle, curation_id = _final_bundle(tmp_path)
    published = publish_final_annotation_bundle(hub, bundle)
    final_destination = tmp_path / "downloaded-final"
    assert cli_main(["corpus", "adjudicate", "download-final", "--curation-id", curation_id,
        "--snapshot-id", published["final_snapshot_id"], "--revision", published["hub_commit_sha"],
        "--destination", str(final_destination), "--hub-repo", hub.repo_id,
    ]) == 0
    assert (final_destination / "_SUCCESS").is_file()


# Real projector bundles exercise pinned license policy, exact-slug copyleft opt-ins, and refusal of
# URL-shaped identities.


def _run_build_v2(tmp_path: Path, capsys: pytest.CaptureFixture[str], extra_args: list[str]) -> tuple[int, str]:
    """Run corpus build through cli.main over fixture bundles; return exit code and captured output."""
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted"])
    out_dir = tmp_path / "corpus-out"
    rc = cli_main(["corpus", "build", "--bundle-root", str(bundle_dir), "--annotation-bundle-root", str(snap.parent),
        "--out", str(out_dir / "corpus.jsonl"), *extra_args,
    ])
    captured = capsys.readouterr()
    return rc, captured.out + captured.err

def test_build_v2_accepts_license_policy_and_repeatable_opt_in(tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = tmp_path / "license-policy.json"
    policy.write_text(json.dumps({"policy_version": "1", "spdx_decisions": {"MIT": "accepted"}}))
    rc, _out = _run_build_v2(tmp_path, capsys, [
        "--license-policy", str(policy), "--allow-copyleft", "a/b", "--allow-copyleft", "c/d",
    ])
    assert rc == 0
    lineage = json.loads((tmp_path / "corpus-out" / "lineage.json").read_text())
    assert lineage["license_policy"]["policy_version"] == "1"
    assert lineage["copyleft_opt_ins"] == ["a/b", "c/d"]

def test_build_v2_requires_license_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out = _run_build_v2(tmp_path, capsys, [])
    assert rc == 1
    assert "license-policy" in out

def test_build_v2_refuses_unknown_policy_version(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy_path = tmp_path / "bad-policy.json"
    policy_path.write_text(json.dumps({"policy_version": "", "spdx_decisions": {}}))
    rc, out = _run_build_v2(tmp_path, capsys, ["--license-policy", str(policy_path)])
    assert rc == 1
    assert "policy_version" in out
    assert not (tmp_path / "corpus-out").exists()

def test_build_v2_refuses_raw_authenticated_url_as_identity(tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out = _run_build_v2(tmp_path, capsys, ["--repo-slug", "https://user:token@github.com/owner/repo"])
    assert rc == 1
    assert "Unsupported --repo-slug" in out

def test_bare_harvest_is_unknown_verb_treated_as_review_target(capsys: pytest.CaptureFixture[str],) -> None:
    # The removed harvest verb parses the review target before reporting the unknown flag.
    assert cli_main(["harvest", "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert "--dry-run" in captured.err
