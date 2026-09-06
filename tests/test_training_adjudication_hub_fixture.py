"""Revision-aware external Hub fixture contracts; no live service access."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.fixtures.training.build_hub_snapshot import AnnotationsHub


def _mapping(root: Path, payloads: dict[str, bytes]) -> dict[str | Path, Path]:
    root.mkdir()
    mapping: dict[str | Path, Path] = {}
    for index, (remote, data) in enumerate(payloads.items()):
        local = root / str(index)
        local.write_bytes(data)
        mapping[remote] = local
    return mapping


def test_annotation_hub_initial_revision_is_content_addressed_and_branch_pinned() -> None:
    first = AnnotationsHub(curation_id="cur", snapshot_id="snap", files={"a": b"one"})
    equal = AnnotationsHub(curation_id="cur", snapshot_id="snap", files={"a": b"one"})
    changed = AnnotationsHub(curation_id="cur", snapshot_id="snap", files={"a": b"two"})
    head = first.repo_info().sha
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert equal.repo_info().sha == head
    assert changed.repo_info().sha != head
    assert first.repo_info("main").sha == head
    assert first.repo_info(head).private is True


def test_publication_hubs_are_separate_durable_stores_with_packaged_pins() -> None:
    from tests.fixtures.training.build_hub_snapshot import build_publication_hubs

    hubs = build_publication_hubs()
    assert hubs.source is not hubs.annotations
    assert hubs.source.repo_id != hubs.annotations.repo_id
    assert hubs.source.repo_info(hubs.source_revision).sha == hubs.source_revision
    assert hubs.annotations.list_repo_files(hubs.annotations.repo_info("main").sha) == []
    assert hubs.policy_path.is_file()


def test_annotation_hub_atomic_commit_preserves_pinned_trees_and_logs(tmp_path: Path) -> None:
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap", files={"a": b"old"})
    base = hub.repo_info("main").sha
    mapping = _mapping(tmp_path / "input", {"b": b"new", "a": b"changed"})
    revision = hub.commit_files_atomic(mapping, "batch", parent_commit=base, branch="main")
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision != base
    assert hub.repo_info("main").sha == revision
    assert hub.download_file("a", base) == b"old"
    assert hub.download_file("a", revision) == b"changed"
    assert "b" not in hub.list_repo_files(base)
    assert "b" in hub.list_repo_files(revision)
    copied = hub.revision_files(revision)
    copied["a"] = b"outside mutation"
    assert hub.download_file("a", revision) == b"changed"
    assert hub.downloaded_revision_log == [("a", base), ("a", revision), ("a", revision)]
    assert hub.listed_revision_log == [base, revision]
    assert hub.commit_order == [{"contains": ["a", "b"], "sha": revision}]
    assert hub.atomic_attempt_log[-1]["parent_commit"] == base
    assert hub.atomic_attempt_log[-1]["branch"] == "main"


def test_annotation_hub_stale_parent_has_no_tree_commit(tmp_path: Path) -> None:
    from daydream.archive.hydrate import HubConcurrentUpdateError

    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
    before = hub.repo_info("main").sha
    tree = hub.revision_files(before)
    mapping = _mapping(tmp_path / "input", {"a": b"new"})
    with pytest.raises(HubConcurrentUpdateError):
        hub.commit_files_atomic(mapping, "stale", parent_commit="f" * 40, branch="main")
    assert hub.repo_info("main").sha == before
    assert hub.revision_files(before) == tree
    assert hub.files == tree
    assert hub.commit_order == []


@pytest.mark.parametrize("stage", ["batch", "data", "success"])
def test_annotation_hub_can_inject_one_rival_commit_per_publication_stage(
    tmp_path: Path, stage: str,
) -> None:
    from daydream.archive.hydrate import HubConcurrentUpdateError

    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
    base = hub.repo_info("main").sha
    path = {
        "batch": "annotations/cur/checkpoints/batch-latest.json",
        "data": "annotations/cur/final/annotations.jsonl",
        "success": "annotations/cur/final/_SUCCESS",
    }[stage]
    mapping = _mapping(tmp_path / "input", {path: b"candidate"})
    hub.queue_concurrent_commit(stage, {"rival": b"durable remote state"})
    with pytest.raises(HubConcurrentUpdateError):
        hub.commit_files_atomic(mapping, "candidate", parent_commit=base, branch="main")
    rival = hub.repo_info("main").sha
    assert rival != base
    assert len(hub.commit_order) == 1
    assert hub.commit_order[0]["contains"] == ["rival"]
    assert path not in hub.revision_files(rival)
    final = hub.commit_files_atomic(mapping, "candidate", parent_commit=rival, branch="main")
    assert hub.download_file(path, final) == b"candidate"
    assert hub.download_file("rival", final) == b"durable remote state"
    assert len(hub.commit_order) == 2


def test_annotation_hub_commit_identity_binds_parent_message_paths_and_bytes(tmp_path: Path) -> None:
    def commit(label: str, *, content: bytes = b"one", message: str = "same", path: str = "a") -> str:
        hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
        parent = hub.repo_info("main").sha
        return hub.commit_files_atomic(
            _mapping(tmp_path / label, {path: content}), message,
            parent_commit=parent, branch="main",
        )

    baseline = commit("baseline")
    assert baseline == commit("equal")
    assert baseline != commit("bytes", content=b"two")
    assert baseline != commit("message", message="different")
    assert baseline != commit("path", path="b")
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
    mapping = _mapping(tmp_path / "parent", {"a": b"one"})
    first = hub.commit_files_atomic(
        mapping, "same", parent_commit=hub.repo_info("main").sha, branch="main",
    )
    second = hub.commit_files_atomic(mapping, "same", parent_commit=first, branch="main")
    assert first == baseline
    assert second != first


def test_annotation_hub_rejects_legacy_in_place_bundle_mutation() -> None:
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
    revision = hub.repo_info("main").sha
    before = hub.revision_files(revision)

    with pytest.raises(NotImplementedError, match="immutable annotation revisions"):
        hub.mutate_bundle(revision, "rogue", b"changed")

    assert hub.repo_info("main").sha == revision
    assert hub.revision_files(revision) == before
    assert hub.commit_order == []


def test_annotation_hub_supported_mutation_creates_one_immutable_revision() -> None:
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
    revision = hub.repo_info("main").sha
    before = hub.revision_files(revision)

    hub.mutate_annotation_file("review.json", b"changed")

    changed_revision = hub.repo_info("main").sha
    path = "annotations/cur/snap/review.json"
    assert changed_revision != revision
    assert hub.revision_files(revision) == before
    assert hub.download_file(path, changed_revision) == b"changed"
    assert hub.commit_order == [{"contains": [path], "sha": changed_revision}]


def test_annotation_hub_commit_revision_cannot_rewrite_an_existing_pin() -> None:
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap", files={"a": b"old"})
    revision = hub.repo_info("main").sha
    before = hub.revision_files(revision)

    hub.commit_revision(revision, ref="main")
    hub.files["a"] = b"changed"
    with pytest.raises(ValueError, match="cannot replace immutable annotation revision"):
        hub.commit_revision(revision, ref="main")

    assert hub.repo_info("main").sha == revision
    assert hub.revision_files(revision) == before


@pytest.mark.parametrize("revision", ["short", "A" * 40])
def test_annotation_hub_rejects_noncanonical_revision_ids(revision: str) -> None:
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap")
    head = hub.repo_info("main").sha

    with pytest.raises(ValueError, match="lowercase 40-hex"):
        hub.commit_revision(revision)

    assert hub.repo_info("main").sha == head


def test_annotation_hub_legacy_upload_uses_atomic_main_commits(tmp_path: Path) -> None:
    hub = AnnotationsHub(curation_id="cur", snapshot_id="snap", files={"a": b"old"})
    base = hub.repo_info("main").sha
    base_tree = hub.revision_files(base)

    hub.upload_files(_mapping(tmp_path / "first", {"a": b"one"}), "same")
    first = hub.repo_info("main").sha
    first_tree = hub.revision_files(first)
    assert first != base
    assert hub.atomic_attempt_log[-1]["parent_commit"] == base
    assert hub.atomic_attempt_log[-1]["branch"] == "main"

    hub.upload_files(_mapping(tmp_path / "second", {"a": b"two"}), "same")
    second = hub.repo_info("main").sha
    assert second != first
    assert hub.revision_files(base) == base_tree
    assert hub.revision_files(first) == first_tree
    assert hub.download_file("a", first) == b"one"
    assert hub.download_file("a", second) == b"two"
    assert [entry["sha"] for entry in hub.commit_order] == [first, second]
