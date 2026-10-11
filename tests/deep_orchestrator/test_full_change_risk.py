"""Full-change routing and bounded prompt populations through the public runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.diff import DeepDiffBoundInfo, bound_deep_diff
from daydream.deep.routing_record import read_routing_record
from daydream.runner import run
from tests.harness.git_helpers import commit, git, init_repo, seed_feature_branch, write_and_stage
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import StubBackend, review_stage_state
from tests.test_deep_orchestrator import MakeConfig


async def test_dropped_sensitive_block_routes_from_full_captured_change(
    tmp_path: Path, make_config: MakeConfig, install_backend: Callable[[object], object],
) -> None:
    repo = tmp_path / "full_change"
    seed_feature_branch(repo,
        base={"aaa.py": "VALUE = 0\n", "zzz.py": "VALUE = 0\n"},
        feature={"aaa.py": "VALUE = 1\n", "zzz.py": "password = 'test-only'\n" + "# ordinary filler é\n" * 3000},
    )
    captured_patch = git(repo, "diff", "--no-ext-diff", "--no-textconv", "main", "HEAD") + "\n"
    stub = StubBackend(repo)
    install_backend(stub)
    assert await run(make_config(repo, output_mode="review", latency_profile="balanced",
                                 review_profile=independent_alternatives_profile())) == 0

    deep = repo / ".daydream" / "deep"
    record = read_routing_record(deep)
    assert record["wonder"]["effort"] == "high"
    assert record["wonder"]["outcome"] == "run"
    assert record["risk"]["signals"]["security_surface"] is True
    assert record["risk"]["signals"]["changed_files"] == 2
    assert record["risk"]["signals"]["diff_bytes"] == len(captured_patch.encode("utf-8"))
    assert record["risk"]["signals"]["diff_lines"] == len(captured_patch.splitlines())
    assert record["diff_population"]["full_patch"] == {
        "bytes": len(captured_patch.encode("utf-8")), "lines": len(captured_patch.splitlines()), "blocks": 2,
    }
    population = record["diff_population"]
    assert population["truncated"] is True and population["dropped_blocks"] == 1
    assert population["retained_patch"]["blocks"] == 1
    assert population["retained_patch"]["bytes"] < population["bounded_prompt_diff"]["bytes"] < 24 * 1024
    assert (repo / ".daydream" / "diff.patch").read_text(encoding="utf-8") == captured_patch
    snapshot_key = hashlib.sha256(captured_patch.encode("utf-8")).hexdigest()
    assert population["diff_key"] == (deep / "diff-key").read_text() == snapshot_key
    coverage = json.loads((deep / "review-coverage.json").read_text())
    assert coverage["analyzed_revision"]["diff_key"] == snapshot_key

    ttt_prompts = [call["prompt"] for call in stub.calls
                   if "understand the intent" in call["prompt"].lower()
                   or "evaluate the implementation" in call["prompt"].lower()]
    assert len(ttt_prompts) == 2
    assert all("password =" not in prompt and "ordinary filler" not in prompt for prompt in ttt_prompts)
    assert all("diff.patch" in prompt for prompt in ttt_prompts)
    stages = [stage for call in stub.calls if (stage := review_stage_state(call["prompt"])) is not None]
    assert stages
    assert {name for stage in stages for name in stage["assigned_files"]} == {"aaa.py", "zzz.py"}
    for stage in stages:
        bundle = Path(stage["supporting_bundle"]["path"])
        artifact_root = next(parent for parent in bundle.parents if parent.name == ".daydream")
        published_bundle = repo / ".daydream" / bundle.relative_to(artifact_root)
        assert len(published_bundle.read_bytes()) <= 24 * 1024


@pytest.mark.parametrize("shape", ["untruncated-utf8", "oversized-first", "empty"])
async def test_population_metrics_preserve_complete_snapshot_at_patch_edges(
    tmp_path: Path, make_config: MakeConfig, install_backend: Callable[[object], object], shape: str,
) -> None:
    repo = tmp_path / "patch_edges"
    if shape == "empty":
        init_repo(repo)
        write_and_stage(repo, "aaa.py", "VALUE = 0\n")
        commit(repo, "base")
        git(repo, "checkout", "-b", "feature")
        expected_patch = ""
    else:
        seed_feature_branch(repo,
            base={"aaa.py": "VALUE = 0\n", "zzz.py": "VALUE = 0\n"},
            feature={"aaa.py": "VALUE = 'é雪'\n" + ("# ordinary filler é\n" * 1000
                                                     if shape == "oversized-first" else ""),
                     "zzz.py": "VALUE = 1\n"},
        )
        expected_patch = git(repo, "diff", "--no-ext-diff", "--no-textconv", "main", "HEAD") + "\n"
    stub = StubBackend(repo)
    install_backend(stub)
    assert await run(make_config(repo, output_mode="review", latency_profile="balanced",
                                 review_profile=independent_alternatives_profile())) == 0
    deep = repo / ".daydream" / "deep"
    record = read_routing_record(deep)
    population = record["diff_population"]
    patch_bytes = len(expected_patch.encode("utf-8"))
    block_count = 0 if shape == "empty" else 2
    assert population["full_patch"] == {"bytes": patch_bytes, "lines": len(expected_patch.splitlines()),
                                          "blocks": block_count}
    assert record["risk"]["signals"]["diff_bytes"] == patch_bytes
    assert record["risk"]["signals"]["changed_files"] == block_count
    assert (repo / ".daydream" / "diff.patch").read_text(encoding="utf-8") == expected_patch
    key = hashlib.sha256(expected_patch.encode("utf-8")).hexdigest()
    assert population["diff_key"] == (deep / "diff-key").read_text() == key
    coverage = json.loads((deep / "review-coverage.json").read_text())
    assert coverage["analyzed_revision"]["diff_key"] == key
    if shape == "oversized-first":
        first_block = git(repo, "diff", "--no-ext-diff", "--no-textconv", "main", "HEAD", "--", "aaa.py") + "\n"
        retained_bytes = len(first_block.encode("utf-8"))
        assert population["retained_patch"] == {"bytes": retained_bytes, "blocks": 1}
        assert retained_bytes > 12_288
        assert population["bounded_prompt_diff"]["bytes"] > retained_bytes
        assert population["truncated"] is True and population["dropped_blocks"] == 1
        ttt = [call["prompt"] for call in stub.calls if "understand the intent" in call["prompt"].lower()
               or "evaluate the implementation" in call["prompt"].lower()]
        assert len(ttt) == 2 and all("ordinary filler" not in prompt for prompt in ttt)
    else:
        assert population["retained_patch"] == {"bytes": patch_bytes, "blocks": block_count}
        assert population["bounded_prompt_diff"] == {"bytes": patch_bytes}
        assert population["truncated"] is False and population["dropped_blocks"] == 0
        if shape == "empty":
            assert not stub.calls
            assert coverage["phase_outcomes"][0]["phase"] == "no_diff"
            assert coverage["phase_outcomes"][0]["status"] == "complete"
        else:
            assert patch_bytes > len(expected_patch)
            intent = next(call["prompt"] for call in stub.calls if "understand the intent" in call["prompt"].lower())
            assert "VALUE = 'é雪'" in intent


@pytest.mark.parametrize("evidence", ["stale-fields", "missing", "corrupt"])
async def test_same_snapshot_resume_replaces_population_evidence_without_requiring_the_record(
    tmp_path: Path, make_config: MakeConfig, install_backend: Callable[[object], object], evidence: str,
) -> None:
    repo = tmp_path / "resume"
    seed_feature_branch(repo, base={"aaa.py": "VALUE = 0\n"}, feature={"aaa.py": "VALUE = 'é'\n"})
    stub = StubBackend(repo)
    install_backend(stub)
    config = make_config(repo, output_mode="review", review_profile=independent_alternatives_profile())
    assert await run(config) == 0
    deep = repo / ".daydream" / "deep"
    original = read_routing_record(deep)
    record_path = deep / "latency-routing.json"
    if evidence == "stale-fields":
        stale = json.loads(record_path.read_text())
        stale["diff_population"]["obsolete"] = True
        stale["diff_population"]["full_patch"]["obsolete"] = 999
        stale["risk"]["obsolete"] = "previous producer"
        stale["risk"]["signals"]["obsolete"] = True
        record_path.write_text(json.dumps(stale))
    elif evidence == "missing":
        record_path.unlink()
    else:
        record_path.write_text("{not valid JSON")
    stub.calls.clear()
    resumed_config = make_config(repo, output_mode="review", start_at="merge",
                                 review_profile=independent_alternatives_profile())
    assert await run(resumed_config) == 0
    resumed = read_routing_record(deep)
    assert resumed["diff_population"] == original["diff_population"]
    assert resumed["risk"] == original["risk"]
    assert not stub.calls
    coverage = json.loads((deep / "review-coverage.json").read_text())
    assert coverage["analyzed_revision"]["diff_key"] == resumed["diff_population"]["diff_key"]
    if evidence == "stale-fields":
        # A committed new snapshot must still fail the freshness gate, even with
        # otherwise valid routing evidence. Rejected resumes do not rewrite it.
        record_before_rejection = record_path.read_bytes()
        (repo / "aaa.py").write_text("VALUE = 'another snapshot'\n")
        git(repo, "add", "aaa.py")
        commit(repo, "new snapshot")
        assert await run(resumed_config) == 1
        assert not stub.calls
        assert record_path.read_bytes() == record_before_rejection
        assert (deep / "diff-key").read_text() == original["diff_population"]["diff_key"]


async def test_prompt_capacity_and_marker_changes_cannot_change_full_snapshot_risk(
    tmp_path: Path, make_config: MakeConfig, install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplementary metamorphic check; primary acceptance has no internal patches.

    Instrument only the real bounding operation to vary its transport policy;
    every phase, filesystem capture, and route calculation still executes.
    """
    repo = tmp_path / "prompt_policies"
    seed_feature_branch(repo, base={"aaa.py": "VALUE = 0\n", "zzz.py": "VALUE = 0\n"},
                        feature={"aaa.py": "VALUE = 1\n", "zzz.py": "password = 'test-only'\n" + "# filler\n" * 2000})
    stub = StubBackend(repo)
    install_backend(stub)
    observations: list[dict[str, Any]] = []
    for capacity, replacement_marker in ((128, None), (128, "# different marker é\n"), (100_000, None)):
        def with_policy(diff: str, budget: int = capacity) -> tuple[str, DeepDiffBoundInfo]:
            bounded, info = bound_deep_diff(diff, budget)
            if replacement_marker is not None and info.marker is not None:
                bounded = replacement_marker + bounded.removeprefix(info.marker)
                info.marker = replacement_marker
            return bounded, info

        monkeypatch.setattr("daydream.deep.orchestrator.bound_deep_diff", with_policy)
        assert await run(make_config(repo, output_mode="review", latency_profile="balanced",
                                     review_profile=independent_alternatives_profile())) == 0
        observations.append(read_routing_record(repo / ".daydream" / "deep"))
    assert all(record["risk"] == observations[0]["risk"] for record in observations)
    assert all(record["wonder"]["effort"] == "high" for record in observations)
    assert all(record["diff_population"]["diff_key"] == observations[0]["diff_population"]["diff_key"]
               for record in observations)
    populations = [record["diff_population"] for record in observations]
    assert [item["truncated"] for item in populations] == [True, True, False]
    assert populations[0]["retained_patch"] == populations[1]["retained_patch"]
    assert populations[0]["bounded_prompt_diff"] != populations[1]["bounded_prompt_diff"]
    assert populations[2]["retained_patch"]["blocks"] == 2
