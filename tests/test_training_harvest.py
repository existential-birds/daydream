"""Tests for the harvest pass — bronze signal assembly + per-run annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.harvest import assemble_scoring_inputs, build_annotation


def _seed_deep_bronze(tmp_path: Path, *, verdict: str, grounding: float) -> Path:
    """Write a deep-run bronze bundle and return its run directory.

    Mirrors the seeding shape used across the labeler tests
    (``deep/recommendation-verdicts.json`` + ``diff.patch``); ``grounding``
    is accepted for parity with the indexed-row grounding signal supplied
    separately to :func:`build_annotation`.
    """
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        json.dumps({"verdicts": [{"issue_id": 1, "verdict": verdict}]})
    )
    (run_dir / "diff.patch").write_text(
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        "+new_line\n"
    )
    return run_dir


def _fake_gh_merged(merged_at: str):
    """Return a ``gh_api(repo, endpoint, **kw)`` responder for a merged PR.

    Reuses the labeler-test responder shape: keyed on the endpoint, the
    ``pulls/<n>`` endpoint reports merged with the given timestamp and the
    ``comments`` endpoint reports no comments (clean → ``accepted``).
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return []
        return {"merged": True, "merged_at": merged_at}

    return responder


def _unused_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """A ``gh_api`` responder the local-branch path must never call."""
    raise AssertionError(f"gh_api should not be called for a local row (endpoint={endpoint})")


def test_build_annotation_pr_row_carries_label_reward_and_merge_valid_at(tmp_path):
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s1", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir,
                           gh_api=_fake_gh_merged("2026-02-01T00:00:00+00:00"),
                           repo_clone=tmp_path, window_days=30)
    assert ann.labels == ["accepted"]
    assert ann.valid_at == "2026-02-01T00:00:00+00:00"        # PR merge time (Q2)
    assert ann.composite_reward == json.loads(ann.reward_json)["composite"]


def test_build_annotation_shallow_local_row_null_valid_at_reward_present(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()           # no deep/ → shallow
    row = {"session_id": "s2", "pr_repo": None, "pr_number": None, "branch": "feat",
           "head_sha": "h", "archive_path": str(run_dir), "grounding_rate": None,
           "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir, gh_api=_unused_gh,
                           repo_clone=tmp_path, window_days=30)
    assert ann.valid_at is None                               # collapses to observed_at on write
    rb = json.loads(ann.reward_json)
    assert rb["axes_present"]["correctness"] is False         # shallow: no verdicts


def test_assemble_reads_verdicts_and_grounding_from_bronze(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        '{"verdicts":[{"issue_id":1,"verdict":"consistent"}]}'
    )
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 0.75})
    assert inputs.verifier_verdicts == [{"issue_id": 1, "verdict": "consistent"}]
    assert inputs.grounding_rate == 0.75 and inputs.format_valid is True


def test_assemble_shallow_run_has_null_verdicts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": None})
    assert inputs.verifier_verdicts is None


def test_assemble_malformed_verdicts_flags_format_invalid(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text("{not json")
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 1.0})
    assert inputs.format_valid is False
