"""Phase 1: taskset construction from a harvested corpus, with C5 enforcement."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from conftest import (
    FIXTURE_BASE_SHA,
    FIXTURE_PR1_HEAD_SHA,
    FIXTURE_PR2_HEAD_SHA,
    FIXTURE_SLUG,
    FIXTURE_TEST_COMMAND,
    build_fixture_repo,
)

from daydream_review_v1.taskset import DaydreamReviewConfig, DaydreamReviewTaskset


def _write_corpus(root: Path, repo: str, prs: list[dict[str, object]]) -> Path:
    """Write a minimal `daydream bench harvest`-shaped corpus at *root*."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(
        json.dumps({"repo": repo, "bot": "some-bot[bot]", "n_prs_with_bot_activity": len(prs), "prs": prs}),
        encoding="utf-8",
    )
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "benchmark_data.json").write_text(json.dumps({}), encoding="utf-8")
    return root


def _pr(number: int, base_sha: str, head_sha: str) -> dict[str, object]:
    return {
        "pr_number": number,
        "title": f"PR {number}",
        "state": "closed",
        "merged": True,
        "base_ref": "main",
        "base_sha": base_sha,
        "review_commit_id": head_sha,
        "n_inline_comments": 0,
        "n_review_summaries": 0,
        "n_resolved_threads": 0,
        "threads_complete": True,
    }


def _write_manifest(path: Path, slugs: list[str]) -> Path:
    body = "\n".join(
        f'[repos."{slug}"]\n'
        f'clone_url = "https://github.com/{slug}"\n'
        f'image = "daydream-rl/{slug.split("/")[-1]}"\n'
        f'test_command = "pytest -q"\n'
        "setup_cmds = []\n"
        for slug in slugs
    )
    path.write_text(body, encoding="utf-8")
    return path


def test_fixture_repo_is_deterministic(tmp_path: Path) -> None:
    """The pinned SHAs in conftest/corpus-mini/manifest describe the real repo."""
    repo = build_fixture_repo(tmp_path / "fx")
    assert (repo.base_sha, repo.pr1_head_sha, repo.pr2_head_sha) == (
        FIXTURE_BASE_SHA,
        FIXTURE_PR1_HEAD_SHA,
        FIXTURE_PR2_HEAD_SHA,
    )
    green = subprocess.run(FIXTURE_TEST_COMMAND.split(), cwd=repo.path, capture_output=True, text=True)
    assert green.returncode == 0, green.stderr


def test_load_builds_tasks_from_fixture_corpus(corpus_mini_dir: Path, fixture_manifest_path: Path) -> None:
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
        )
    )
    tasks = list(taskset.load())
    assert len(tasks) == 2

    by_pr = {task.data.pr_number: task.data for task in tasks}
    assert set(by_pr) == {1, 2}

    pr1 = by_pr[1]
    assert pr1.repo_slug == FIXTURE_SLUG
    assert pr1.base_sha == FIXTURE_BASE_SHA
    assert pr1.head_sha == FIXTURE_PR1_HEAD_SHA
    assert pr1.base_ref == "main"
    assert pr1.test_command == FIXTURE_TEST_COMMAND
    assert pr1.clone_url == f"https://github.com/{FIXTURE_SLUG}"
    assert pr1.image == f"daydream-rl/fixture:{FIXTURE_PR1_HEAD_SHA[:12]}"
    assert pr1.name == f"{FIXTURE_SLUG}#1"
    assert pr1.prompt is not None and "#1" in pr1.prompt
    assert pr1.timeout.harness == 5400
    assert [c.path for c in pr1.golden_comments] == ["calc.py"]
    assert "ZeroDivisionError" in pr1.golden_comments[0].comment
    assert pr1.golden_comments[0].resolved is True

    pr2 = by_pr[2]
    assert pr2.base_sha == FIXTURE_PR1_HEAD_SHA
    assert pr2.head_sha == FIXTURE_PR2_HEAD_SHA
    assert pr2.image == f"daydream-rl/fixture:{FIXTURE_PR2_HEAD_SHA[:12]}"
    assert pr2.golden_comments[0].resolved is False

    assert [task.data.idx for task in tasks] == [0, 1]


def test_use_images_false_leaves_tasks_imageless(corpus_mini_dir: Path, fixture_manifest_path: Path) -> None:
    """The subprocess smoke path needs imageless tasks (verifiers env.py:189-195)."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            use_images=False,
        )
    )
    tasks = list(taskset.load())
    assert [task.data.image for task in tasks] == [None, None]
    assert all(task.data.test_command == FIXTURE_TEST_COMMAND for task in tasks)


def test_load_rejects_excluded_repo(tmp_path: Path) -> None:
    """C5 is unconditional: an excluded slug fails the load, manifest or not."""
    corpus = _write_corpus(tmp_path / "corpus", "getsentry/sentry", [_pr(7, "a" * 40, "b" * 40)])
    manifest = _write_manifest(tmp_path / "manifest.toml", [])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest)
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)
    assert "getsentry/sentry" in str(excinfo.value)


def test_load_rejects_excluded_repo_case_insensitively(tmp_path: Path) -> None:
    """GitHub slugs are case-insensitive; `GetSentry/Sentry` is the same repo."""
    corpus = _write_corpus(tmp_path / "corpus", "GetSentry/Sentry", [_pr(7, "a" * 40, "b" * 40)])
    manifest = _write_manifest(tmp_path / "manifest.toml", ["GetSentry/Sentry"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest)
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)


def test_load_rejects_corpus_without_benchmark_data(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus", "acme/widgets", [_pr(3, "a" * 40, "b" * 40)])
    (corpus / "results" / "benchmark_data.json").unlink()
    manifest = _write_manifest(tmp_path / "manifest.toml", ["acme/widgets"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest)
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "benchmark_data.json" in str(excinfo.value)


def test_load_rejects_missing_manifest_entry(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus", "acme/widgets", [_pr(3, "a" * 40, "b" * 40)])
    manifest = _write_manifest(tmp_path / "manifest.toml", ["other/repo"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest)
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "acme/widgets" in str(excinfo.value)
    assert "manifest" in str(excinfo.value).lower()


def test_load_rejects_record_without_base_sha(tmp_path: Path) -> None:
    """No base SHA means no reviewable diff and no image to build — fail loudly."""
    record = _pr(4, "a" * 40, "b" * 40)
    record["base_sha"] = None
    corpus = _write_corpus(tmp_path / "corpus", "acme/widgets", [record])
    manifest = _write_manifest(tmp_path / "manifest.toml", ["acme/widgets"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest)
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "base_sha" in str(excinfo.value)
    assert "acme/widgets#4" in str(excinfo.value)


def test_load_requires_corpus_dir_flag(tmp_path: Path) -> None:
    taskset = DaydreamReviewTaskset(DaydreamReviewConfig(id="daydream-review-v1"))
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "--taskset.corpus-dir" in str(excinfo.value)


def test_load_requires_manifest_path_flag(corpus_mini_dir: Path) -> None:
    taskset = DaydreamReviewTaskset(DaydreamReviewConfig(id="daydream-review-v1", corpus_dir=corpus_mini_dir))
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "--taskset.manifest-path" in str(excinfo.value)


def test_loader_contract_resolves_package(corpus_mini_dir: Path, fixture_manifest_path: Path) -> None:
    """The real path the verifiers CLI/orchestrator takes (loaders.py:110-127)."""
    from verifiers.v1.loaders import load_taskset, taskset_config_type

    config_type = taskset_config_type("daydream-review-v1")
    assert config_type is DaydreamReviewConfig

    taskset = load_taskset(
        config_type(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
        )
    )
    assert len(taskset.select()) == 2
