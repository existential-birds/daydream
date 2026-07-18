"""Tests for the pluggable benchmark corpus seam."""

import json
from pathlib import Path

import pytest

from daydream.benchmark.config import BenchConfig
from daydream.benchmark.corpus import resolve_corpus


def _config(**overrides) -> BenchConfig:
    base = dict(
        benchmark_repo=None,
        cache_dir=Path("/cache"),
        force=False,
        score=False,
        only=None,
        limit=None,
        trajectory_dir=Path("/traj"),
    )
    base.update(overrides)
    return BenchConfig(**base)  # type: ignore[arg-type]


def _write_index(harvest_dir: Path, prs: list[dict]) -> None:
    harvest_dir.mkdir(parents=True, exist_ok=True)
    (harvest_dir / "index.json").write_text(
        json.dumps({"repo": "acme/widgets", "bot": "cr[bot]", "prs": prs}), encoding="utf-8"
    )


def test_withmartian_corpus_pins_26_prs():
    source = resolve_corpus(_config(benchmark_repo=Path("/bench")))
    assert source.kind == "withmartian"
    assert source.root == Path("/bench")
    assert len(source.prs) == 26
    assert all(pr.base_sha is not None and pr.base_ref is None for pr in source.prs)


def test_harvested_corpus_builds_prs_from_index(tmp_path):
    harvest_dir = tmp_path / "harvest"
    _write_index(
        harvest_dir,
        [
            {"pr_number": 12, "review_commit_id": "a" * 40, "base_ref": "develop"},
            {"pr_number": 7, "review_commit_id": "b" * 40, "base_ref": "master"},
        ],
    )
    source = resolve_corpus(_config(harvest_dir=harvest_dir))

    assert source.kind == "harvested"
    assert source.root == harvest_dir
    assert [pr.pr_number for pr in source.prs] == [12, 7]
    first, second = source.prs
    assert first.golden_url == "https://github.com/acme/widgets/pull/12"
    assert first.clone_url == "https://github.com/acme/widgets"
    assert first.source_repo == "acme/widgets"
    assert first.base_sha is None  # pre-base_sha corpus: derived from base_ref at acquisition time
    assert first.head_sha == "a" * 40  # the bot's review snapshot, not the PR head
    assert first.base_ref == "develop"
    assert second.base_ref == "master"


def test_harvested_corpus_pins_recorded_base_sha(tmp_path):
    harvest_dir = tmp_path / "harvest"
    _write_index(
        harvest_dir,
        [
            {"pr_number": 12, "review_commit_id": "a" * 40, "base_ref": "develop", "base_sha": "c" * 40},
            {"pr_number": 7, "review_commit_id": "b" * 40, "base_ref": "master", "base_sha": ""},
        ],
    )
    source = resolve_corpus(_config(harvest_dir=harvest_dir))

    pinned, legacy = source.prs
    assert pinned.base_sha == "c" * 40
    assert pinned.base_ref == "develop"  # kept as the fallback for acquisition
    assert legacy.base_sha is None


@pytest.mark.parametrize("base_ref", [None, "", "missing"])
def test_harvested_corpus_rejects_record_without_base_ref(tmp_path, base_ref):
    harvest_dir = tmp_path / "harvest"
    record = {"pr_number": 7, "review_commit_id": "b" * 40}
    if base_ref != "missing":
        record["base_ref"] = base_ref
    _write_index(harvest_dir, [{"pr_number": 12, "review_commit_id": "a" * 40, "base_ref": "develop"}, record])

    with pytest.raises(ValueError, match=r"acme/widgets PR #7 has no base_ref"):
        resolve_corpus(_config(harvest_dir=harvest_dir))


def test_harvested_corpus_skips_records_without_commit_id(tmp_path):
    harvest_dir = tmp_path / "harvest"
    _write_index(
        harvest_dir,
        [
            {"pr_number": 1, "review_commit_id": None, "base_ref": "main"},
            {"pr_number": 2, "review_commit_id": "", "base_ref": "main"},
            {"pr_number": 3, "review_commit_id": "c" * 40, "base_ref": "main"},
        ],
    )
    source = resolve_corpus(_config(harvest_dir=harvest_dir))
    assert [pr.pr_number for pr in source.prs] == [3]


def test_config_requires_exactly_one_corpus_root():
    with pytest.raises(ValueError, match="benchmark_repo or harvest_dir"):
        _ = _config().corpus_root
    with pytest.raises(ValueError, match="benchmark_repo or harvest_dir"):
        resolve_corpus(_config())
    assert _config(benchmark_repo=Path("/b")).corpus_root == Path("/b")
    assert _config(harvest_dir=Path("/h")).corpus_root == Path("/h")
