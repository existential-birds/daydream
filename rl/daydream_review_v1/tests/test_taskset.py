"""Phase 1: taskset construction from a harvested corpus, with C5 enforcement."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from daydream_review_v1.fixture import (
    FIXTURE_BASE_SHA,
    FIXTURE_PR1_HEAD_SHA,
    FIXTURE_PR2_HEAD_SHA,
    FIXTURE_SLUG,
    FIXTURE_TEST_COMMAND,
    build_fixture_repo,
)
from daydream_review_v1.gate_refusal import Stage0GateRefused
from daydream_review_v1.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewTaskset,
    GoldenComment,
    load_manifest,
)


def _write_corpus(root: Path, repo: str, prs: list[dict[str, object]]) -> Path:
    """Write a minimal harvested-corpus at *root*."""
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
        f'protected_test_paths = {json.dumps(["tests"])}\n'
        for slug in slugs
    )
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("precreate_dest", [False, True], ids=["missing-destination", "empty-destination"])
def test_fixture_repo_is_deterministic(tmp_path: Path, precreate_dest: bool) -> None:
    """The SHAs pinned in fixture.py, corpus-mini and the manifest are the real ones."""
    dest = tmp_path / "fx"
    if precreate_dest:
        dest.mkdir()
    repo = build_fixture_repo(dest)
    assert (repo.base_sha, repo.pr1_head_sha, repo.pr2_head_sha) == (
        FIXTURE_BASE_SHA,
        FIXTURE_PR1_HEAD_SHA,
        FIXTURE_PR2_HEAD_SHA,
    )
    green = subprocess.run(FIXTURE_TEST_COMMAND.split(), cwd=repo.path, capture_output=True, text=True)
    assert green.returncode == 0, green.stderr


@pytest.mark.parametrize(
    "kind",
    ["non-empty-dir", "existing-file"],
    ids=["non-empty-directory", "existing-file"],
)
def test_fixture_repo_rejects_occupied_destination(tmp_path: Path, kind: str) -> None:
    """build_fixture_repo raises before mutating an occupied destination."""
    dest = tmp_path / "occupied"
    if kind == "non-empty-dir":
        dest.mkdir()
        (dest / "caller.txt").write_text("caller-owned", encoding="utf-8")
    else:
        dest.write_text("caller-owned", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture destination must be a new or empty directory"):
        build_fixture_repo(dest)

    if kind == "non-empty-dir":
        assert (dest / "caller.txt").read_text(encoding="utf-8") == "caller-owned"
        assert sorted(p.name for p in dest.iterdir()) == ["caller.txt"]
    else:
        assert dest.read_text(encoding="utf-8") == "caller-owned"


def test_fixture_cli_rejects_existing_git_repository_without_modification(tmp_path: Path) -> None:
    """An occupied git destination is rejected before any mutation; the CLI exits 2."""
    dest = tmp_path / "occupied"
    dest.mkdir()
    subprocess.run(["git", "-C", str(dest), "init", "--quiet", "--initial-branch", "main"], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "commit.gpgsign", "false"], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "Caller"], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "caller@example.com"], check=True)
    (dest / "caller.txt").write_text("caller-owned", encoding="utf-8")
    subprocess.run(["git", "-C", str(dest), "add", "caller.txt"], check=True)
    subprocess.run(["git", "-C", str(dest), "commit", "-m", "caller commit"], check=True)

    head_before = (dest / ".git" / "HEAD").read_text(encoding="utf-8")
    config_before = (dest / ".git" / "config").read_text(encoding="utf-8")
    status_before = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"], capture_output=True, text=True
    ).stdout
    entries_before = sorted(p.name for p in dest.iterdir())

    proc = subprocess.run(
        [sys.executable, "-m", "daydream_review_v1.fixture", str(dest)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "fixture destination must be a new or empty directory" in proc.stderr
    assert (dest / "caller.txt").read_text(encoding="utf-8") == "caller-owned"
    assert (dest / ".git" / "HEAD").read_text(encoding="utf-8") == head_before
    assert (dest / ".git" / "config").read_text(encoding="utf-8") == config_before
    assert (
        subprocess.run(["git", "-C", str(dest), "status", "--porcelain"], capture_output=True, text=True).stdout
        == status_before
    )
    assert sorted(p.name for p in dest.iterdir()) == entries_before


def test_load_builds_tasks_from_fixture_corpus(
    corpus_mini_dir: Path, fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
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
    assert pr1.protected_test_paths == ["tests"]
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


def test_use_images_false_leaves_tasks_imageless(
    corpus_mini_dir: Path, fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The subprocess smoke path needs imageless tasks (verifiers env.py:189-195)."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
            use_images=False,
        )
    )
    tasks = list(taskset.load())
    assert [task.data.image for task in tasks] == [None, None]
    assert all(task.data.test_command == FIXTURE_TEST_COMMAND for task in tasks)


def test_load_rejects_excluded_repo(tmp_path: Path, stage0_gate_report: Path) -> None:
    """C5 is unconditional: an excluded slug fails the load, manifest or not."""
    corpus = _write_corpus(tmp_path / "corpus", "getsentry/sentry", [_pr(7, "a" * 40, "b" * 40)])
    manifest = _write_manifest(tmp_path / "manifest.toml", [])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)
    assert "getsentry/sentry" in str(excinfo.value)


def test_load_rejects_excluded_repo_case_insensitively(tmp_path: Path, stage0_gate_report: Path) -> None:
    """GitHub slugs are case-insensitive; `GetSentry/Sentry` is the same repo."""
    corpus = _write_corpus(tmp_path / "corpus", "GetSentry/Sentry", [_pr(7, "a" * 40, "b" * 40)])
    manifest = _write_manifest(tmp_path / "manifest.toml", ["GetSentry/Sentry"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)


def test_load_rejects_corpus_without_benchmark_data(tmp_path: Path, stage0_gate_report: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus", "acme/widgets", [_pr(3, "a" * 40, "b" * 40)])
    (corpus / "results" / "benchmark_data.json").unlink()
    manifest = _write_manifest(tmp_path / "manifest.toml", ["acme/widgets"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "benchmark_data.json" in str(excinfo.value)


def test_load_rejects_missing_manifest_entry(tmp_path: Path, stage0_gate_report: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus", "acme/widgets", [_pr(3, "a" * 40, "b" * 40)])
    manifest = _write_manifest(tmp_path / "manifest.toml", ["other/repo"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "acme/widgets" in str(excinfo.value)
    assert "manifest" in str(excinfo.value).lower()


def test_load_manifest_rejects_unknown_key(tmp_path: Path) -> None:
    """A misspelled optional key (setp_cmds) fails load_manifest, naming the key."""
    manifest = _write_manifest(tmp_path / "manifest.toml", ["acme/widgets"])
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + 'setp_cmds = ["true"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert ("extra_forbidden", ("setp_cmds",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


@pytest.mark.parametrize(
    "block, error_type",
    [
        (  # missing field entirely
            '[repos."acme/widgets"]\n'
            'clone_url = "https://github.com/acme/widgets"\n'
            'image = "daydream-rl/widgets"\n'
            'test_command = "pytest -q"\n'
            "setup_cmds = []\n",
            "missing",
        ),
        (  # present but empty
            '[repos."acme/widgets"]\n'
            'clone_url = "https://github.com/acme/widgets"\n'
            'image = "daydream-rl/widgets"\n'
            'test_command = "pytest -q"\n'
            "setup_cmds = []\n"
            "protected_test_paths = []\n",
            "too_short",
        ),
    ],
    ids=["missing", "empty"],
)
def test_load_manifest_rejects_missing_or_empty_protected_test_paths(
    tmp_path: Path, block: str, error_type: str
) -> None:
    """A missing or empty protected_test_paths inventory is a load error.

    The security boundary is structural: an entry that ships without a
    protected-path inventory must never silently load into an unprotected task.
    """
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(block, encoding="utf-8")

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert (error_type, ("protected_test_paths",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


@pytest.mark.parametrize(
    "entry",
    ["", ":(exclude)tests", "tests/*", "tests/?", "tests/[x]"],
    ids=["empty", "leading-colon-magic", "glob-star", "glob-question", "glob-bracket"],
)
def test_load_manifest_rejects_non_literal_protected_test_paths(
    tmp_path: Path, entry: str
) -> None:
    """A protected_test_paths entry with git pathspec syntax is a load error.

    The manifest promises LITERAL repository-relative paths, but the scoring gate
    passes each entry to git as a bare pathspec, where ``*``/``?``/``[`` are glob
    metacharacters and a leading ``:`` is pathspec magic. A glob-shaped entry that
    matches nothing would read as a clean diff and an empty ls-files list, letting
    test_command run against an unprotected oracle — so such entries must never
    load, exactly like a missing or empty inventory.
    """
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        '[repos."acme/widgets"]\n'
        'clone_url = "https://github.com/acme/widgets"\n'
        'image = "daydream-rl/widgets"\n'
        'test_command = "pytest -q"\n'
        "setup_cmds = []\n"
        f"protected_test_paths = {json.dumps([entry])}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert ("value_error", ("protected_test_paths",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


def test_golden_comment_rejects_unknown_key() -> None:
    """A misspelled/extra key in benchmark_data.json must not be silently dropped.

    GoldenComment parses user-supplied corpus data (harvested upstream review
    comments). Rejecting unknown keys mirrors the extra="forbid" guard on
    _ManifestEntry so schema drift in the corpus fails loudly instead of being
    silently ignored.
    """
    with pytest.raises(ValidationError) as excinfo:
        GoldenComment.model_validate(
            {"comment": "looks good", "typo_field": "nope"}
        )
    assert ("extra_forbidden", ("typo_field",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


def test_load_rejects_record_without_base_sha(tmp_path: Path, stage0_gate_report: Path) -> None:
    """No base SHA means no reviewable diff and no image to build — fail loudly."""
    record = _pr(4, "a" * 40, "b" * 40)
    record["base_sha"] = None
    corpus = _write_corpus(tmp_path / "corpus", "acme/widgets", [record])
    manifest = _write_manifest(tmp_path / "manifest.toml", ["acme/widgets"])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1", corpus_dir=corpus, manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "base_sha" in str(excinfo.value)
    assert "acme/widgets#4" in str(excinfo.value)


def test_load_refuses_without_gate_report_path(corpus_mini_dir: Path, fixture_manifest_path: Path) -> None:
    """M4: an unconfigured gate path is itself a refusal, never a default-to-allow."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            gate_report_path=Path(""),
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "--taskset.gate-report-path" in str(excinfo.value)


def test_load_refuses_missing_gate_report(corpus_mini_dir: Path, fixture_manifest_path: Path, tmp_path: Path) -> None:
    """A missing gate report refuses the load, not just the require_stage0_gate leaf."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            gate_report_path=tmp_path / "missing-gate.json",
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "gate report missing" in str(excinfo.value)


def test_load_refuses_failed_gate_report(corpus_mini_dir: Path, fixture_manifest_path: Path, tmp_path: Path) -> None:
    """A report that did not pass refuses the load, not just the require_stage0_gate leaf."""
    gate = tmp_path / "failed-gate.json"
    gate.write_text(json.dumps({"passed": False, "separation": 0.01}), encoding="utf-8")
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            gate_report_path=gate,
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "failed" in str(excinfo.value)


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


def test_loader_contract_resolves_package(
    corpus_mini_dir: Path, fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The real path the verifiers CLI/orchestrator takes (loaders.py:110-127)."""
    from verifiers.v1.loaders import load_taskset, taskset_config_type

    config_type = taskset_config_type("daydream-review-v1")
    assert config_type is DaydreamReviewConfig

    taskset = load_taskset(
        config_type(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
        )
    )
    # load(), not select(): select() is a 0.2.1 convenience the verifiers
    # submodule prime-rl trains against does not have, and load() is the payload.
    assert len(list(taskset.load())) == 2


def test_reference_corpus_loads_against_the_manifest(
    fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The reference entry is real upstream history, not another synthetic repo.

    pallets/itsdangerous PR #406 (BSD-3-Clause, not on the C5 exclusion list) is
    the proof that the manifest + corpus + image pipeline works on a repository
    nobody here authored. Its image is built from these exact SHAs, so a drift
    between the corpus and the manifest would silently point tasks at an image
    that does not exist.
    """
    corpus = Path(__file__).parent / "fixtures" / "corpus-reference"
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus,
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
        )
    )
    (task,) = taskset.load()
    assert task.data.repo_slug == "pallets/itsdangerous"
    assert task.data.pr_number == 406
    assert task.data.head_sha == "4bb03cd6819228f30079885297299fe568a62863"
    assert task.data.base_sha == "4dffa1963f896a0a311dec3c14f003a5f382c446"
    assert task.data.test_command == "/opt/repo-venv/bin/python -m pytest -q"
    assert task.data.protected_test_paths == [
        "tests",
        "conftest.py",
        ".pytest.ini",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
    ]
    assert task.data.image == "daydream-rl/itsdangerous:4bb03cd68192"
