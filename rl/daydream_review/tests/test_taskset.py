"""Phase 1: taskset construction from the image manifest, with C5 enforcement."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError
from verifiers.v1.loaders import load_taskset, taskset_config_type

from daydream_review.fixture import (
    FIXTURE_BASE_SHA,
    FIXTURE_PR1_HEAD_SHA,
    FIXTURE_PR2_HEAD_SHA,
    FIXTURE_SLUG,
    FIXTURE_TEST_COMMAND,
    build_fixture_repo,
)
from daydream_review.gate_refusal import Stage0GateRefused
from daydream_review.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewTaskset,
    GoldenComment,
    load_manifest,
)

A_SHA = "a" * 40
B_SHA = "b" * 40


def _pr(number: int, base_sha: str, head_sha: str) -> dict[str, object]:
    return {"pr_number": number, "base_sha": base_sha, "head_sha": head_sha, "base_ref": "main"}


def _manifest_entry(slug: str, prs: list[dict[str, object]], protected_test_paths: Sequence[str] | None = ("tests",),
) -> str:
    block = (
        f'[repos."{slug}"]\n'
        f'clone_url = "https://github.com/{slug}"\n'
        f'image = "daydream-rl/{slug.split("/")[-1]}"\n'
        f'test_command = "pytest -q"\n'
        "setup_cmds = []\n"
    )
    if protected_test_paths is not None:
        block += f"protected_test_paths = {json.dumps(list(protected_test_paths))}\n"
    for pr in prs:
        block += f'\n[[repos."{slug}".prs]]\n'
        block += "\n".join(f"{key} = {json.dumps(value)}" for key, value in pr.items()) + "\n"
    return block


def _write_manifest(path: Path, entries: list[tuple[str, list[dict[str, object]]]],
    protected_test_paths: Sequence[str] | None = ("tests",),
) -> Path:
    path.write_text(
        "\n".join(_manifest_entry(slug, prs, protected_test_paths) for slug, prs in entries),
        encoding="utf-8",
    )
    return path


def _taskset(manifest_path: Path, gate_report_path: Path, **overrides: object) -> DaydreamReviewTaskset:
    return DaydreamReviewTaskset(DaydreamReviewConfig(
            id="daydream-review", manifest_path=manifest_path, gate_report_path=gate_report_path, **overrides,
        )
    )


@pytest.mark.parametrize("precreate_dest", [False, True], ids=["missing-destination", "empty-destination"])
def test_fixture_repo_is_deterministic(tmp_path: Path, precreate_dest: bool) -> None:
    dest = tmp_path / "fx"
    if precreate_dest:
        dest.mkdir()
    repo = build_fixture_repo(dest)
    assert (repo.base_sha, repo.pr1_head_sha, repo.pr2_head_sha) == (
        FIXTURE_BASE_SHA, FIXTURE_PR1_HEAD_SHA, FIXTURE_PR2_HEAD_SHA,
    )
    green = subprocess.run(FIXTURE_TEST_COMMAND.split(), cwd=repo.path, capture_output=True, text=True)
    assert green.returncode == 0, green.stderr

@pytest.mark.parametrize("kind", ["non-empty-dir", "existing-file"], ids=["non-empty-directory", "existing-file"])
def test_fixture_repo_rejects_occupied_destination(tmp_path: Path, kind: str) -> None:
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
    status_before = subprocess.run(["git", "-C", str(dest), "status", "--porcelain"], capture_output=True, text=True
    ).stdout
    entries_before = sorted(p.name for p in dest.iterdir())

    proc = subprocess.run([sys.executable, "-m", "daydream_review.fixture", str(dest)], capture_output=True, text=True)

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "fixture destination must be a new or empty directory" in proc.stderr
    assert (dest / "caller.txt").read_text(encoding="utf-8") == "caller-owned"
    assert (dest / ".git" / "HEAD").read_text(encoding="utf-8") == head_before
    assert (dest / ".git" / "config").read_text(encoding="utf-8") == config_before
    assert (subprocess.run(["git", "-C", str(dest), "status", "--porcelain"], capture_output=True, text=True).stdout
        == status_before
    )
    assert sorted(p.name for p in dest.iterdir()) == entries_before

def test_load_builds_tasks_from_the_committed_manifest(fixture_manifest_path: Path, stage0_gate_report: Path) -> None:
    taskset = _taskset(fixture_manifest_path, stage0_gate_report)
    tasks = list(taskset.load())
    assert len(tasks) == 3

    by_pr = {task.data.pr_number: task.data for task in tasks}
    assert set(by_pr) == {1, 2, 406}

    pr1 = by_pr[1]
    assert pr1.repo_slug == FIXTURE_SLUG
    assert pr1.base_sha == FIXTURE_BASE_SHA
    assert pr1.head_sha == FIXTURE_PR1_HEAD_SHA
    assert pr1.base_ref == "main"
    assert pr1.test_command == FIXTURE_TEST_COMMAND
    assert pr1.protected_test_paths == ["tests"]
    assert pr1.clone_url == "fixture://daydream-rl-fixture"
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

    assert [task.data.idx for task in tasks] == [0, 1, 2]

def test_use_images_false_leaves_tasks_imageless(fixture_manifest_path: Path, stage0_gate_report: Path) -> None:
    """The subprocess smoke path needs imageless tasks (verifiers env.py:189-195)."""
    taskset = _taskset(fixture_manifest_path, stage0_gate_report, use_images=False)
    tasks = list(taskset.load())
    assert [task.data.image for task in tasks] == [None, None, None]
    assert [task.data.test_command for task in tasks
    ] == [FIXTURE_TEST_COMMAND, FIXTURE_TEST_COMMAND, "/opt/repo-venv/bin/python -m pytest -q"]

@pytest.mark.parametrize("slug", ["getsentry/sentry", "GetSentry/Sentry"])
def test_load_rejects_excluded_repo(tmp_path: Path, stage0_gate_report: Path, slug: str,) -> None:
    """C5 is unconditional and case-insensitive: an excluded slug fails the load."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [(slug, [_pr(7, "a" * 40, "b" * 40)])])

    taskset = _taskset(manifest, stage0_gate_report)
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)
    assert slug in str(excinfo.value)

def test_load_rejects_manifest_without_pr_snapshots(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [])])

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert (("too_short", ("prs",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]
        or ("missing", ("prs",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]
    )

def test_load_manifest_rejects_unknown_key(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [_pr(3, "a" * 40, "b" * 40)])])
    # A misspelled entry key must land inside the entry table, not in a PR
    # snapshot table: insert it just ahead of the entry's first pr table.
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(
            '\n[[repos."acme/widgets".prs]]', 'setp_cmds = ["true"]\n\n[[repos."acme/widgets".prs]]', 1
        ), encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert ("extra_forbidden", ("setp_cmds",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()
    ], excinfo.value.errors()

@pytest.mark.parametrize("protected_test_paths, error_type",
    [
        (None, "missing"),  # missing field entirely
        ([], "too_short"),  # present but empty
    ], ids=["missing", "empty"],
)
def test_load_manifest_rejects_missing_or_empty_protected_test_paths(
    tmp_path: Path, protected_test_paths: list[str] | None, error_type: str
) -> None:
    """Refuse missing/empty protected paths so no task loads with an unprotected test oracle."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [_pr(3, A_SHA, B_SHA)])],
        protected_test_paths=protected_test_paths,
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert (error_type, ("protected_test_paths",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]

@pytest.mark.parametrize("entry",
    ["", ":(exclude)tests", "tests/*", "tests/?", "tests/[x]", "/tests", "tests/./unit", "docs/../missing",
        "./tests", "tests/.", "../tests", "tests/..",
    ], ids=["empty", "leading-colon-magic", "glob-star", "glob-question", "glob-bracket", "absolute", "dot-component",
        "dotdot-component", "leading-dot-component", "trailing-dot-component", "leading-dotdot-component",
        "trailing-dotdot-component",
    ],
)
def test_load_manifest_rejects_non_literal_protected_test_paths(tmp_path: Path, entry: str) -> None:
    """Require literal protected paths, rejecting globs and Git pathspec magic.

    A pattern matching nothing could make both diff and ls-files falsely report
    a clean oracle and permit an unprotected suite.
    """
    manifest = _write_manifest(
        tmp_path / "manifest.toml", [("acme/widgets", [_pr(3, A_SHA, B_SHA)])], protected_test_paths=[entry],
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert ("value_error", ("protected_test_paths",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]

def test_load_manifest_accepts_canonical_protected_test_paths(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [_pr(3, A_SHA, B_SHA)])],
        protected_test_paths=["tests/unit", "tests/unit/test_api.py", ".pytest.ini", "tests/.hidden"],
    )
    loaded = load_manifest(manifest)
    assert loaded["acme/widgets"].protected_test_paths == [
        "tests/unit", "tests/unit/test_api.py", ".pytest.ini", "tests/.hidden"
    ]

def test_golden_comment_rejects_unknown_key() -> None:
    """Reject unknown golden-comment fields so manifest schema drift cannot silently discard data."""
    with pytest.raises(ValidationError) as excinfo:
        GoldenComment.model_validate({"comment": "looks good", "typo_field": "nope"})
    assert ("extra_forbidden", ("typo_field",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]

def test_load_rejects_snapshot_without_base_sha(tmp_path: Path, stage0_gate_report: Path) -> None:
    """No base SHA means no reviewable diff and no image to build — fail loudly."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [_pr(4, "", "b" * 40)])])

    taskset = _taskset(manifest, stage0_gate_report)
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "base_sha" in str(excinfo.value)
    assert "acme/widgets#4" in str(excinfo.value)

def test_load_refuses_without_gate_report_path(fixture_manifest_path: Path) -> None:
    taskset = _taskset(fixture_manifest_path, Path(""))
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "--taskset.gate-report-path" in str(excinfo.value)

def test_load_refuses_missing_gate_report(fixture_manifest_path: Path, tmp_path: Path) -> None:
    taskset = _taskset(fixture_manifest_path, tmp_path / "missing-gate.json")
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "gate report missing" in str(excinfo.value)

def test_load_refuses_failed_gate_report(fixture_manifest_path: Path, tmp_path: Path) -> None:
    gate = tmp_path / "failed-gate.json"
    gate.write_text(json.dumps({"passed": False, "separation": 0.01}), encoding="utf-8")
    taskset = _taskset(fixture_manifest_path, gate)
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "failed" in str(excinfo.value)

def test_load_refuses_model_not_bound_to_gate_report(
    fixture_manifest_path: Path, stage0_gate_report: Path, tmp_path: Path
) -> None:
    """M4 binding: a passed report plus a checkpoint that does not re-derive its
    evidence_digest refuses the load — any-checkpoint-plus-any-report must not
    schedule rollouts."""
    model = tmp_path / "outcome-model.json"
    model.write_text(json.dumps({"weights": {"bug": 1.0}, "bias": -0.25,
                "split_digest": "some-other-split", "label_ratio_reported": 0.5, "train_rows": 10, "held_out_rows": 4,
                "held_out_accuracy": 0.75, "model_fingerprint": "",
            }
        ), encoding="utf-8",
    )
    taskset = _taskset(fixture_manifest_path, stage0_gate_report, outcome_model_path=model)
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "does not bind" in str(excinfo.value)
    assert str(model) in str(excinfo.value)

def test_load_refuses_missing_outcome_model(fixture_manifest_path: Path, stage0_gate_report: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-outcome-model.json"
    taskset = _taskset(fixture_manifest_path, stage0_gate_report, outcome_model_path=missing)
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "missing" in str(excinfo.value)
    assert str(missing) in str(excinfo.value)

def test_load_binds_outcome_model_to_gate_report(
    fixture_manifest_path: Path, stage0_gate_report: Path, outcome_model_path: Path
) -> None:
    taskset = _taskset(fixture_manifest_path, stage0_gate_report, outcome_model_path=outcome_model_path)
    tasks = list(taskset.load())
    assert tasks
    assert all(task.config._outcome_scorer is not None for task in tasks)
    assert all("_outcome_scorer" not in task.config.model_dump() for task in tasks)

def test_load_requires_manifest_path_flag() -> None:
    taskset = DaydreamReviewTaskset(DaydreamReviewConfig(id="daydream-review"))
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "--taskset.manifest-path" in str(excinfo.value)

def test_loader_contract_resolves_package(fixture_manifest_path: Path, stage0_gate_report: Path) -> None:
    """The real path the verifiers CLI/orchestrator takes (loaders.py:110-127)."""
    config_type = taskset_config_type("daydream-review")
    assert config_type is DaydreamReviewConfig

    taskset = load_taskset(
        config_type(id="daydream-review", manifest_path=fixture_manifest_path, gate_report_path=stage0_gate_report)
    )
    # load(), not select(): select() is a 0.2.1 convenience the verifiers
    # submodule prime-rl trains against does not have, and load() is the payload.
    assert len(list(taskset.load())) == 3

def test_reference_manifest_loads_against_the_committed_snapshot(fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The real itsdangerous PR pins must match the image tags built from their exact SHAs."""
    taskset = _taskset(fixture_manifest_path, stage0_gate_report)
    tasks = taskset.load()
    (task,) = [t for t in tasks if t.data.repo_slug == "pallets/itsdangerous"]
    assert task.data.pr_number == 406
    assert task.data.head_sha == "4bb03cd6819228f30079885297299fe568a62863"
    assert task.data.base_sha == "4dffa1963f896a0a311dec3c14f003a5f382c446"
    assert task.data.test_command == "/opt/repo-venv/bin/python -m pytest -q"
    assert task.data.protected_test_paths == [
        "tests", "conftest.py", ".pytest.ini", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini",
    ]
    assert task.data.image == "daydream-rl/itsdangerous:4bb03cd68192"
