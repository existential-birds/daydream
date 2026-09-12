"""Phase 1: taskset construction from the image manifest, with C5 enforcement."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

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


def _manifest_entry(slug: str, prs: list[dict[str, object]]) -> str:
    block = (
        f'[repos."{slug}"]\n'
        f'clone_url = "https://github.com/{slug}"\n'
        f'image = "daydream-rl/{slug.split("/")[-1]}"\n'
        f'test_command = "pytest -q"\n'
        "setup_cmds = []\n"
        f'protected_test_paths = {json.dumps(["tests"])}\n'
    )
    for pr in prs:
        block += f'\n[[repos."{slug}".prs]]\n'
        block += "\n".join(f"{key} = {json.dumps(value)}" for key, value in pr.items()) + "\n"
    return block


def _write_manifest(path: Path, entries: list[tuple[str, list[dict[str, object]]]]) -> Path:
    path.write_text("\n".join(_manifest_entry(slug, prs) for slug, prs in entries), encoding="utf-8")
    return path


@pytest.mark.parametrize("precreate_dest", [False, True], ids=["missing-destination", "empty-destination"])
def test_fixture_repo_is_deterministic(tmp_path: Path, precreate_dest: bool) -> None:
    """The SHAs pinned in fixture.py and the manifest are the real ones."""
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
        [sys.executable, "-m", "daydream_review.fixture", str(dest)],
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


def test_load_builds_tasks_from_the_committed_manifest(
    fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
        )
    )
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


def test_use_images_false_leaves_tasks_imageless(
    fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The subprocess smoke path needs imageless tasks (verifiers env.py:189-195)."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
            use_images=False,
        )
    )
    tasks = list(taskset.load())
    assert [task.data.image for task in tasks] == [None, None, None]
    assert [
        task.data.test_command for task in tasks
    ] == [FIXTURE_TEST_COMMAND, FIXTURE_TEST_COMMAND, "/opt/repo-venv/bin/python -m pytest -q"]


def test_load_rejects_excluded_repo(tmp_path: Path, stage0_gate_report: Path) -> None:
    """C5 is unconditional: an excluded slug fails the load."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("getsentry/sentry", [_pr(7, "a" * 40, "b" * 40)])])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review", manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)
    assert "getsentry/sentry" in str(excinfo.value)


def test_load_rejects_excluded_repo_case_insensitively(tmp_path: Path, stage0_gate_report: Path) -> None:
    """GitHub slugs are case-insensitive; `GetSentry/Sentry` is the same repo."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("GetSentry/Sentry", [_pr(7, "a" * 40, "b" * 40)])])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review", manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "C5" in str(excinfo.value)


def test_load_rejects_manifest_without_pr_snapshots(tmp_path: Path) -> None:
    """An entry without at least one PR snapshot is a load error, not an empty taskset."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [])])
    manifest.write_text(
        # _manifest_entry with an empty prs list still writes the entry table;
        # strip the (absent) pr tables and rely on min_length to reject.
        manifest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert (
        ("too_short", ("prs",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]
        or ("missing", ("prs",)) in [(err["type"], err["loc"]) for err in excinfo.value.errors()]
    )


def test_load_manifest_rejects_unknown_key(tmp_path: Path) -> None:
    """A misspelled optional key (setp_cmds) fails load_manifest, naming the key."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [_pr(3, "a" * 40, "b" * 40)])])
    # A misspelled entry key must land inside the entry table, not in a PR
    # snapshot table: insert it just ahead of the entry's first pr table.
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            '\n[[repos."acme/widgets".prs]]', 'setp_cmds = ["true"]\n\n[[repos."acme/widgets".prs]]', 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert ("extra_forbidden", ("setp_cmds",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ], excinfo.value.errors()


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
    manifest.write_text(
        block
        + '\n[[repos."acme/widgets".prs]]\n'
        + "pr_number = 3\n"
        + f'base_sha = "{A_SHA}"\n'
        + f'head_sha = "{B_SHA}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert (error_type, ("protected_test_paths",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


@pytest.mark.parametrize(
    "entry",
    [
        "",
        ":(exclude)tests",
        "tests/*",
        "tests/?",
        "tests/[x]",
        "/tests",
        "tests/./unit",
        "docs/../missing",
        "./tests",
        "tests/.",
        "../tests",
        "tests/..",
    ],
    ids=[
        "empty",
        "leading-colon-magic",
        "glob-star",
        "glob-question",
        "glob-bracket",
        "absolute",
        "dot-component",
        "dotdot-component",
        "leading-dot-component",
        "trailing-dot-component",
        "leading-dotdot-component",
        "trailing-dotdot-component",
    ],
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
        f"protected_test_paths = {json.dumps([entry])}\n"
        '\n[[repos."acme/widgets".prs]]\n'
        "pr_number = 3\n"
        f'base_sha = "{A_SHA}"\n'
        f'head_sha = "{B_SHA}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_manifest(manifest)
    assert ("value_error", ("protected_test_paths",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


def test_load_manifest_accepts_canonical_protected_test_paths(tmp_path: Path) -> None:
    """Canonical nested paths and dotfiles load unchanged through the loader."""
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        '[repos."acme/widgets"]\n'
        'clone_url = "https://github.com/acme/widgets"\n'
        'image = "daydream-rl/widgets"\n'
        'test_command = "pytest -q"\n'
        'setup_cmds = []\n'
        'protected_test_paths = ["tests/unit", "tests/unit/test_api.py", ".pytest.ini", "tests/.hidden"]\n'
        '\n[[repos."acme/widgets".prs]]\n'
        "pr_number = 3\n"
        f'base_sha = "{A_SHA}"\n'
        f'head_sha = "{B_SHA}"\n',
        encoding="utf-8",
    )
    loaded = load_manifest(manifest)
    assert loaded["acme/widgets"].protected_test_paths == [
        "tests/unit", "tests/unit/test_api.py", ".pytest.ini", "tests/.hidden"
    ]


def test_golden_comment_rejects_unknown_key() -> None:
    """A misspelled/extra key in a manifest golden_comments table must not be
    silently dropped.

    GoldenComment parses manifest-declared upstream review comments. Rejecting
    unknown keys mirrors the extra="forbid" guard on _ManifestEntry so schema
    drift in the manifest fails loudly instead of being silently ignored.
    """
    with pytest.raises(ValidationError) as excinfo:
        GoldenComment.model_validate(
            {"comment": "looks good", "typo_field": "nope"}
        )
    assert ("extra_forbidden", ("typo_field",)) in [
        (err["type"], err["loc"]) for err in excinfo.value.errors()
    ]


def test_load_rejects_snapshot_without_base_sha(tmp_path: Path, stage0_gate_report: Path) -> None:
    """No base SHA means no reviewable diff and no image to build — fail loudly."""
    manifest = _write_manifest(tmp_path / "manifest.toml", [("acme/widgets", [_pr(4, "", "b" * 40)])])

    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review", manifest_path=manifest, gate_report_path=stage0_gate_report
        )
    )
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "base_sha" in str(excinfo.value)
    assert "acme/widgets#4" in str(excinfo.value)


def test_load_refuses_without_gate_report_path(fixture_manifest_path: Path) -> None:
    """M4: an unconfigured gate path is itself a refusal, never a default-to-allow."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=Path(""),
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "--taskset.gate-report-path" in str(excinfo.value)


def test_load_refuses_missing_gate_report(fixture_manifest_path: Path, tmp_path: Path) -> None:
    """A missing gate report refuses the load, not just the require_stage0_gate leaf."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=tmp_path / "missing-gate.json",
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "gate report missing" in str(excinfo.value)


def test_load_refuses_failed_gate_report(fixture_manifest_path: Path, tmp_path: Path) -> None:
    """A report that did not pass refuses the load, not just the require_stage0_gate leaf."""
    gate = tmp_path / "failed-gate.json"
    gate.write_text(json.dumps({"passed": False, "separation": 0.01}), encoding="utf-8")
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=gate,
        )
    )
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
    model.write_text(
        json.dumps(
            {
                "weights": {"bug": 1.0},
                "bias": -0.25,
                # Not the split the report's evidence_digest was computed over.
                "split_digest": "some-other-split",
                "label_ratio_reported": 0.5,
                "train_rows": 10,
                "held_out_rows": 4,
                "held_out_accuracy": 0.75,
                "model_fingerprint": "",
            }
        ),
        encoding="utf-8",
    )
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
            outcome_model_path=model,
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "does not bind" in str(excinfo.value)
    assert str(model) in str(excinfo.value)


def test_load_refuses_missing_outcome_model(
    fixture_manifest_path: Path, stage0_gate_report: Path, tmp_path: Path
) -> None:
    """A configured-but-absent checkpoint refuses the load, not the first score."""
    missing = tmp_path / "missing-outcome-model.json"
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
            outcome_model_path=missing,
        )
    )
    with pytest.raises(Stage0GateRefused) as excinfo:
        list(taskset.load())
    assert "missing" in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_load_binds_outcome_model_to_gate_report(
    fixture_manifest_path: Path, stage0_gate_report: Path, outcome_model_path: Path
) -> None:
    """The bound pair (conftest fixtures) loads, stamping the model onto each task (M13)."""
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
            outcome_model_path=outcome_model_path,
        )
    )
    tasks = list(taskset.load())
    assert tasks
    assert all(task.config.outcome_model_path == outcome_model_path for task in tasks)


def test_load_requires_manifest_path_flag() -> None:
    taskset = DaydreamReviewTaskset(DaydreamReviewConfig(id="daydream-review"))
    with pytest.raises(ValueError) as excinfo:
        list(taskset.load())
    assert "--taskset.manifest-path" in str(excinfo.value)


def test_loader_contract_resolves_package(
    fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The real path the verifiers CLI/orchestrator takes (loaders.py:110-127)."""
    from verifiers.v1.loaders import load_taskset, taskset_config_type

    config_type = taskset_config_type("daydream-review")
    assert config_type is DaydreamReviewConfig

    taskset = load_taskset(
        config_type(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
        )
    )
    # load(), not select(): select() is a 0.2.1 convenience the verifiers
    # submodule prime-rl trains against does not have, and load() is the payload.
    assert len(list(taskset.load())) == 3


def test_reference_manifest_loads_against_the_committed_snapshot(
    fixture_manifest_path: Path, stage0_gate_report: Path
) -> None:
    """The reference entry is real upstream history, not another synthetic repo.

    pallets/itsdangerous PR #406 (BSD-3-Clause, not on the C5 exclusion list) is
    the proof that the manifest + image pipeline works on a repository nobody
    here authored. Its image is built from the exact SHAs the entry pins, so a
    drift between a task and its image would silently point rollouts at an image
    that does not exist.
    """
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review",
            manifest_path=fixture_manifest_path,
            gate_report_path=stage0_gate_report,
        )
    )
    tasks = taskset.load()
    (task,) = [t for t in tasks if t.data.repo_slug == "pallets/itsdangerous"]
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
