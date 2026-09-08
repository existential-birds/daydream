"""Bounded filesystem observation controls for the external CLI fixture."""

from pathlib import Path

import pytest

from tests.harness.protocol_cli import _cwd_observation, _observed_argv


@pytest.mark.parametrize("backend", ["codex", "pi", "osprey"])
def test_protocol_observation_refuses_unknown_content_flags(backend: str) -> None:
    with pytest.raises(ValueError, match="unsupported or incomplete fixture argument"):
        _observed_argv(backend, ["--unknown-content", "PRIVATE_FIXTURE_CONTENT"])


@pytest.mark.parametrize("shape", ["empty", "deep", "mixed"])
def test_protocol_cwd_budget_includes_directories(tmp_path: Path, shape: str) -> None:
    cursor = tmp_path
    deep_directories: list[Path] = []
    try:
        for index in range(300):
            if shape == "deep":
                cursor /= "d"
                cursor.mkdir()
                deep_directories.append(cursor)
            elif shape == "mixed" and index % 2:
                (tmp_path / f"file-{index}").write_bytes(b"SOURCE_CANARY")
            else:
                (tmp_path / f"dir-{index}").mkdir()

        observed = _cwd_observation(tmp_path)

        assert observed["walk_truncated"] is True
        assert len(observed["cwd_entries"]) == 256
    finally:
        # pytest's recursive temp cleanup can exceed its recursion budget here.
        for directory in reversed(deep_directories):
            directory.rmdir()


def test_protocol_cwd_observation_does_not_follow_directory_or_file_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_bytes(b"PRIOR_REASONING_CANARY")
    target = tmp_path / "target"
    target.mkdir()
    (target / "dir-link").symlink_to(outside, target_is_directory=True)
    (target / "file-link").symlink_to(outside / "private.txt")
    (target / "source.py").write_bytes(b"SOURCE_CANARY")

    observed = _cwd_observation(target)

    assert observed["walk_truncated"] is False
    assert observed["cwd_entries"] == ["dir-link", "file-link", "source.py"]
    assert observed["cwd_canaries"]["SOURCE_CANARY"] is True
    assert observed["cwd_canaries"]["PRIOR_REASONING_CANARY"] is False


def test_protocol_cwd_observation_refuses_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "oversized.txt").write_bytes(b"x" * 131_072 + b"PRIOR_REASONING_CANARY")

    observed = _cwd_observation(tmp_path)

    assert observed["walk_truncated"] is True
    assert observed["cwd_canaries"]["PRIOR_REASONING_CANARY"] is False
