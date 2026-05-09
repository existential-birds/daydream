"""Deep-mode CLI validation tests.

Deep is the default; ``--shallow`` opts into the single-stack flow.
"""

import pytest

from daydream.cli import _parse_args


def test_default_is_deep() -> None:
    """Without --shallow, the run is deep (config.shallow == False)."""
    config = _parse_args(["target"])
    assert config.shallow is False


def test_shallow_flag_opts_in() -> None:
    config = _parse_args(["target", "--shallow"])
    assert config.shallow is True


@pytest.mark.parametrize("stage", ["ttt", "per-stack", "merge"])
def test_deep_resume_stages_accepted(stage: str) -> None:
    """ttt/per-stack/merge are valid resume stages in the (default) deep mode."""
    config = _parse_args(["target", "--start-at", stage])
    assert config.start_at == stage


@pytest.mark.parametrize("stage", ["ttt", "per-stack", "merge"])
def test_shallow_rejects_deep_resume_stages(stage: str) -> None:
    """Deep-pipeline resume stages are not valid with --shallow."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--shallow", "--start-at", stage])


@pytest.mark.parametrize("stage", ["parse", "test"])
def test_deep_rejects_shallow_only_stages(stage: str) -> None:
    """parse/test resume points are ambiguous in deep mode and are rejected."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--start-at", stage])


@pytest.mark.parametrize("stage", ["parse", "test", "fix", "review"])
def test_shallow_accepts_loop_stages(stage: str) -> None:
    """review/parse/fix/test resume stages are valid with --shallow."""
    config = _parse_args(["target", "--shallow", "--start-at", stage])
    assert config.start_at == stage
