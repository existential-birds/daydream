"""Deep-mode CLI validation tests.

Deep is the default; ``--shallow`` opts into the single-stack flow.
"""

import pytest

from daydream.cli import _parse_args


def test_default_is_deep() -> None:
    """Without --shallow, the run is deep (config.shallow == False)."""
    config = _parse_args(["target"])
    assert config.shallow is False


def test_trajectory_help_states_public_default_and_live_external(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--help-all names the public post-finalization default and live external updates.

    Semantics only: stable option names and source/public-path wording, never
    frozen argparse wrapping.
    """
    with pytest.raises(SystemExit):
        _parse_args(["--help-all"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--trajectory" in out
    assert "<target>/.daydream/runs/<session_id>/trajectory.json" in out
    assert "after finalization" in out
    assert "explicit external paths receive live updates" in out


def test_dump_artifacts_help_states_merge_semantics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--help-all describes --dump-artifacts as a preserving per-file merge."""
    with pytest.raises(SystemExit):
        _parse_args(["--help-all"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--dump-artifacts" in out
    assert "Merge the finalized run bundle" in out
    assert "Preserves unrelated destination files" in out


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
def test_deep_rejects_legacy_resume_stages(stage: str) -> None:
    """parse/test are legacy shallow-loop stages; every mode rejects them."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--start-at", stage])


@pytest.mark.parametrize("stage", ["parse", "test"])
def test_shallow_rejects_legacy_resume_stages(
    stage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """parse/test have no mapping in the unified pipeline, even with --shallow.

    They must error out rather than silently restart the full pipeline (the
    pre-#330 behavior that let ``--shallow --start-at test --yes`` re-review
    and re-apply fixes).
    """
    with pytest.raises(SystemExit):
        _parse_args(["target", "--shallow", "--start-at", stage])
    assert "no mapping in the unified pipeline" in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["fix", "review"])
def test_shallow_accepts_unified_resume_stages(stage: str) -> None:
    """review (fresh) and fix (resume after the merged report) are valid with --shallow."""
    config = _parse_args(["target", "--shallow", "--start-at", stage])
    assert config.start_at == stage
