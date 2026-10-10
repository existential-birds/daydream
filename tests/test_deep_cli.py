"""Deep-mode CLI validation tests.

Deep is the default; ``--shallow`` opts into the single-stack flow.
"""

import pytest

from daydream.commands.review import _parse_args


def test_help_all_states_trajectory_and_dump_artifacts_semantics(capsys: pytest.CaptureFixture[str],) -> None:
    """--help-all names the public post-finalization trajectory default, live external
    updates, raw diagnostic dumps, and mandatory direct-upload scanning.

    Semantics only: stable option names and source/public-path wording, never
    frozen argparse wrapping.
    """
    with pytest.raises(SystemExit):
        _parse_args(["--help-all"])
    out = " ".join(capsys.readouterr().out.split())
    for fragment in ("--trajectory", "<target>/.daydream/runs/<session_id>/trajectory.json", "after finalization",
        "explicit external paths receive live updates", "--dump-artifacts", "Merge the finalized run bundle",
        "Preserves unrelated destination files",
        "Always copies exact assembled bytes, including credentials and binary files",
        "without scanning or sanitizing", "Works on every flow",
        "Always refuses blocking credentials and scanner failures", "Advisory findings are allowed",
    ):
        assert fragment in out, fragment

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
def test_shallow_rejects_legacy_resume_stages(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    """parse/test have no mapping in the unified pipeline, even with --shallow.

    They must error out rather than silently restart the full pipeline (the
    pre-#330 behavior that let ``--shallow --start-at test --yes`` re-review
    and re-apply fixes).
    """
    with pytest.raises(SystemExit):
        _parse_args(["target", "--shallow", "--start-at", stage])
    assert "no mapping in the unified pipeline" in capsys.readouterr().err
