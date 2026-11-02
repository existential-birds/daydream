"""Deep-mode CLI validation tests.

Deep is the default; ``--shallow`` opts into the single-stack flow.
"""

import pytest

from daydream.commands.review import _parse_args


def test_resume_stage_validation_rejects_mode_mismatches() -> None:
    """Deep resume stages are invalid with --shallow; legacy shallow-loop stages are invalid everywhere."""
    for argv in (["target", "--shallow", "--start-at", "per-stack"], ["target", "--start-at", "test"]):
        with pytest.raises(SystemExit):
            _parse_args(argv)
