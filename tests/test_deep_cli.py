"""Deep-mode CLI parsing tests (D-01..D-06).

Every test is xfail(strict=True) until Wave 1 plan 05-02 implements the
``--deep`` flag and ``--start-at`` deep stages in ``daydream.cli``.
"""

import pytest

from daydream.cli import _parse_args


@pytest.mark.xfail(reason="Wave 1 plan 05-02 not yet implemented", strict=True)
def test_deep_flag_parsed() -> None:
    """D-01: --deep sets config.deep=True."""
    args = _parse_args(["target", "--deep"])
    assert args.deep is True


@pytest.mark.xfail(reason="Wave 1 plan 05-02 not yet implemented", strict=True)
def test_no_stack_declaration_flags() -> None:
    """D-02: no --stack / --also-review / --skip-stack flags."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--deep", "--stack", "python"])


@pytest.mark.xfail(reason="Wave 1 plan 05-02 not yet implemented", strict=True)
@pytest.mark.parametrize("stage", ["ttt", "per-stack", "merge"])
def test_start_at_deep_stages_accepted(stage: str) -> None:
    """D-03: --start-at ttt|per-stack|merge accepted under --deep."""
    args = _parse_args(["target", "--deep", "--start-at", stage])
    assert args.start_at == stage


@pytest.mark.xfail(reason="Wave 1 plan 05-02 not yet implemented", strict=True)
def test_deep_rejects_start_at_parse() -> None:
    """D-04: --start-at parse rejected under --deep."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--deep", "--start-at", "parse"])


@pytest.mark.xfail(reason="Wave 1 plan 05-02 not yet implemented", strict=True)
@pytest.mark.parametrize("stage", ["ttt", "per-stack", "merge"])
def test_deep_stages_require_deep_flag(stage: str) -> None:
    """D-05: deep-only stages only legal with --deep."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--start-at", stage])


@pytest.mark.xfail(reason="Wave 1 plan 05-02 not yet implemented", strict=True)
@pytest.mark.parametrize("other_flag", [["--pr", "123"], ["--loop"], ["--ttt"], ["--review-only"]])
def test_deep_mutex(other_flag: list[str]) -> None:
    """D-06: --deep is mutually exclusive with --pr, --loop, --ttt, --review-only."""
    with pytest.raises(SystemExit):
        _parse_args(["target", "--deep", *other_flag])
