# tests/test_cli_verbs.py
"""Tests for verb-first dispatch and the default-``review`` shim.

``_first_verb`` is the pure routing primitive: it inspects the leading token
and decides which verb owns the rest of argv. A bare path, a leading flag, or
empty argv all fall through to the ``review`` golden path. ``_parse_argv_for_test``
drives the real ``_parse_args`` (the production RunConfig builder) so that the
bare-target and explicit-``review`` forms are proven to parse identically.
"""

import pytest

from daydream.cli import _first_verb, _parse_args, _parse_improve_args
from daydream.runner import RunConfig


def _parse_argv_for_test(argv: list[str]) -> RunConfig:
    """Build a RunConfig from argv through the production parser."""
    return _parse_args(argv)


def test_first_verb_routing() -> None:
    assert _first_verb(["feedback", "42", "--bot", "x"]) == "feedback"
    assert _first_verb(["/some/path"]) == "review"  # bare path → review shim
    assert _first_verb(["--comment", "/p"]) == "review"  # leading flag → review
    assert _first_verb([]) == "review"  # empty → review (interactive target prompt)


@pytest.mark.parametrize("argv", [["/t"], ["review", "/t"]])
def test_bare_and_review_verb_parse_identically(argv: list[str]) -> None:
    cfg = _parse_argv_for_test(argv)
    assert cfg.target == "/t" and cfg.output_mode == "loop"


def test_improve_verb_builds_improve_config() -> None:
    config = _parse_improve_args(
        ["improve", "/tmp/x", "--effort", "deep", "--focus", "security"]
    )
    assert (
        config.flow_name,
        config.improve_effort,
        config.improve_focus,
    ) == ("improve", "deep", "security")


def test_improve_plan_subverb_parses_description() -> None:
    config = _parse_improve_args(
        ["improve", "plan", "add rate limiting", "/tmp/x"]
    )
    assert config.improve_plan_description == "add rate limiting"


def test_improve_rejects_unknown_effort() -> None:
    with pytest.raises(SystemExit):
        _parse_improve_args(["improve", "/tmp/x", "--effort", "extreme"])
