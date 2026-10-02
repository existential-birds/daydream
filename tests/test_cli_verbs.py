"""Pin verb-first routing and equivalent parsing of bare-target and explicit-review forms.

Empty argv and leading flags also select review.
"""

import pytest

from daydream.cli import _first_verb
from daydream.commands.improve import _parse_improve_args
from daydream.commands.review import _parse_args


def test_first_verb_routing() -> None:
    assert _first_verb(["/some/path"]) == "review"  # bare path → review shim
    assert _first_verb(["--comment", "/p"]) == "review"  # leading flag → review
    assert _first_verb([]) == "review"
    # The legacy feedback token falls through to the review shim.
    assert _first_verb(["feedback", "42", "--bot", "x"]) == "review"

@pytest.mark.parametrize("argv", [["/t"], ["review", "/t"]])
def test_bare_and_review_verb_parse_identically(argv: list[str]) -> None:
    cfg = _parse_args(argv)
    assert cfg.target == "/t" and cfg.output_mode == "loop"

def test_improve_verb_builds_improve_config() -> None:
    config = _parse_improve_args(["improve", "/tmp/x", "--effort", "deep", "--focus", "security"])
    assert (config.flow_name, config.improve_effort, config.improve_focus,) == ("improve", "deep", "security")

def test_improve_plan_subverb_parses_description() -> None:
    config = _parse_improve_args(["improve", "plan", "add rate limiting", "/tmp/x"])
    assert config.improve_plan_description == "add rate limiting"

def test_improve_rejects_unknown_effort(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_improve_args(["improve", "/tmp/x", "--effort", "extreme"])
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "invalid choice" in error
    assert "extreme" in error
