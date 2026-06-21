"""Tests for daydream.agent module-level state accessors."""

from daydream.agent import (
    get_non_interactive,
    is_environmental_failure,
    reset_state,
    set_non_interactive,
)


def test_non_interactive_defaults_false():
    reset_state()
    assert get_non_interactive() is False


def test_set_and_get_non_interactive():
    try:
        set_non_interactive(True)
        assert get_non_interactive() is True
    finally:
        reset_state()


def test_reset_state_clears_non_interactive():
    set_non_interactive(True)
    reset_state()
    assert get_non_interactive() is False


def test_is_environmental_failure_both_directions():
    environmental = [
        "The dev Postgres container is not running",
        "could not connect to server: Connection refused",
        "localhost:5432",
        "ECONNREFUSED",
    ]
    for output in environmental:
        assert is_environmental_failure(output) is True, output

    ordinary = [
        "AssertionError: assert 1 == 2",
        "1 failed, 3 passed",
        "ValueError: bad input",
    ]
    for output in ordinary:
        assert is_environmental_failure(output) is False, output
