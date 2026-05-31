"""Tests for daydream.agent module-level state accessors."""

from daydream.agent import (
    get_non_interactive,
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
