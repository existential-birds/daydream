# tests/test_cli_bench_rejection.py
"""The ONE intentional 'daydream bench' rejection test (issue #785).

``daydream bench`` is the removed legacy verb. It must exit non-zero with a
clear error and must NOT fall through to the review path (``_first_verb``
would otherwise treat ``bench`` as a bare target path).
"""

import sys

import pytest

from daydream import cli


def test_cli_bench_is_rejected_not_routed(monkeypatch, capsys) -> None:
    # If the rejection is missing, `_first_verb(["bench"])` returns "review"
    # and main() calls `_parse_args`; fail loudly if that happens.
    def _boom(argv):
        raise AssertionError("bench fell through to the review path")

    monkeypatch.setattr(cli, "_parse_args", _boom)
    monkeypatch.setattr(sys, "argv", ["daydream", "bench"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "no longer a command" in err
    assert "daydream benchmark" in err  # points at the replacement
