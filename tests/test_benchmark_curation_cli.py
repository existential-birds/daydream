"""Tests for the ``daydream benchmark curate --apply-gold`` non-TTY CLI path.

Covers real filesystem effects: the case YAML is rewritten atomically with mode
``0600``, a ready case edit clears attestation and reopens draft, ``--apply-gold``
leaves a ready-snapshot case draft and un-attested, and expected workspace errors
map to stderr + exit code ``1`` (no bare traceback). The in-process route is
exercised with no terminal mocks.
"""
import sys
from pathlib import Path

import pytest

from daydream.benchmark import cli
from tests.harness.fake_gh import FakeGh
from tests.test_benchmark_curation import _seed_ready_case


def test_curate_non_tty_keeps_guidance_and_exit_1(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case, _ = _seed_ready_case(tmp_path, fake_gh, lines=2)
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    rc = cli._handle_benchmark_command(["curate", str(ws), "--case", case])
    assert rc == 1 and "apply-gold" in capsys.readouterr().err.lower()
    routed: list[tuple[Path, str]] = []

    def tui(root: Path, case_id: str) -> int:
        routed.append((root, case_id))
        return 0

    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr("daydream.benchmark.curate_tui.run_curate_tui", tui)
    assert cli._handle_benchmark_command(["curate", str(ws), "--case", case]) == 0
    assert routed == [(ws, case)]
    case_file = ws / "cases" / f"{case}.yaml"
    before = case_file.read_bytes()
    bad = tmp_path / "gold.yaml"
    bad.write_text("findings:\n  - title: x\n    body: b\n    source_ids: [github:review:404]\n")
    assert cli._handle_benchmark_command(["curate", str(ws), "--case", case, "--apply-gold", str(bad)]) == 1
    assert "Traceback" not in capsys.readouterr().err and case_file.read_bytes() == before
    monkeypatch.undo()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert cli._is_interactive_tty() is True
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert cli._is_interactive_tty() is False
