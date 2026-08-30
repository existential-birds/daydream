"""Tests for the ``daydream benchmark curate --apply-gold`` non-TTY CLI path.

Covers real filesystem effects: the case YAML is rewritten atomically with mode
``0600``, a ready case edit clears attestation and reopens draft, ``--apply-gold``
leaves a ready-snapshot case draft and un-attested, and expected workspace errors
map to stderr + exit code ``1`` (no bare traceback). The in-process route is
exercised with no terminal mocks.
"""
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream.benchmark.storage import load_yaml_strict
from tests.harness.fake_gh import FakeGh
from tests.test_benchmark_curation import _seed_ready_case


def test_cli_curate_apply_gold_writes_0600_and_never_ready(
    tmp_path: Path,
    fake_gh: FakeGh,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daydream.benchmark import curation as cu
    from daydream.benchmark.cli import _handle_benchmark_command

    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    cand = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    frag_path = tmp_path / "gold.yaml"
    frag_path.write_text(yaml.safe_dump({
        "findings": [{"title": cand["title"], "body": cand["body"], "severity": None,
                      "location": cand["location"], "source_ids": [cand["source_id"]]}],
        "exclusions": [], "case_exclusion": None, "clean": False,
    }, sort_keys=False))

    rc = _handle_benchmark_command(
        ["curate", str(ws), "--case", case_id, "--apply-gold", str(frag_path)])
    assert rc == 0
    out = capsys.readouterr()
    assert "ready" not in out.err                      # no error printed

    case_path = ws / "cases" / f"{case_id}.yaml"
    assert oct(case_path.stat().st_mode & 0o777) == "0o600"   # atomic 0600
    raw = load_yaml_strict(case_path)
    assert raw["curation"]["state"] == "draft" and raw["curation"]["snapshot_attested"] is False
    assert raw["curation"]["findings"][0]["provenance"]["kind"] == "historical"


def test_cli_curate_apply_gold_malformed_fragment_clean_exit(
    tmp_path: Path,
    fake_gh: FakeGh,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed fragment (missing required keys) maps to exit 1, no traceback."""
    from daydream.benchmark.cli import _handle_benchmark_command

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=2)
    frag_path = tmp_path / "gold.yaml"
    frag_path.write_text(yaml.safe_dump({
        "findings": [{"body": "missing title"}],      # dereferences frag['title']
        "exclusions": [{"reason": "other"}],           # dereferences exc['source_id']
        "case_exclusion": {"note": "no reason"},       # dereferences case_exclusion['reason']
    }, sort_keys=False))

    rc = _handle_benchmark_command(
        ["curate", str(ws), "--case", case_id, "--apply-gold", str(frag_path)])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_curate_on_tty_dispatches_to_tui(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.benchmark import cli

    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    called: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        "daydream.benchmark.curate_tui.run_curate_tui",
        lambda root, cid=None, **k: called.update(root=str(root), cid=cid) or 0,
    )
    rc = cli._handle_benchmark_command(["curate", str(ws), "--case", case_id])
    assert rc == 0 and called == {"root": str(ws), "cid": case_id}


def test_curate_non_tty_keeps_guidance_and_exit_1(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daydream.benchmark import cli
    ws, case, _ = _seed_ready_case(tmp_path, fake_gh, lines=2)
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    rc = cli._handle_benchmark_command(["curate", str(ws), "--case", case])
    assert rc == 1 and "apply-gold" in capsys.readouterr().err.lower()


def test_is_interactive_tty_detects_stdin_and_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.benchmark import cli
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert cli._is_interactive_tty() is True
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert cli._is_interactive_tty() is False
