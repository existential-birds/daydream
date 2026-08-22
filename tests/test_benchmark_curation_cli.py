"""Tests for the ``daydream benchmark curate --apply-gold`` non-TTY CLI path.

Covers real filesystem effects: the case YAML is rewritten atomically with mode
``0600``, a ready case edit clears attestation and reopens draft, ``--apply-gold``
leaves a ready-snapshot case draft and un-attested, and expected workspace errors
map to stderr + exit code ``1`` (no bare traceback). The in-process route is
exercised with no terminal mocks.
"""

import yaml

from daydream.benchmark.storage import load_yaml_strict
from tests.test_benchmark_curation import _seed_ready_case


def test_cli_curate_apply_gold_writes_0600_and_never_ready(tmp_path, fake_gh, capsys):
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


def test_cli_curate_apply_gold_malformed_fragment_clean_exit(tmp_path, fake_gh, capsys):
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


def test_cli_curate_without_apply_gold_rejects_on_non_tty(tmp_path, fake_gh, capsys):
    from daydream.benchmark.cli import _handle_benchmark_command

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=2)
    rc = _handle_benchmark_command(["curate", str(ws), "--case", case_id])
    assert rc == 1
    assert "apply-gold" in capsys.readouterr().err.lower()
