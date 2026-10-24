import contextlib
import hashlib
import io
import json
import re
import shlex
from pathlib import Path

import pytest

from daydream.benchmark.cli import _build_benchmark_parser
from daydream.commands.review import _parse_args
from tests.harness.git_helpers import git

ROOT = Path(__file__).resolve().parents[1]

def test_readme_run_examples_parse() -> None:
    readme = (ROOT / "README.md").read_text()
    # Parse the shipped review examples without depending on surrounding prose.
    fences = re.findall(r"```bash\n(.*?)```", readme, re.DOTALL)
    lines = [line for fence in fences for line in fence.splitlines() if line.startswith("daydream ")]
    assert lines, "README documents no review commands"
    for line in lines:
        tokens = shlex.split(line, comments=True)[1:]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _parse_args(tokens)  # parses without SystemExit

def test_benchmark_objective_aggregate_parse() -> None:
    p = _build_benchmark_parser()                    # production parser
    p.parse_args(["objective", "./ws", "--run-id", "run-abc123", "--json", "-"])
    p.parse_args(["aggregate", "./suite.json", "--json", "-"])

def test_agent_document_reference_paths_exist() -> None:
    # Relocated contracts must remain reachable from the guidance and skill entrypoints.
    # Local ignored tooling must not hide a broken link in a clean checkout.
    tracked = {ROOT / name for name in git(ROOT, "ls-files", "-z").split("\0") if name}
    shipped = tracked | {parent for path in tracked for parent in path.parents if parent.is_relative_to(ROOT)}
    paths = [ROOT / name for name in (
        "README.md", "AGENTS.md", "CONTRIBUTING.md", ".github/workflows/README.md",
        ".agents/skills/test-audit/SKILL.md", "daydream/templates/workflows/README.md",
        "daydream/templates/workflows/single/README.md",
    )]
    paths.extend((ROOT / "docs").rglob("*.md"))
    for path in paths:
        links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text())
        for target in links:
            if "://" not in target:
                destination = (path.parent / target).resolve()
                assert destination in shipped and destination.exists(), (
                    f"broken reference {target!r} in {path.relative_to(ROOT)}"
                )

def test_help_exposes_native_surface(capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("--help", "--help-all"):
        with pytest.raises(SystemExit):
            _parse_args([flag])
        out = capsys.readouterr().out
        assert "--review-profile" in out and "--stack" in out
        assert "feedback" not in out and "--skill" not in out


def test_replay_manifest_pins_fixture_and_identity() -> None:
    manifest = json.loads((ROOT / "tests/fixtures/observability_contract/replay-manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "sanitized_protocol_replay"
    fixture_pin = manifest["fixture"]
    raw = (ROOT / fixture_pin["path"]).read_bytes()
    assert len(raw) == fixture_pin["bytes"]
    assert hashlib.sha256(raw).hexdigest() == fixture_pin["sha256"]
    identity = fixture_pin["identity"]
    assert identity["first_message_end_receipt_unix_ns"] == 1788690709621000000
    assert identity["duration_ns"] == 395332000000
    assert identity["normalized_input_tokens"] == 86936
    assert identity["reported_cost_usd"] == 0.00402781
    assert "https://github.com/earendil-works/pi-coding-agent.git" in manifest["public_repo_allowlist"]

def test_readback_matrix_subset_is_machine_readable() -> None:
    matrix = json.loads((ROOT / "tests/fixtures/observability_contract/readback-matrix.json").read_text())
    assert matrix["schema_version"] == 1
    assert matrix["contract_version"] == "94f432d"
    for section in ("honeyhive", "langsmith"):
        assert isinstance(matrix[section], dict)
    assert "_detectedAgent" in matrix["forbidden_vendor_fields"]
    assert matrix["expected_shape"]["sanitized_protocol_replay"]["duration_ns"] == 395332000000
