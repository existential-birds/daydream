import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

import daydream.extensions as ext
from daydream.benchmark.cli import _build_benchmark_parser
from daydream.cli import _parse_args
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins

ROOT = Path(__file__).resolve().parents[1]


def test_readme_run_examples_parse() -> None:
    readme = (ROOT / "README.md").read_text()
    # the "common commands" code fence — the required run examples
    section = readme.split("Use the common commands for the common tasks:", 1)[1]
    fence = section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    lines = [
        line.strip()
        for line in fence.splitlines()
        if line.strip().startswith("daydream")
    ]
    assert lines, "the common-commands fence documents no daydream command"
    # each documented run example must be accepted by the production parser
    for line in lines:
        tokens = line.split()[1:]  # drop the leading 'daydream' verb
        if "#" in tokens:  # drop any inline comment
            tokens = tokens[: tokens.index("#")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            _parse_args(tokens)  # parses without SystemExit


def test_benchmark_objective_aggregate_parse() -> None:
    p = _build_benchmark_parser()                    # production parser
    p.parse_args(["objective", "./ws", "--run-id", "run-abc123", "--json", "-"])
    p.parse_args(["aggregate", "./suite.json", "--json", "-"])


def test_extensions_doc_claims_only_exposed() -> None:
    doc = (ROOT / "docs" / "extensions.md").read_text()
    reg = Registry()
    register_builtins(reg)
    # every public API symbol the doc's table names must exist in production
    table = doc.split("### Public API symbols")[1].split("### Discovery order")[0]
    for row in table.splitlines():
        if row.strip().startswith("| `"):
            symbol = row.split("`")[1]
            assert hasattr(ext, symbol), f"doc claims unexposed symbol {symbol!r}"
    # every prompt the doc's table names must be registered
    prompts = doc.split("### Prompts")[1].split("### Renderers")[0]
    for row in prompts.splitlines():
        if row.strip().startswith("| `"):
            name = row.split("`")[1]
            assert name in reg.prompt_names(), f"doc claims unexposed prompt {name!r}"


def test_help_exposes_native_surface(capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("--help", "--help-all"):
        with pytest.raises(SystemExit):
            _parse_args([flag])
        out = capsys.readouterr().out
        assert "--review-profile" in out and "--stack" in out
        assert "feedback" not in out and "--skill" not in out


# ---------------------------------------------------------------------------
# P18/#1156 observability field contract (docs/observability-fields.md +
# docs/observability.md + readback/replay tooling)
# ---------------------------------------------------------------------------


def _observability_fields() -> str:
    return (ROOT / "docs" / "observability-fields.md").read_text(encoding="utf-8")

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
    # The machine-readable subset must stay aligned with the markdown matrix:
    # every field name the subset asserts must appear in the expanded doc.
    doc = _observability_fields()
    for key in matrix["identity_metadata_keys"]:
        assert key in doc, f"matrix subset identity key {key!r} missing from observability-fields.md"

