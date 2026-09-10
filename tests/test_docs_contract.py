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


def _observability_doc() -> str:
    return (ROOT / "docs" / "observability.md").read_text(encoding="utf-8")


def test_observability_fields_doc_exists_and_pins_contract() -> None:
    doc = _observability_fields()
    # Version block + semantic/producer pins required by the plan's Matrix
    # contract (P18-plan.md Task 6 Step 3).
    assert "contract_version" in doc
    assert "94f432d" in doc
    for pin in ("1.44.0", "0.62.3", "0.5.1", "0.28.1", "4.14.2", "0.85.1"):
        assert pin in doc, f"missing dependency pin {pin} in observability-fields.md"
    assert "94f432d7126f5884d30a2cdde6f4e89908ebb6fd" in doc  # semconv commit pin


def test_observability_fields_doc_has_matrix_columns() -> None:
    doc = _observability_fields()
    for column in (
        "field name",
        "source backend/event",
        "source authority/provenance",
        "owning span",
        "type",
        "unit",
        "cardinality",
        "derivation",
        "completeness",
        "capture/redaction",
        "generic OTLP disposition",
        "HoneyHive canonical destination",
        "LangSmith native destination",
        "offline test node",
        "live evidence status",
        "applicability and omission reason",
    ):
        assert column in doc, f"matrix doc missing column {column!r}"


def test_observability_fields_doc_covers_aliases_and_removal_policy() -> None:
    doc = _observability_fields()
    lower = doc.lower()
    assert "alias" in lower
    assert "removal" in lower
    # The two documented compatibility aliases must be present.
    assert "reasoning_effort" in doc
    assert "gen_ai.system" in doc


def test_observability_fields_doc_has_unavailable_rows_per_backend() -> None:
    doc = _observability_fields()
    for token in (
        "agent id/description/version",
        "response id",
        "top-p",
        "seed",
        "provider endpoint",
        "ttft",
        "prefill",
        "reasoning duration",
        "modalities",
        "opaque",
    ):
        assert token in doc.lower(), f"missing unavailable-row token {token!r} in matrix doc"


def test_observability_fields_doc_covers_binding_decisions() -> None:
    doc = _observability_fields()
    lower = doc.lower()
    for token in (
        "invoke_agent",
        "structural_attempt",
        "generation_children",
        "billing owner",
        "single billing",
        "no double billing",
        "395332000000",
        "message_end",
        "native",
        "full mode",
        "metadata mode",
        "no model-child",
        "interrupt",
    ):
        assert token in lower, f"matrix doc missing binding-decision token {token!r}"


def test_readback_verifier_deadline_contract_documented() -> None:
    """docs/observability.md must document the verifier's ONE immutable deadline."""
    doc = _observability_doc()
    lower = doc.lower()
    for token in (
        "verify_observability_readback.py",
        "immutable",
        "deadline",
        "0.35",
        "trust_env",
        "no redirects",
        "timeout",
        "ui",
    ):
        assert token in lower, f"observability.md missing verifier-contract token {token!r}"
    assert "readback" in lower


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


def test_observability_fields_doc_exposes_replay_and_readback_commands() -> None:
    doc = _observability_fields()
    for token in (
        "replay_observability_acceptance.py",
        "verify_observability_readback.py",
        "sanitized_protocol_replay",
    ):
        assert token in doc, f"matrix doc missing operator-tool token {token!r}"


def test_observability_docs_state_structural_and_generation_span_contract() -> None:
    """Structural attempts vs generation model spans; INTERNAL vs CLIENT."""
    doc = _observability_fields()
    lower = doc.lower()
    for token in (
        "structural attempt",
        "invoke_agent",
        "internal",
        "client",
        "generation model spans",
        "claude/codex/osprey attempts stay structural",
        "full mode only",
    ):
        assert token in lower, f"matrix doc missing span-contract token {token!r}"


def test_observability_docs_cover_accounting_and_ambient_isolation() -> None:
    doc = _observability_fields()
    lower = doc.lower()
    for token in (
        "complete/partial",
        "no double billing",
        "no invented usage",
        "ambient",
        "notebook",
        "zero global",
        "effective configuration admission contract",
        "redacted",
        "resource precedence",
        "force_flush",
        "partial success never retried",
        "four backend",
        "opaqu",
    ):
        assert token in lower, f"matrix doc missing accounting/isolation token {token!r}"


def test_observability_docs_separate_real_and_sanitized_commands() -> None:
    doc = _observability_fields()
    lower = doc.lower()
    assert "representative_real_run" in lower or "representative real run" in lower
    assert "sanitized_protocol_replay" in lower
    assert "no private honeyhive fields" in lower
    assert "underscore" in lower  # underscore-private fields never written


def test_observability_docs_pin_readback_deadline_semantics() -> None:
    doc = _observability_fields()
    lower = doc.lower()
    for token in (
        "one immutable",
        "monotonic deadline",
        "fixed redacted",
        "timeout",
        "cleanup",
        "ui not inspected",
        "ui limitation",
    ):
        assert token in lower, f"matrix doc missing deadline-semantics token {token!r}"
