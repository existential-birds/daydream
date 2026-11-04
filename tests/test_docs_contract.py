"""Static documentation and fixture contracts that no runtime path reads."""

import hashlib
import json
import re
from pathlib import Path

from tests.harness.git_helpers import git

ROOT = Path(__file__).resolve().parents[1]


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


def test_observability_contract_fixtures_are_pinned_and_machine_readable() -> None:
    fixtures = ROOT / "tests/fixtures/observability_contract"
    manifest = json.loads((fixtures / "replay-manifest.json").read_text())
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

    matrix = json.loads((fixtures / "readback-matrix.json").read_text())
    assert matrix["schema_version"] == 1
    assert matrix["contract_version"] == "94f432d"
    for section in ("honeyhive", "langsmith"):
        assert isinstance(matrix[section], dict)
    assert "_detectedAgent" in matrix["forbidden_vendor_fields"]
    assert matrix["expected_shape"]["sanitized_protocol_replay"]["duration_ns"] == identity["duration_ns"]
