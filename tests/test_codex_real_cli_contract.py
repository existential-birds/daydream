"""The sanitized real Codex capture keeps truthful provenance; its maintenance script only writes candidates.

The separate sanitized codex 0.153.4 omission capture retains raw/sanitized digests. Its maintenance
script creates only an unpublished public candidate for exact-run corroboration and review.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from tests.harness.codex_replay import FIXTURES_DIR

REAL_GENERIC_TOOL_FAILURE = "real/generic-tool-failure.jsonl"
REAL_GENERIC_TOOL_FAILURE_META = "real/generic-tool-failure.meta.json"


def test_codex_capture_provenance_is_sanitized_and_candidates_never_replace_it(tmp_path: Path) -> None:
    """The genuine public capture is immutable apart from its declared path redaction.

    A fake external CLI proves the maintenance script's candidate capture is non-publishing and public-only.
    """
    capture_script = Path(__file__).parents[1] / "scripts" / "capture-codex-generic-tool-failure.sh"
    fixture_path = FIXTURES_DIR / REAL_GENERIC_TOOL_FAILURE
    metadata_path = FIXTURES_DIR / REAL_GENERIC_TOOL_FAILURE_META
    fixture_bytes = fixture_path.read_bytes()
    metadata = json.loads(metadata_path.read_text())
    records = [json.loads(line) for line in fixture_bytes.splitlines()]

    assert capture_script.is_file() and os.access(capture_script, os.X_OK)
    assert hashlib.sha256(fixture_bytes).hexdigest() == metadata["fixture_sha256_sanitized"]
    assert metadata["capture_sha256_raw"] == ("896389a5da72401e74ee50324bedd745c513cb7468fa5dcb0fbf682e4b125683")
    assert metadata["capture_sha256_raw"] != metadata["fixture_sha256_sanitized"]
    assert metadata["fixture_sanitized"] is True
    assert metadata["sanitizations"] == [{
            "field": "item.message", "occurrences": 1, "replacement": "/Users/[REDACTED_USER]/.codex/config.toml",
            "reason": "personal_home_path",
        }
    ]
    assert [record["type"]
        if record["type"] != "item.completed"
        else f"item.completed/{record['item']['type']}"
        for record in records
    ] == ["thread.started", "item.completed/error", "turn.started", "item.completed/error",
        "item.completed/agent_message", "item.completed/agent_message", "turn.completed",
    ]
    assert metadata["ordered_public_shape"] == [
        "thread.started", "item.completed/error", "turn.started", "item.completed/error",
        "item.completed/agent_message", "item.completed/agent_message", "turn.completed",
    ]
    assert metadata["cli_version"] == "0.153.4"
    assert metadata["captured_at_utc"] == "2026-09-06T02:31:50Z"
    assert metadata["capture_working_directory"] == "disposable_git_repo"
    assert metadata["public_flags"] == ["--json", "--sandbox", "read-only", "--enable", "code_mode"]
    assert metadata["exit_code"] == 0
    assert metadata["public_generic_function_items_exposed"] is False
    assert metadata["private_rollout_correlated"] is True
    assert metadata["private_custom_tool_calls"] == [
        {"name": "exec", "count": 1, "matching_outputs": 1, "error_class": "TypeError"}
    ]
    assert metadata["probe_prompt"] == (
        "This is a transport-contract probe. You MUST invoke the functions.exec tool exactly once "
        "with this exact JavaScript body: const r = await tools.wait({cell_id:\"definitely-missing-cell\","
        "yield_time_ms:250}); text(r); Do not call any other tool. Do not claim success unless you "
        "receive the tool output. Afterward report whether the nested wait call was visible to you "
        "and include its error category without inventing it."
    )

    public_items = [record["item"] for record in records if "item" in record]
    assert [item["type"] for item in public_items] == ["error", "error", "agent_message", "agent_message"]
    for item in public_items:
        assert not ({"name", "call_id", "arguments", "output"} & item.keys())

    fixture_text = fixture_bytes.decode()
    metadata_text = metadata_path.read_text()
    assert "/Users/ka" not in fixture_text
    assert "/Users/ka" not in metadata_text
    assert "definitely-missing-cell" not in fixture_text
    for forbidden_key in {
        "private_rollout_bytes", "private_session_path", "home_path", "environment_context", "credentials", "call_id",
        "arguments", "output_text",
    }:
        assert forbidden_key not in metadata

    committed_fixture = fixture_path
    committed_metadata = metadata_path
    before_fixture = committed_fixture.read_bytes()
    before_metadata = committed_metadata.read_bytes()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        "  printf '%s\\n' 'codex-cli 9.9.9'\n"
        "  exit 0\n"
        "fi\n"
        # Read stdin before replying; early exit can race the producer and yield SIGPIPE under
        # pipefail.
        "IFS= read -r probe_prompt\n"
        'printf \'%s\\n\' "$probe_prompt" > "$FAKE_CODEX_PROBE_PATH"\n'
        'if [[ "${FAKE_CODEX_EXPOSE_PAIR:-}" == "1" ]]; then\n'
        "  printf '%s\\n' \\\n"
        "    '{\"type\":\"item.completed\",\"item\":{\"type\":\"error\"}}' \\\n"
        "    '{\"type\":\"item.completed\",\"item\":{"
        "\"type\":\"custom_tool_call\"}}' \\\n"
        "    '{\"type\":\"turn.completed\"}'\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \\\n"
        "  '{\"type\":\"thread.started\",\"thread_id\":\"candidate-thread\"}' \\\n"
        "  '{\"type\":\"item.completed\",\"item\":{\"id\":\"warning\","
        "\"type\":\"error\",\"message\":\"see /Users/fake-person/.codex/config.toml\"}}' \\\n"
        "  '{\"type\":\"turn.started\"}' \\\n"
        "  '{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":1,"
        "\"output_tokens\":1}}'\n"
    )
    fake_codex.chmod(0o755)
    candidate_root = tmp_path / "candidates"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["DAYDREAM_CODEX_CAPTURE_ROOT"] = str(candidate_root)
    probe_path = tmp_path / "received-probe.txt"
    env["FAKE_CODEX_PROBE_PATH"] = str(probe_path)
    env.pop("DAYDREAM_CODEX_PRIVATE_CORROBORATED", None)
    env.pop("DAYDREAM_CODEX_PRIVATE_ERROR_CLASS", None)

    captured = subprocess.run(
        [str(capture_script)], cwd=Path(__file__).parents[1], env=env, capture_output=True, text=True, check=False,
    )

    assert captured.returncode == 0, captured.stderr
    candidates = [path for path in candidate_root.iterdir() if not path.name.startswith(".")]
    assert len(candidates) == 1
    candidate_fixture = candidates[0] / "public.jsonl"
    candidate_metadata = candidates[0] / "public.meta.json"
    assert candidate_fixture.is_file() and candidate_metadata.is_file()
    candidate = json.loads(candidate_metadata.read_text())
    assert candidate["cli_version"] == "9.9.9"
    assert candidate["publication_status"] == "candidate_unpublished"
    assert probe_path.read_text() == candidate["probe_prompt"] + "\n"
    assert candidate["review_requirements"] == ["privately_correlate_this_exact_candidate_run_before_fixture_update"]
    assert not [key for key in candidate if key.startswith("private_")]
    assert hashlib.sha256(candidate_fixture.read_bytes()).hexdigest() == candidate["fixture_sha256_sanitized"]
    assert "/Users/fake-person" not in candidate_fixture.read_text()
    assert committed_fixture.read_bytes() == before_fixture
    assert committed_metadata.read_bytes() == before_metadata

    rejected_root = tmp_path / "rejected-candidates"
    env["DAYDREAM_CODEX_CAPTURE_ROOT"] = str(rejected_root)
    env["FAKE_CODEX_EXPOSE_PAIR"] = "1"
    rejected = subprocess.run(
        [str(capture_script)], cwd=Path(__file__).parents[1], env=env, capture_output=True, text=True, check=False,
    )
    assert rejected.returncode != 0
    assert "update the parser contract" in rejected.stderr
    assert not [path for path in rejected_root.iterdir() if not path.name.startswith(".")]
    assert committed_fixture.read_bytes() == before_fixture
    assert committed_metadata.read_bytes() == before_metadata
