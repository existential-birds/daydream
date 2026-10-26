"""Pin upstream GenAI fixture provenance and bytes independently of runtime span behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TESTS_DIR = Path(__file__).resolve().parent
SEMCONV_DIR = TESTS_DIR / "fixtures" / "observability_semconv"
CONTRACT_DIR = TESTS_DIR / "fixtures" / "observability_contract"

SEMCONV_COMMIT = "94f432d7126f5884d30a2cdde6f4e89908ebb6fd"



# Manifest byte verification


def _load_manifest() -> dict[str, Any]:
    manifest_path = CONTRACT_DIR / "manifest.json"
    assert manifest_path.is_file(), "contract manifest missing"
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data

def test_manifest_pins_expected_commit_and_files() -> None:
    manifest = _load_manifest()
    assert manifest["source"]["commit"] == SEMCONV_COMMIT
    assert manifest["source"]["type"] == "git_commit"
    assert manifest["source"]["repo"] == "open-telemetry/semantic-conventions-genai"
    files = manifest["files"]
    expected = {"registry.yaml", "spans.yaml", "gen-ai-input-messages.json", "gen-ai-output-messages.json",
        "gen-ai-system-instructions.json", "gen-ai-tool-definitions.json",
    }
    assert expected <= set(files), f"manifest missing entries: {expected - set(files)}"
    for name, entry in files.items():
        raw = (SEMCONV_DIR / name).read_bytes()
        assert entry["bytes"] == len(raw), f"{name}: byte length drifted"
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"{name}: sha256 drifted"
        assert (entry["url"].startswith("https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/")
            and SEMCONV_COMMIT in entry["url"]
        ), f"{name}: url not pinned to {SEMCONV_COMMIT[:12]}"
