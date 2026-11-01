"""Run the installed wheel's actual Harbor entrypoint in a networkless image."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from daydream.benchmark.harbor import entrypoint, verifier_core
from tests.harness.harbor_review import answer_review, assignment_from_request, forbid_github
from tests.harness.native_pi import _NativeProvider, native_provider_server


def main() -> None:
    config = Path("/tmp/isolated-pi")
    config.mkdir()
    fixture = _NativeProvider(config, choose=answer_review)
    with forbid_github() as attempts, native_provider_server(fixture) as url:
        (config / "models.json").write_text(json.dumps({"providers": {"openrouter": {
            "baseUrl": f"{url}/v1", "api": "openai-completions", "apiKey": "synthetic-loopback-only",
            "models": [{"id": "output-model", "reasoning": False, "input": ["text"],
                        "contextWindow": 131072, "maxTokens": 16384}]}}}))
        (config / "settings.json").write_text(json.dumps({"retry": {"enabled": False, "maxRetries": 0}}))
        rc = asyncio.run(entrypoint.main({
            "PATH": os.environ["PATH"], "LANG": "C.UTF-8", "DAYDREAM_REVIEW_BACKEND": "pi",
            "DAYDREAM_REVIEW_MODEL": "output-model", "DAYDREAM_REVIEW_API_KEY": "synthetic-loopback-only",
            "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api/v1", "DAYDREAM_REVIEW_CASE_ID": "case-package",
            "DAYDREAM_REVIEW_AGENT_DIR": str(config),
        }))
        assert rc == 0 and attempts == []
    candidates = verifier_core.validate_candidate_artifact(json.loads(Path("/logs/artifacts/review.json").read_text()))
    assert len(candidates) == 1 and candidates[0].path == "feature.py" and candidates[0].start_line == 2
    result_path, = Path("/logs/agent").glob("committed-*/committed-review.json")
    result = json.loads(result_path.read_text())
    immutable = json.loads(Path("/opt/daydream-input/immutable-request.json").read_text())
    assert result["immutable_input"] == immutable
    assert result["terminal_result"]["analysis_state"] == "complete"
    assert result["analyzed_revision"]["head_sha"] == immutable["materialization"]["head_sha"]
    trajectory = json.loads(Path("/logs/agent/trajectory.json").read_text())
    assert trajectory["extra"]["pr_repo"] == "o/r" and trajectory["extra"]["pr_number"] == 101
    assignments = [assignment_from_request(request) for request in fixture.requests]
    assert [assignment["kind"] for assignment in assignments] == ["discovery", "discovery", "challenge", "challenge"]
    for request in fixture.requests[::2]:
        assert not any(message["role"] in {"assistant", "tool"} for message in request["messages"])
    assert subprocess.check_output(["git", "-C", "/workspace/repo", "remote"]) == b""
    assert subprocess.check_output(["node", "--version"]).strip() == b"v24.18.1"
    assert subprocess.check_output(["pi", "--version"]).strip() == b"1.4.0"
    print(json.dumps({"packaged_native_review": "complete", "native_requests": len(fixture.requests)}))


if __name__ == "__main__":
    main()
