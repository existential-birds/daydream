"""Run the packaged judge in an isolated subprocess against a loopback server.

Only the candidate artifact crosses from the host; credentials, source,
reviewer configuration, and other agent outputs must remain private.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

_TEMPLATES_TESTS = (Path(__file__).resolve().parents[1]
    / "daydream" / "benchmark" / "harbor" / "templates" / "tests"
)

_SENTINELS = {"GH_TOKEN": "GH_TOKEN_SENTINEL_9f3a", "GITHUB_TOKEN": "GITHUB_TOKEN_SENTINEL_7b2c",
    "HF_TOKEN": "HF_TOKEN_SENTINEL_1a4d", "DAYDREAM_APP_PRIVATE_KEY": "APP_PRIVATE_KEY_SENTINEL_0c8e",
}

_JUDGE_KEY = "sk-or-isolation-only-7f1e"


class _JudgeServer(HTTPServer):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.posted_bodies: list[bytes] = []


class _Judge(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        cast(Any, self.server).posted_bodies.append(body)  # observed by the leakage assertions below
        resp = json.dumps({"choices": [{"message": {"content":
            '{"match": true, "confidence": 1.0, "reasoning": "identical"}'}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a: Any) -> None: pass


def _serve() -> _JudgeServer:
    srv = _JudgeServer(("127.0.0.1", 0), _Judge)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _write_verifier_metadata(verifier_dir: Path) -> None:
    # Bind both entrypoints to the shipped gold and its case-x oracle fixture.
    gold_bytes = (verifier_dir / "golden-review.json").read_bytes()
    (verifier_dir / "verifier-metadata.json").write_text(json.dumps({
        "schema_version": 1, "case_id": "case-x", "source_case_id": "case-x", "base_ref": "base", "head_ref": "head",
        "template_version": "1", "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
    }))


def test_test_sh_runs_copied_bundle_from_unrelated_cwd_without_host_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped test.sh scores from any cwd; only the candidate crosses from the host.

    Host credentials, source, reviewer configuration and agent outputs never reach
    the reward files, verifier output or judge request bodies, and stay unmodified.
    """
    for name, val in _SENTINELS.items():
        monkeypatch.setenv(name, val)
    private = {tmp_path / "secrets/reviewer.toml": "token = 'REVIEWER_SECRET_SENTINEL_5d91'\n",
               tmp_path / "source/main.py": "# PRIVATE_SOURCE_SENTINEL_2e4f\n",
               tmp_path / "agent_outputs/trajectory.json": '{"output": "AGENT_OUTPUT_SENTINEL_6b3a"}\n'}
    for path, text in private.items():
        path.parent.mkdir(parents=True)
        path.write_text(text)
    verifier_dir = tmp_path / "copied bundle" / "tests"
    verifier_dir.mkdir(parents=True)
    unrelated_cwd = tmp_path / "unrelated cwd"
    unrelated_cwd.mkdir()
    for name in ("test.sh", "score_review.py", "judge_prompt.md", "golden-review.json"):
        shutil.copy2(_TEMPLATES_TESTS / name, verifier_dir / name)
    shutil.copy2(_TEMPLATES_TESTS.parents[1] / "verifier_core.py", verifier_dir / "verifier_core.py")
    _write_verifier_metadata(verifier_dir)
    artifact_path = tmp_path / "artifacts" / "review.json"
    artifact_path.parent.mkdir()
    shutil.copy2(_TEMPLATES_TESTS.parent / "solution" / "golden-review.json", artifact_path)
    out_dir = tmp_path / "verifier-out"

    # Relocate only the copied script's image-specific log paths. Keep its
    # interpreter and bare import intact so the caller cwd cannot be masked.
    source_script = _TEMPLATES_TESTS / "test.sh"
    source_text = source_script.read_text()
    for required in ("/logs/artifacts/review.json", "/logs/verifier/reward.json",
        "python3 -c", "from verifier_core import validate_candidate_artifact", "python3 score_review.py",
    ):
        assert required in source_text
    copied_script = verifier_dir / "test.sh"
    copied_text = source_text.replace("/logs/artifacts/review.json", str(artifact_path)).replace(
        "/logs/verifier/reward.json", str(out_dir / "reward.json")
    )
    copied_script.write_text(copied_text)
    assert copied_script.read_text().replace(str(artifact_path), "/logs/artifacts/review.json").replace(
        str(out_dir / "reward.json"), "/logs/verifier/reward.json"
    ) == source_text

    srv = _serve()
    try:
        env = {"PATH": os.pathsep.join((str(Path(sys.executable).parent), os.environ.get("PATH", ""))),
            "DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
            "DAYDREAM_JUDGE_API_KEY": _JUDGE_KEY, "DAYDREAM_JUDGE_BASE_URL": f"http://127.0.0.1:{srv.server_port}",
            "DAYDREAM_JUDGE_ALLOWED_HOSTS": "127.0.0.1", "DAYDREAM_JUDGE_ARTIFACT_PATH": str(artifact_path),
            "DAYDREAM_JUDGE_OUT_PATH": str(out_dir),
        }  # No PYTHONPATH/PYTHONHOME: python3 -c must find verifier_core via cwd.
        proc = subprocess.run(
            [str(copied_script)], cwd=unrelated_cwd, env=env, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    finally:
        srv.shutdown()
        srv.server_close()

    assert "candidate artifact valid" in proc.stdout
    assert "verifier: reward.json written" in proc.stdout
    assert srv.posted_bodies
    reward = json.loads((out_dir / "reward.json").read_text())
    assert reward["reward"] == 1.0
    assert reward["verifier_error"] == 0
    assert source_script.read_text() == source_text
    blob = ((out_dir / "reward.json").read_text() + (out_dir / "reward-details.json").read_text()
            + proc.stdout + proc.stderr)
    judge_blob = b"".join(srv.posted_bodies).decode("utf-8", errors="replace")
    for sentinel in [*_SENTINELS.values(), "REVIEWER_SECRET_SENTINEL_5d91", "PRIVATE_SOURCE_SENTINEL_2e4f",
                     "AGENT_OUTPUT_SENTINEL_6b3a", _JUDGE_KEY]:
        assert sentinel not in blob and sentinel not in judge_blob
    assert {path: path.read_text() for path in private} == private
