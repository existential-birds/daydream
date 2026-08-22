"""Real separate-process/environment isolation test for the judge verifier.

Runs the actual templates/tests/score_review.py entrypoint in a subprocess with
a whitelisted env + a local loopback judge server, and proves the verifier
cannot see host credentials, source, reviewer config, or agent outputs beyond
the candidate artifact.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_TEMPLATES_TESTS = (
    Path(__file__).resolve().parents[1]
    / "daydream" / "benchmark" / "harbor" / "templates" / "tests"
)

_SENTINELS = {
    "GH_TOKEN": "GH_TOKEN_SENTINEL_9f3a",
    "GITHUB_TOKEN": "GITHUB_TOKEN_SENTINEL_7b2c",
    "HF_TOKEN": "HF_TOKEN_SENTINEL_1a4d",
    "DAYDREAM_APP_PRIVATE_KEY": "APP_PRIVATE_KEY_SENTINEL_0c8e",
}

_JUDGE_KEY = "sk-or-isolation-only-7f1e"


class _Judge(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"choices": [{"message": {"content":
            '{"match": true, "confidence": 1.0, "reasoning": "identical"}'}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass


def _serve() -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", 0), _Judge)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_entrypoint_in_isolation_cannot_see_secrets_or_source(tmp_path, monkeypatch) -> None:
    # host workspace carries credentials + source + reviewer config + agent outputs
    for name, val in _SENTINELS.items():
        monkeypatch.setenv(name, val)
    secret_file = tmp_path / "secrets" / "reviewer.toml"
    secret_file.parent.mkdir(parents=True)
    secret_file.write_text("token = 'REVIEWER_SECRET_SENTINEL_5d91'\n")
    source_file = tmp_path / "source" / "main.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# PRIVATE_SOURCE_SENTINEL_2e4f\n")
    agent_file = tmp_path / "agent_outputs" / "trajectory.json"
    agent_file.parent.mkdir(parents=True)
    agent_file.write_text('{"output": "AGENT_OUTPUT_SENTINEL_6b3a"}\n')
    pre = {p: p.read_bytes() for p in (secret_file, source_file, agent_file)}

    # isolate the verifier in its own env dir
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    for rel in ("score_review.py", "verifier_core.py", "judge_prompt.md", "golden-review.json"):
        (verifier_dir / rel).write_bytes((_TEMPLATES_TESTS / rel).read_bytes())
    # task-binding metadata: run_verifier binds the candidate to the immutable
    # verifier-metadata.json beside the gold (case id + base/head refs + digest)
    gold_bytes = (verifier_dir / "golden-review.json").read_bytes()
    (verifier_dir / "verifier-metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "case_id": "case-x",
        "source_case_id": "case-x",
        "base_ref": "base",
        "head_ref": "head",
        "template_version": "1",
        "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
    }))
    artifact_path = tmp_path / "artifacts" / "review.json"
    artifact_path.parent.mkdir()
    oracle = Path(_TEMPLATES_TESTS / ".." / "solution" / "golden-review.json").resolve()
    artifact_path.write_bytes(oracle.read_bytes())
    out_dir = tmp_path / "verifier-out"

    srv = _serve()
    try:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(verifier_dir),            # bare `import verifier_core` resolves here
            "DAYDREAM_JUDGE_PROVIDER": "openai-compatible",
            "DAYDREAM_JUDGE_MODEL": "m",
            "DAYDREAM_JUDGE_API_KEY": _JUDGE_KEY,
            "DAYDREAM_JUDGE_BASE_URL": f"http://127.0.0.1:{srv.server_port}",
            "DAYDREAM_JUDGE_ALLOWED_HOSTS": "127.0.0.1",
            "DAYDREAM_JUDGE_ARTIFACT_PATH": str(artifact_path),
            "DAYDREAM_JUDGE_OUT_PATH": str(out_dir),
        }  # whitelist: NONE of the host sentinels are inherited
        proc = subprocess.run([sys.executable, "score_review.py"], cwd=verifier_dir,
                              env=env, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
    finally:
        srv.shutdown()

    rj = json.loads((out_dir / "reward.json").read_text())
    assert rj["verifier_error"] == 0 and rj["reward"] == 1  # judged orbitally, not an error exit
    artifacts_blob = ((out_dir / "reward.json").read_text() + (out_dir / "reward-details.json").read_text()
                      + proc.stdout + proc.stderr)
    for sentinel in list(_SENTINELS.values()) + ["REVIEWER_SECRET_SENTINEL_5d91", "PRIVATE_SOURCE_SENTINEL_2e4f",
                                                 "AGENT_OUTPUT_SENTINEL_6b3a", _JUDGE_KEY]:
        assert sentinel not in artifacts_blob        # no credential/source/agent/agent-key leakage
    for p, digest in zip((secret_file, source_file, agent_file), pre.values()):
        assert p.read_bytes() == digest             # host files untouched (no writes outside out_dir)
