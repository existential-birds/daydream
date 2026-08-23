"""Hermetic suite for the `daydream benchmark run` supervisor (issue #781).

Task 1: parser + dispatch routing.
Task 2: fail-closed preflight checks.
"""
import json
from pathlib import Path

from daydream.benchmark.cli import _build_benchmark_parser, _handle_benchmark_command


# ---------------------------------------------------------------------------
# shared hermetic fixtures (Tasks 2-8)
# ---------------------------------------------------------------------------


def _ws(tmp_path, **privacy):
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    (ws / "harbor").mkdir()
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps({"schema_version": 1, "cases": {}}))
    (ws / "harbor" / "harbor-job.yaml").write_text("jobs_dir: jobs\n")
    (ws / "harbor" / "harbor-oracle.yaml").write_text("jobs_dir: jobs\n")
    p = {"classification": "confidential", "reviewer_data": "source_snapshot",
         "reviewer_allowed_hosts": ["review.example"], "judge_data": "finding_text_and_location_only",
         "judge_allowed_hosts": ["127.0.0.1"], "archive": "disabled", "uploads": "disabled"}
    p.update(privacy)
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com", "repository": "OWNER/REPO",
                   "repository_id": None, "visibility": "unresolved"},
        "privacy": p, "pull_requests": [], "cases": []}))
    return ws


def _env(**over):
    base = {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
            "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9", "DAYDREAM_JUDGE_API_KEY": "k",
            "DAYDREAM_REVIEW_MODEL": "rm", "DAYDREAM_REVIEW_BASE_URL": "http://review.example"}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Task 1: parser + dispatch
# ---------------------------------------------------------------------------


def test_benchmark_parser_has_run_subcommand():
    parser = _build_benchmark_parser()
    args = parser.parse_args(["run", "/ws", "--oracle", "--yes"])
    assert args.subcommand == "run"
    assert args.dir == Path("/ws")
    assert args.oracle is True and args.yes is True
    plain = parser.parse_args(["run", "/ws"])
    assert plain.oracle is False and plain.yes is False


def test_handle_benchmark_run_routes_to_supervisor(tmp_path, monkeypatch):
    import daydream.benchmark.harbor.run as run_mod

    captured = {}

    def fake_run_run(ws, *, oracle, yes, env):
        captured["ws"] = Path(ws)
        captured["oracle"] = oracle
        captured["yes"] = yes
        captured["env"] = env
        return 0

    monkeypatch.setattr(run_mod, "run_run", fake_run_run)
    code = _handle_benchmark_command(["run", str(tmp_path), "--yes"])
    assert code == 0
    assert captured["ws"] == tmp_path
    assert captured["yes"] is True and captured["oracle"] is False
    assert "DAYDREAM_REVIEW_MODEL" in captured["env"]  # env threaded through


# ---------------------------------------------------------------------------
# Task 2: fail-closed preflight checks
# ---------------------------------------------------------------------------


def _seed_calibration_receipt(ws):
    """Write a current calibration receipt matching ``_env()`` via the #872 writer."""
    from daydream.benchmark.harbor import calibrate

    sr = calibrate._load_judge_template()
    pairs = calibrate._load_fixture()
    receipt = calibrate._build_receipt(
        sr, pairs, _env(), passed=True, balanced_accuracy=1.0,
        confusion={"tp": 12, "fp": 0, "tn": 12, "fn": 0}, disagreements=[],
    )
    calibrate._write_receipt(ws, receipt)


def test_preflight_ok_when_all_checks_pass(tmp_path):
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    _seed_calibration_receipt(ws)
    errs = run_mod._preflight(ws, oracle=True, env=_env(), docker_ok=lambda: True)
    assert errs == []


def test_preflight_blocks_judge_host_outside_allowlist(tmp_path):
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True,
                              env=_env(DAYDREAM_JUDGE_BASE_URL="http://evil.example"), docker_ok=lambda: True)
    assert any("judge host" in e and "evil.example" in e for e in errs)


def test_preflight_blocks_reviewer_host_outside_allowlist(tmp_path):
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True,
                              env=_env(DAYDREAM_REVIEW_BASE_URL="http://other.example"), docker_ok=lambda: True)
    assert any("reviewer host" in e and "other.example" in e for e in errs)


def test_preflight_blocks_uploads_enabled(tmp_path):
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path, uploads="enabled"), oracle=True,
                              env=_env(), docker_ok=lambda: True)
    assert any("upload" in e for e in errs)


def test_preflight_blocks_unsupported_docker_allowlist(tmp_path):
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True, env=_env(), docker_ok=lambda: False)
    assert any("Docker allowlist" in e for e in errs)


def test_preflight_blocks_missing_calibration_receipt_for_oracle(tmp_path):
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True, env=_env(), docker_ok=lambda: True)
    assert any("calibration" in e for e in errs)  # runtime/calibration-receipt.json absent