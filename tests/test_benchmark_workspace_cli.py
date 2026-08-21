import subprocess


def test_benchmark_help_lists_subcommands():
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "benchmark", "--help"], capture_output=True, text=True  # noqa: S607 - trusted command
    )
    assert r.returncode == 0 and "init" in r.stdout and "status" in r.stdout and "validate" in r.stdout


def test_benchmark_init_status_validate_roundtrip(tmp_path):
    ws = tmp_path / "ws"
    r = subprocess.run(  # noqa: S603
        [
            "daydream", "benchmark", "init", str(ws),
            "--repo", "OWNER/REPO",
            "--reviewer-host", "api.anthropic.com",
            "--judge-host", "api.anthropic.com",
        ],
        capture_output=True, text=True,  # noqa: S607
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "confidential" in r.stdout  # prints privacy classification
    assert "api.anthropic.com" in r.stdout  # prints egress boundary
    assert (ws / "benchmark.yaml").exists()

    r2 = subprocess.run(  # noqa: S603
        ["daydream", "benchmark", "status", str(ws)],  # noqa: S607
        capture_output=True, text=True,
    )
    assert r2.returncode == 0 and "empty" in r2.stdout and "unresolved" in r2.stdout

    r3 = subprocess.run(  # noqa: S603
        ["daydream", "benchmark", "validate", str(ws)],  # noqa: S607
        capture_output=True, text=True,
    )
    assert r3.returncode == 2  # fresh workspace: structurally valid but incomplete
