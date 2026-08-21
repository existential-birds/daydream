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


def test_legacy_bench_still_works_alongside_benchmark():
    # The old `bench` verb must remain registered and dispatch to its own help,
    # proving coexistence (issue 15 owns removal).
    r = subprocess.run(  # noqa: S603
        ["daydream", "bench", "--help"], capture_output=True, text=True  # noqa: S607
    )
    assert r.returncode == 0 and "--benchmark-repo" in r.stdout


def test_private_benchmark_docs_document_the_new_verb():
    # The new verb must be documented so users can discover init/status/validate.
    from pathlib import Path

    text = Path("docs/benchmark.md").read_text(encoding="utf-8")
    assert "daydream benchmark init" in text
    assert "--reviewer-host" in text and "--judge-host" in text
    assert "daydream benchmark validate" in text
    assert "exit" in text.lower()  # 0/2/1 codes surfaced
