from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = (REPO / "CLAUDE.md").read_text()


def test_claude_md_commands_list_deadcode():
    assert "make deadcode" in CLAUDE_MD


def test_check_definition_stays_in_sync_with_makefile():
    """The commands block must list every dep from Makefile's check: target,
    in order, minus lockcheck which is named first."""
    check_line = next(
        line for line in (REPO / "Makefile").read_text().splitlines() if line.startswith("check:")
    )
    deps = check_line.removeprefix("check:").strip().split()
    assert deps[0] == "lockcheck"
    # Expected docs comment: "# lockcheck + deadcode + lint + typecheck + test (the gate)"
    expected_body = " + ".join(deps)
    expected_comment = f"# {expected_body} (the gate)"
    assert expected_comment in CLAUDE_MD, (
        f"CLAUDE.md commands block must document `check:` deps in order. "
        f"Makefile has: {expected_body}"
    )
