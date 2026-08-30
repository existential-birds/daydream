from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = (REPO / "CLAUDE.md").read_text()


def test_claude_md_commands_list_deadcode() -> None:
    assert "make deadcode" in CLAUDE_MD


def test_readme_documents_git_token() -> None:
    readme = (REPO / "README.md").read_text()
    assert "`DAYDREAM_GIT_TOKEN`" in readme
    assert "never embedded in the remote URL" in readme
    assert "never on the command line" in readme


def test_claude_md_git_token_row() -> None:
    table_row = "| `DAYDREAM_GIT_TOKEN` | Harvest |"
    assert table_row in CLAUDE_MD
    idx_token = CLAUDE_MD.index("DAYDREAM_GH_TIMEOUT_SECONDS")
    idx_new = CLAUDE_MD.index("`DAYDREAM_GIT_TOKEN`")
    idx_hub = CLAUDE_MD.index("`DAYDREAM_TRAJECTORY_HUB_REPO`")
    assert idx_token < idx_new < idx_hub
    assert "http.extraHeader" in CLAUDE_MD


def test_check_definition_stays_in_sync_with_makefile() -> None:
    """The commands block must list every dep from Makefile's check: target, in order."""
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
