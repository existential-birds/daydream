from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_deadcode_recipe_runs_both_scans():
    makefile = (REPO / "Makefile").read_text()
    dead_section = makefile.split("\ndeadcode:", 1)[1].split("\n", 3)[:3]
    recipes = [line.lstrip("\t").strip() for line in dead_section]
    assert any(r.endswith("--config pyproject.toml daydream tests") for r in recipes)
    assert any(
        "rl/daydream_review_v1" in r and "daydream_review_v1 tests" in r
        for r in recipes
    )


def test_check_prerequisite_includes_deadcode():
    makefile = (REPO / "Makefile").read_text()
    check_line = next(line for line in makefile.splitlines() if line.startswith("check:"))
    assert "deadcode" in check_line.split()
    assert "lockcheck" in check_line.split()  # ordering preserved


def test_check_macro_lists_deadcode_first_after_lockcheck():
    makefile = (REPO / "Makefile").read_text()
    check_line = next(line for line in makefile.splitlines() if line.startswith("check:"))
    deps = check_line.removeprefix("check:").split()
    assert deps.index("deadcode") > deps.index("lockcheck")
