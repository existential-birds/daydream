"""Stack detection routing tests (D-11..D-16).

Covers ``daydream.deep.detection.detect_stacks`` implemented in plan 05-01.
"""


def test_extension_routing_python() -> None:
    """D-11: .py files route to python stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py"], skill_availability={"python"})
    names = {a.stack_name for a in result}
    assert "python" in names


def test_extension_routing_react() -> None:
    """D-11: .tsx files route to react stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/App.tsx"], skill_availability={"react"})
    assert "react" in {a.stack_name for a in result}


def test_ambiguous_single_stack_shortcut() -> None:
    """D-12: single stack in diff -> ambiguous files unconditionally join it."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/app.py", "migrations/001.sql"], skill_availability={"python"})
    python = next(a for a in result if a.stack_name == "python")
    assert "migrations/001.sql" in python.files


def test_ambiguous_nearest_ancestor() -> None:
    """D-12: ambiguous file routes to nearest-ancestor unambiguous stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(
        ["backend/api/main.py", "backend/api/queries.sql", "frontend/App.tsx"],
        skill_availability={"python", "react"},
    )
    python = next(a for a in result if a.stack_name == "python")
    assert "backend/api/queries.sql" in python.files


def test_equal_depth_fallthrough() -> None:
    """D-12c: equal-depth ambiguity falls through to generic."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(
        ["main.py", "App.tsx", "shared.sql"],  # .sql has no unambiguous ancestor
        skill_availability={"python", "react"},
    )
    generic = next(a for a in result if a.stack_name == "generic")
    assert "shared.sql" in generic.files


def test_config_default_generic() -> None:
    """D-13a: .yaml / .toml route to generic by default."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["config.yaml"], skill_availability=set())
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == {"generic"}


def test_config_promotion_pyproject() -> None:
    """D-13b: pyproject.toml + .py co-change -> python."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["pyproject.toml", "src/main.py"], skill_availability={"python"})
    python = next(a for a in result if a.stack_name == "python")
    assert "pyproject.toml" in python.files


def test_no_static_promotion_without_cochange() -> None:
    """D-13c: static paths alone do not promote config to a stack."""
    from daydream.deep.detection import detect_stacks

    # pyproject.toml alone (no .py in diff) stays generic
    result = detect_stacks(["pyproject.toml"], skill_availability={"python"})
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == {"generic"}


def test_md_pinned_to_generic() -> None:
    """D-14: .md files pinned to generic even when co-changed with code."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py", "README.md"], skill_availability={"python"})
    generic = next(a for a in result if a.stack_name == "generic")
    assert "README.md" in generic.files
    assert generic.is_docs_only is False  # mixed with py stack, but docs go here


def test_no_files_dropped() -> None:
    """D-15: every file is routed somewhere."""
    from daydream.deep.detection import detect_stacks

    files = ["src/main.py", "README.md", "config.yaml", "Dockerfile", "src/App.tsx"]
    result = detect_stacks(files, skill_availability={"python", "react"})
    routed = {f for a in result for f in a.files}
    assert routed == set(files)


def test_missing_skill_routes_to_generic() -> None:
    """D-16: detected stack with no installed skill -> generic."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/lib.rs"], skill_availability=set())  # rust not installed
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == {"generic"}


def test_structure_stack_emitted_for_code_diff() -> None:
    """Structure stack is unconditionally present on any non-docs-only code diff."""
    from daydream.config import STRUCTURE_SKILL, STRUCTURE_STACK_NAME
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py", "src/util.py"], skill_availability={"python"})
    structure = next((a for a in result if a.stack_name == STRUCTURE_STACK_NAME), None)
    assert structure is not None
    assert structure.skill_invocation == STRUCTURE_SKILL
    assert structure.files == ["src/main.py", "src/util.py"]
    assert structure.is_docs_only is False


def test_structure_stack_files_are_union_across_languages() -> None:
    """Structure stack sees every changed file regardless of language."""
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import detect_stacks

    files = ["api/main.py", "ui/App.tsx", "infra/Dockerfile"]
    result = detect_stacks(files, skill_availability={"python", "react"})
    structure = next(a for a in result if a.stack_name == STRUCTURE_STACK_NAME)
    assert sorted(structure.files) == sorted(files)


def test_structure_stack_skipped_for_docs_only_diff() -> None:
    """Structural rubric does not apply when the entire diff is docs."""
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["README.md", "CHANGELOG.md"], skill_availability=set())
    assert all(a.stack_name != STRUCTURE_STACK_NAME for a in result)


def test_structure_stack_skipped_for_empty_diff() -> None:
    """Empty changed_files yields no stacks at all, including structure."""
    from daydream.deep.detection import detect_stacks

    assert detect_stacks([], skill_availability=set()) == []


# --- Issue #731: deep-review sharding splitter ---


def test_shard_stacks_splits_oversized_stack_by_file_count() -> None:
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(
        stack_name="python",
        skill_invocation="beagle-python:review-python",
        files=[f"src/m{i}.py" for i in range(6)],
    )
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    shards = [s for s in out if s.stack_name.startswith("python#")]
    assert len(shards) == 3                      # 6 files / 2 per shard
    assert [s.stack_name for s in shards] == ["python#0", "python#1", "python#2"]
    union = [f for s in shards for f in s.files]
    assert sorted(union) == sorted(stack.files)  # no file dropped, no duplicate
    assert all(len(s.files) <= 2 for s in shards)


def test_shard_stacks_never_splits_structure_meta_stack() -> None:
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    structure = StackAssignment(
        stack_name=STRUCTURE_STACK_NAME,
        skill_invocation="beagle-core:review-structure",
        files=[f"src/m{i}.py" for i in range(50)],
    )
    out = shard_stacks([structure], "", max_files=5, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    assert [s for s in out if s.stack_name == STRUCTURE_STACK_NAME] == [structure]  # unchanged, single


def test_shard_stacks_deterministic_names_and_assignments() -> None:
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(stack_name="python", skill_invocation="s", files=[f"src/m{i}.py" for i in range(5)])
    kw = dict(max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    a = shard_stacks([stack], "", **kw)
    b = shard_stacks([stack], "", **kw)
    assert [(s.stack_name, s.files) for s in a] == [(s.stack_name, s.files) for s in b]


def test_shard_stacks_under_bound_returns_original_unsplit() -> None:
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(stack_name="python", skill_invocation="s", files=["a.py", "b.py"])
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    assert out == [stack]


def test_shard_stacks_splits_by_changed_bytes_not_file_count() -> None:
    """Issue #731: an oversized *byte* budget forces a split even when the file
    count is within ``max_files``; the union is still exact (no drop/dup)."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    # 3 files but one huge hunk pushes total changed bytes over max_bytes.
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+'x'*2000\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n+'y'\n"
        "diff --git a/c.py b/c.py\n--- a/c.py\n+++ b/c.py\n@@ -1 +1 @@\n+'z'\n"
    )
    stack = StackAssignment(stack_name="python", skill_invocation="s",
                            files=["a.py", "b.py", "c.py"])
    out = shard_stacks([stack], diff, max_files=100, max_bytes=100, fanout_cap=16, frontier_max=8)
    shards = [s for s in out if s.stack_name.startswith("python#")]
    assert len(shards) >= 2                     # byte budget forces a split
    assert all(len(s.files) >= 1 for s in shards)
    union = [f for s in shards for f in s.files]
    assert sorted(union) == ["a.py", "b.py", "c.py"]  # still no drop/dup
