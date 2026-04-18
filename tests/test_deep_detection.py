"""Stack detection routing tests (D-11..D-16).

Every test is xfail(strict=True) until Wave 1 plan 05-01 implements the
``daydream.deep.detection`` module with ``detect_stacks(...)``.
"""

import pytest


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_extension_routing_python() -> None:
    """D-11: .py files route to python stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py"], skill_availability={"python"})
    names = {a.stack_name for a in result}
    assert "python" in names


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_extension_routing_react() -> None:
    """D-11: .tsx files route to react stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/App.tsx"], skill_availability={"react"})
    assert "react" in {a.stack_name for a in result}


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_ambiguous_single_stack_shortcut() -> None:
    """D-12: single stack in diff -> ambiguous files unconditionally join it."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/app.py", "migrations/001.sql"], skill_availability={"python"})
    python = next(a for a in result if a.stack_name == "python")
    assert "migrations/001.sql" in python.files


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_ambiguous_nearest_ancestor() -> None:
    """D-12: ambiguous file routes to nearest-ancestor unambiguous stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(
        ["backend/api/main.py", "backend/api/queries.sql", "frontend/App.tsx"],
        skill_availability={"python", "react"},
    )
    python = next(a for a in result if a.stack_name == "python")
    assert "backend/api/queries.sql" in python.files


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_equal_depth_fallthrough() -> None:
    """D-12c: equal-depth ambiguity falls through to generic."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(
        ["main.py", "App.tsx", "shared.sql"],  # .sql has no unambiguous ancestor
        skill_availability={"python", "react"},
    )
    generic = next(a for a in result if a.stack_name == "generic")
    assert "shared.sql" in generic.files


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_config_default_generic() -> None:
    """D-13a: .yaml / .toml route to generic by default."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["config.yaml"], skill_availability=set())
    assert {a.stack_name for a in result} == {"generic"}


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_config_promotion_pyproject() -> None:
    """D-13b: pyproject.toml + .py co-change -> python."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["pyproject.toml", "src/main.py"], skill_availability={"python"})
    python = next(a for a in result if a.stack_name == "python")
    assert "pyproject.toml" in python.files


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_no_static_promotion_without_cochange() -> None:
    """D-13c: static paths alone do not promote config to a stack."""
    from daydream.deep.detection import detect_stacks

    # pyproject.toml alone (no .py in diff) stays generic
    result = detect_stacks(["pyproject.toml"], skill_availability={"python"})
    assert {a.stack_name for a in result} == {"generic"}


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_md_pinned_to_generic() -> None:
    """D-14: .md files pinned to generic even when co-changed with code."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py", "README.md"], skill_availability={"python"})
    generic = next(a for a in result if a.stack_name == "generic")
    assert "README.md" in generic.files
    assert generic.is_docs_only is False  # mixed with py stack, but docs go here


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_no_files_dropped() -> None:
    """D-15: every file is routed somewhere."""
    from daydream.deep.detection import detect_stacks

    files = ["src/main.py", "README.md", "config.yaml", "Dockerfile", "src/App.tsx"]
    result = detect_stacks(files, skill_availability={"python", "react"})
    routed = {f for a in result for f in a.files}
    assert routed == set(files)


@pytest.mark.xfail(reason="Wave 1 plan 05-01 not yet implemented", strict=True)
def test_missing_skill_routes_to_generic() -> None:
    """D-16: detected stack with no installed skill -> generic."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/lib.rs"], skill_availability=set())  # rust not installed
    assert {a.stack_name for a in result} == {"generic"}
