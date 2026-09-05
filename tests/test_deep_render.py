from typing import Any

from daydream.deep.render import insert_diagrams_section, render_report


def _item(id: Any, lens: Any, file: Any, line: Any, sev: Any, desc: Any) -> dict[str, Any]:
    return {"id": id, "lens": lens, "file": file, "line": line, "severity": sev,
            "description": desc, "confidence": "HIGH", "rationale": "r"}


def test_render_places_structural_above_issues_and_keeps_all_lenses() -> None:
    md = render_report([
        _item(1, "structural", "big.py", 1, "high", "1k-line file"),
        _item(2, "per-stack", "a.py", 9, "medium", "bug"),
        _item(3, "cross-stack", "b.py", 2, "high", "contract drift"),
        _item(4, "wonder", "w.py", 5, "medium", "wonder finding"),
    ])
    assert md.index("## Structural Review") < md.index("## Issues") < md.index("## Cross-Stack Issues")
    assert "[cross-stack]" in md                       # cross-stack prefix preserved
    assert "big.py" in md and "a.py" in md and "b.py" in md   # nothing dropped
    assert "## Wonder Findings" in md and "w.py" in md  # wonder items are shipped (issue #741)


# ---------------------------------------------------------------------------
# ## Diagrams section (issue #1113). The blocks themselves are rendered by
# deep/diagram_render.py and golden-tested in tests/test_diagram_render.py;
# these tests pin the pure text surgery render.py performs on the report.
# ---------------------------------------------------------------------------

_BLOCKS = "<details><summary><h3>Flowchart</h3></summary>\n\n```mermaid\nflowchart TD\n```\n\n</details>"


def _items() -> list[dict[str, Any]]:
    return [_item(1, "per-stack", "a.py", 9, "medium", "bug"),
            _item(2, "cross-stack", "b.py", 2, "high", "drift")]


def test_render_report_is_byte_identical_without_diagram_blocks() -> None:
    plain = render_report(_items())
    assert render_report(_items(), diagram_blocks=None) == plain
    assert render_report(_items(), diagram_blocks="") == plain
    assert render_report(_items(), diagram_blocks="  \n\n ") == plain
    assert "## Diagrams" not in plain


def test_render_report_puts_the_diagrams_section_between_review_and_issues() -> None:
    report = render_report(_items(), diagram_blocks=_BLOCKS)
    assert report.split("\n")[:3] == ["# Review", "", "## Diagrams"]
    assert report.index("## Diagrams") < report.index("## Issues")
    assert f"## Diagrams\n{_BLOCKS}\n\n## Issues" in report
    assert report.endswith("\n")
    # The kwarg and the textual re-apply are one code path.
    assert report == insert_diagrams_section(render_report(_items()), _BLOCKS)


def test_insert_diagrams_section_is_idempotent_and_replaces_the_existing_section() -> None:
    once = insert_diagrams_section(render_report(_items()), _BLOCKS)
    assert insert_diagrams_section(once, _BLOCKS) == once
    assert insert_diagrams_section(insert_diagrams_section(once, _BLOCKS), _BLOCKS) == once
    assert once.count("## Diagrams") == 1
    replaced = insert_diagrams_section(once, "<details>NEW</details>")
    assert "Flowchart" not in replaced
    assert replaced == insert_diagrams_section(render_report(_items()), "<details>NEW</details>")
    # Empty blocks remove a stale section, restoring the plain report byte-for-byte.
    assert insert_diagrams_section(once, "") == render_report(_items())


def test_insert_diagrams_section_leaves_coverage_and_every_other_section_intact() -> None:
    # `## Coverage` is appended textually by load-items AFTER the merge write,
    # so the diagram step's insertion must not disturb it.
    report = render_report(_items()).rstrip("\n") + "\n\n## Coverage\n- Files in diff: 3\n"
    out = insert_diagrams_section(report, _BLOCKS)
    assert out.index("## Diagrams") < out.index("## Issues") < out.index("## Cross-Stack Issues")
    assert out.index("## Cross-Stack Issues") < out.index("## Coverage")
    assert "- Files in diff: 3" in out
    assert out.count("## Coverage") == 1
    assert insert_diagrams_section(out, _BLOCKS) == out
    assert insert_diagrams_section(out, "") == report


def test_insert_diagrams_section_without_a_review_heading_and_without_a_trailing_newline() -> None:
    out = insert_diagrams_section("## Coverage\n- Files in diff: 1\n", "<details>D</details>")
    assert out == "## Diagrams\n<details>D</details>\n\n## Coverage\n- Files in diff: 1\n"
    assert insert_diagrams_section(out, "<details>D</details>") == out
    # A report with no trailing newline keeps that convention.
    assert insert_diagrams_section("# Review", "<details>D</details>") == (
        "# Review\n\n## Diagrams\n<details>D</details>"
    )
    assert insert_diagrams_section("", "<details>D</details>") == "## Diagrams\n<details>D</details>"
    assert insert_diagrams_section("", "") == ""
