from daydream.deep.render import render_report


def _item(id, lens, file, line, sev, desc):
    return {"id": id, "lens": lens, "file": file, "line": line, "severity": sev,
            "description": desc, "confidence": "HIGH", "rationale": "r"}


def test_render_places_structural_above_issues_and_keeps_all_lenses():
    md = render_report([
        _item(1, "structural", "big.py", 1, "high", "1k-line file"),
        _item(2, "per-stack", "a.py", 9, "medium", "bug"),
        _item(3, "cross-stack", "b.py", 2, "high", "contract drift"),
    ])
    assert md.index("## Structural Review") < md.index("## Issues") < md.index("## Cross-Stack Issues")
    assert "[cross-stack]" in md                       # cross-stack prefix preserved
    assert "big.py" in md and "a.py" in md and "b.py" in md   # nothing dropped
