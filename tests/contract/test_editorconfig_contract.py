"""Contract test: the root .editorconfig keeps its load-bearing settings.

Parses `.editorconfig` with stdlib configparser (no editorconfig tooling, no
network) and asserts the core keys survive: root marker, charset/EOL/final-
newline defaults, per-type indentation, the Makefile tab rule, the Markdown
trim rule, and the daydream/atif/** vendored carve-out.
"""

from __future__ import annotations

import configparser
import io
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(path: Path) -> configparser.ConfigParser:
    """Parse an EditorConfig file with stdlib configparser.

    EditorConfig allows `root = true` before any section header; on Python
    3.14+ configparser raises MissingSectionHeaderError for such keys instead
    of routing them into DEFAULTSECT. Prepending a synthetic `[DEFAULT]`
    header restores the historical behavior the assertions rely on, without
    touching the shipped file.
    """
    cp = configparser.ConfigParser()
    text = "[DEFAULT]\n" + path.read_text(encoding="utf-8")
    cp.read_file(io.StringIO(text), source=str(path))
    return cp


def _load() -> configparser.ConfigParser:
    path = _REPO_ROOT / ".editorconfig"
    assert path.exists(), ".editorconfig missing at repo root"
    return _parse(path)


def test_root_marker_is_true() -> None:
    _load()  # missing file is a contract failure
    # `root = true` sits before any section header; with the synthetic
    # [DEFAULT] header injected by _parse it lands in configparser's
    # defaults, which every section inherits.
    defaults = _parse(_REPO_ROOT / ".editorconfig").defaults()
    assert defaults.get("root", "").strip().lower() == "true", (
        "first line `root = true` missing (configparser puts it in defaults)"
    )


def test_core_defaults() -> None:
    cp = _load()
    star = cp["*"]
    assert star["charset"] == "utf-8"
    assert star["end_of_line"] == "lf"
    assert star["insert_final_newline"] == "true"


def test_python_indent() -> None:
    section = _load()["*.py"]
    assert section["indent_style"] == "space"
    assert section["indent_size"] == "4"


def test_yaml_and_toml_indent() -> None:
    cp = _load()
    for name in ("*.{yml,yaml,toml}",):
        section = cp[name]
        assert section["indent_style"] == "space"
        assert section["indent_size"] == "2"


def test_makefile_uses_tabs() -> None:
    cp = _load()
    for name in ("Makefile", "*.mk"):
        assert cp[name]["indent_style"] == "tab", f"{name} must use tabs"


def test_markdown_preserves_hard_breaks() -> None:
    assert _load()["*.md"]["trim_trailing_whitespace"] == "false"


def test_atif_carve_out_is_whitespace_neutral() -> None:
    section = _load()["daydream/atif/**"]
    for forbidden in ("indent_style", "indent_size", "trim_trailing_whitespace"):
        assert forbidden not in section, (
            f"atif carve-out must not impose {forbidden} (vendored D-03 policy)"
        )
    assert section["insert_final_newline"] == "true"
