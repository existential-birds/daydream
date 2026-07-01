"""Packaged-template discovery + name-drift cross-check.

The three #147 workflow templates ship as package data under
``daydream/templates/workflows/`` so they are present in both editable and
built installs. These tests pin the discovery accessor and guard against the
deposited secret/variable names drifting away from the YAML the workflows
actually reference.
"""

from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

from daydream import config
from daydream.templates import workflow_template_files

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPECTED_TEMPLATES = {"daydream-review.yml", "daydream-command.yml", "daydream-post.yml"}
_TEMPLATES_DIR = _REPO_ROOT / "daydream" / "templates" / "workflows"
# Optional manual-copy variant; NOT part of the default `daydream setup` deposit.
_SINGLE_TEMPLATE_PATH = "daydream/templates/workflows/single/daydream.yml"


def test_all_three_workflows_ship_in_package() -> None:
    # The discovery accessor returns exactly the split trio — the optional
    # single-file variant lives in a subdirectory so it is never auto-deposited
    # (installing both would double-fire the review/post).
    names = {p.name for p in workflow_template_files()}
    assert names == _EXPECTED_TEMPLATES


def test_workflows_reference_the_canonical_secret_and_var_names() -> None:
    blob = "\n".join(p.read_text(encoding="utf-8") for p in workflow_template_files())
    for secret in config.SETUP_SECRET_NAMES:  # the deposit step + YAML cannot drift
        assert secret in blob
    assert config.BOT_HANDLE_VAR in blob


def test_single_variant_references_the_canonical_secret_and_var_names() -> None:
    single = (_REPO_ROOT / _SINGLE_TEMPLATE_PATH).read_text(encoding="utf-8")
    for secret in config.SETUP_SECRET_NAMES:
        assert secret in single
    assert config.BOT_HANDLE_VAR in single


def test_browser_guide_documents_canonical_names_and_pem_limit() -> None:
    guide = Path("docs/self-hosted-bot-setup.md").read_text(encoding="utf-8")
    for name in (*config.SETUP_SECRET_NAMES, config.BOT_HANDLE_VAR):
        assert name in guide
    assert "download" in guide.lower() and "pem" in guide.lower()  # honest PEM-floor stated
    assert "Use this template" not in guide  # no maintainer-hosted repo


def test_yaml_templates_present_in_built_wheel() -> None:
    """Build a real wheel and confirm the YAML templates are included as package data.

    The editable-install tests above exercise importlib.resources against the
    source tree; they would pass even if the [tool.hatch.build.targets.wheel]
    include glob were missing or mis-typed. This test catches that gap: it
    builds an actual wheel in a temp directory and inspects the zip archive to
    verify each template lands at the expected path inside the distribution.
    """
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", tmp],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        )
        wheels = list(Path(tmp).glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
        expected_paths = {f"daydream/templates/workflows/{name}" for name in _EXPECTED_TEMPLATES}
        expected_paths.add(_SINGLE_TEMPLATE_PATH)  # optional variant ships too
        with zipfile.ZipFile(wheels[0]) as zf:
            names_in_wheel = set(zf.namelist())
        assert expected_paths <= names_in_wheel, (
            f"Built wheel is missing YAML templates. "
            f"Found YAML entries: {sorted(n for n in names_in_wheel if n.endswith('.yml'))!r}. "
            f"Expected paths: {sorted(expected_paths)!r}. "
            f"Check [tool.hatch.build.targets.wheel] include in pyproject.toml."
        )
