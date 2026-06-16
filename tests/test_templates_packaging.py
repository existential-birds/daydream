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


def test_all_three_workflows_ship_in_package() -> None:
    names = {p.name for p in workflow_template_files()}
    assert names == _EXPECTED_TEMPLATES


def test_workflows_reference_the_canonical_secret_and_var_names() -> None:
    blob = "\n".join(p.read_text(encoding="utf-8") for p in workflow_template_files())
    for secret in config.SETUP_SECRET_NAMES:  # the deposit step + YAML cannot drift
        assert secret in blob
    assert config.BOT_HANDLE_VAR in blob


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
        with zipfile.ZipFile(wheels[0]) as zf:
            names_in_wheel = set(zf.namelist())
        assert expected_paths <= names_in_wheel, (
            f"Built wheel is missing YAML templates. "
            f"Found YAML entries: {sorted(n for n in names_in_wheel if n.endswith('.yml'))!r}. "
            f"Expected paths: {sorted(expected_paths)!r}. "
            f"Check [tool.hatch.build.targets.wheel] include in pyproject.toml."
        )
