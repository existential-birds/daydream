"""Packaged-template discovery + name-drift cross-check.

The three #147 workflow templates ship as package data under
``daydream/templates/workflows/`` so they are present in both editable and
built installs. These tests pin the discovery accessor and guard against the
deposited secret/variable names drifting away from the YAML the workflows
actually reference.
"""

from __future__ import annotations

from daydream import config
from daydream.templates import workflow_template_files


def test_all_three_workflows_ship_in_package() -> None:
    names = {p.name for p in workflow_template_files()}
    assert names == {"daydream-review.yml", "daydream-command.yml", "daydream-post.yml"}


def test_workflows_reference_the_canonical_secret_and_var_names() -> None:
    blob = "\n".join(p.read_text(encoding="utf-8") for p in workflow_template_files())
    for secret in config.SETUP_SECRET_NAMES:  # the deposit step + YAML cannot drift
        assert secret in blob
    assert config.BOT_HANDLE_VAR in blob
