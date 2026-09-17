"""Model-facing severity enums must derive from ``severity.CANONICAL_LEVELS``.

Discovery is by introspection over the public ``*_SCHEMA`` constants of
``daydream.phases`` (the collection principle of ``test_output_schema_strict.py``),
never a hand-maintained call-site list, so a newly added severity-bearing schema is
covered automatically.

Teeth: the schema constants are built at import. The only way to prove a site
*tracks* the declaration — rather than coincidentally matching it today — is to
rebuild the module under a declaration the production vocabulary does not have and
watch the emitted levels move. A hand-written list does not move, and fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import daydream.phases as phases
import daydream.severity as severity

REPO = Path(__file__).resolve().parents[1]

# Requirement 9: the frozen model-facing order, asserted literally — never computed
# from the code under test.
FROZEN_MODEL_FACING = ["high", "medium", "low"]

# A declaration production never has, so a derived site must move in BOTH membership
# and order: ("high","medium","low","critical") -> ("critical","low","medium","high").
PATCHED_DECLARATION = ("high", "medium", "low", "critical")
PATCHED_MODEL_FACING = ["critical", "low", "medium", "high"]

# Presence guard (NOT the discovery mechanism): every schema constant that carried a
# model-facing severity enum before this change.
_EXPECTED_ROOTS = frozenset(
    {
        "ALTERNATIVE_REVIEW_SCHEMA",
        "ARBITER_SCHEMA",
        "MERGED_ITEMS_SCHEMA",
        "PER_STACK_RECORD_SCHEMA",
        "SUPERVISE_SCHEMA",
        "SUPPRESSION_SCHEMA",
        "UNCOVERED_SWEEP_SCHEMA",
    }
)

_REBUILD_SCRIPT = '''
import json

import daydream.severity as severity

severity.CANONICAL_LEVELS = ("high", "medium", "low", "critical")

import daydream.phases as phases  # built AFTER the declaration moves


def walk(node, path, out):
    if isinstance(node, dict):
        if isinstance(node.get("severity"), dict):
            out.append([path, node["severity"]])
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                walk(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk(value, f"{path}[{index}]", out)


found = []
for name in sorted(dir(phases)):
    if name.startswith("_") or not name.endswith("_SCHEMA"):
        continue
    schema = getattr(phases, name)
    if isinstance(schema, dict):
        walk(schema, f"daydream.phases.{name}", found)
print(json.dumps(found))
'''


def _walk(node: Any, path: str, out: list[tuple[str, dict[str, Any]]]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("severity"), dict):
            out.append((path, node["severity"]))
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                _walk(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", out)


def _severity_sites() -> list[tuple[str, dict[str, Any]]]:
    """Every (site path, severity fragment) in the public ``*_SCHEMA`` constants."""
    out: list[tuple[str, dict[str, Any]]] = []
    for name in sorted(dir(phases)):
        if name.startswith("_") or not name.endswith("_SCHEMA"):
            continue
        schema = getattr(phases, name)
        if isinstance(schema, dict):
            _walk(schema, f"daydream.phases.{name}", out)
    return out


def _levels(fragment: dict[str, Any]) -> list[str]:
    """The enumerated levels themselves (the live list, not a copy)."""
    if "anyOf" in fragment:
        nullable: list[str] = next(branch["enum"] for branch in fragment["anyOf"] if "enum" in branch)
        return nullable
    plain: list[str] = fragment["enum"]
    return plain


def _root_of(site: str) -> str:
    return site.split(".")[2]


_SITES = _severity_sites()
_SITE_IDS = [site for site, _ in _SITES]


def test_severity_sites_are_discovered() -> None:
    """Guard the guard: discovery must not match zero sites, and no known site may vanish."""
    assert _SITES, "no severity-bearing *_SCHEMA discovered — collection is broken"
    assert _EXPECTED_ROOTS <= {_root_of(site) for site in _SITE_IDS}


@pytest.mark.parametrize("site,fragment", _SITES, ids=_SITE_IDS)
def test_site_emits_the_frozen_model_facing_order(site: str, fragment: dict[str, Any]) -> None:
    assert _levels(fragment) == FROZEN_MODEL_FACING, (
        f"{site} emits {_levels(fragment)}; the frozen model-facing order is {FROZEN_MODEL_FACING}"
    )


@pytest.fixture(scope="module")
def rebuilt_under_patched_declaration(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Any]]:
    script = tmp_path_factory.mktemp("rebuild") / "rebuild.py"
    script.write_text(_REBUILD_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=REPO, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    return {site: fragment for site, fragment in json.loads(proc.stdout)}


@pytest.mark.parametrize("site,fragment", _SITES, ids=_SITE_IDS)
def test_site_follows_the_declaration_when_the_declaration_moves(
    site: str,
    fragment: dict[str, Any],
    rebuilt_under_patched_declaration: dict[str, dict[str, Any]],
) -> None:
    assert site in rebuilt_under_patched_declaration, f"{site} vanished when the declaration moved"
    emitted = _levels(rebuilt_under_patched_declaration[site])
    assert emitted == PATCHED_MODEL_FACING, (
        f"{site} emitted {emitted} under CANONICAL_LEVELS={PATCHED_DECLARATION!r}; the declaration "
        f"derives {PATCHED_MODEL_FACING}. A hand-written level list matches today's declaration and "
        f"cannot follow a later change."
    )


def test_supervise_severity_still_accepts_null() -> None:
    fragment = phases.SUPERVISE_SCHEMA["properties"]["verdicts"]["items"]["properties"]["severity"]
    assert _levels(fragment) == FROZEN_MODEL_FACING
    assert {"type": "null"} in fragment["anyOf"]


def test_no_two_sites_share_one_enum_list_object() -> None:
    owners = [(_root_of(site), _levels(fragment)) for site, fragment in _SITES]
    shared = len({id(levels) for _, levels in owners}) != len(owners)
    assert not shared, f"severity enum list objects are shared between schemas: {sorted(owners)}"


def test_pr_review_severity_breakdown_follows_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream import pr_comment_renderer
    from daydream.pr_review import (
        ParsedIssue,
        PRInfo,
        ReviewRenderers,
        _ClassifiedIssues,
        build_payload,
        default_render_finding,
        default_render_summary,
    )

    # Declaration reversed: the model-facing order must follow it, so a hand-written
    # tuple at the call site renders "1 high, 1 low" and fails here.
    monkeypatch.setattr(severity, "CANONICAL_LEVELS", ("high", "medium", "low"))
    classified = _ClassifiedIssues(
        body_only=[
            ParsedIssue(path="a.py", line=10, title="t1", body="b", confidence="HIGH", severity="high"),
            ParsedIssue(path="a.py", line=12, title="t2", body="b", confidence="LOW", severity="low"),
        ]
    )
    body = build_payload(
        PRInfo(
            number=42,
            head_sha="head123",
            base_sha="base456",
            base_ref="main",
            head_ref="feature",
            owner="acme",
            repo="widgets",
            url="https://github.com/acme/widgets/pull/42",
        ),
        classified,
        renderers=ReviewRenderers(default_render_finding, default_render_summary),
        run_info=pr_comment_renderer._render_fallback(),
    )["body"]
    assert "- **Severity:** 1 low, 1 high" in body
