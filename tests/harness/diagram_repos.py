"""Real git fixture repositories for the grounded-diagram tests (issue #1113).

Four shapes, each built to satisfy (or deliberately miss) one deterministic
eligibility rule, plus the canonical grounded specs that go with them. Shared by
``tests/test_deep_diagram_integration.py`` (the review paths) and
``tests/test_diagram_only_integration.py`` (the ``--diagram-only`` flow), which
need identical repositories to make their assertions comparable.

Every path, line number and symbol in the specs below is real: the grounding
pass reads the head tree, so a spec that drifts from the fixture stops being a
"grounded spec" fixture and silently becomes an omission fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.harness.git_helpers import commit, git, init_repo

# ---------------------------------------------------------------------------
# Repository builders
# ---------------------------------------------------------------------------


def _finish(repo: Path) -> Path:
    """Commit the working tree onto a ``feature`` branch off ``main``."""
    git(repo, "add", ".")
    commit(repo, "change")
    return repo


def build_cross_module_repo(root: Path) -> Path:
    """Two packages, three changed code files, one cross-module import edge.

    Satisfies the sequence diagram's cross-module rule (>= 3 code files, >= 2
    modules, >= 1 import edge crossing modules) and deliberately misses the
    flowchart rule: no changed function gains a single branch point.
    """
    repo = root / "cross_module"
    (repo / "pkg_a").mkdir(parents=True)
    (repo / "pkg_b").mkdir(parents=True)
    (repo / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_b" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_a" / "core.py").write_text(
        "def handle(payload):\n    return payload\n", encoding="utf-8"
    )
    (repo / "pkg_a" / "util.py").write_text(
        "def normalize(text):\n    return text\n", encoding="utf-8"
    )
    (repo / "pkg_b" / "client.py").write_text(
        "from pkg_a.core import handle\n\n\ndef call_handle(payload):\n"
        "    return handle(payload)\n",
        encoding="utf-8",
    )
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "init")
    git(repo, "checkout", "-b", "feature")
    (repo / "pkg_a" / "core.py").write_text(CORE_PY, encoding="utf-8")
    (repo / "pkg_a" / "util.py").write_text(
        "def normalize(text):\n    return text.strip()\n", encoding="utf-8"
    )
    (repo / "pkg_b" / "client.py").write_text(CLIENT_PY, encoding="utf-8")
    return _finish(repo)


def build_branch_heavy_repo(root: Path) -> Path:
    """One module, one changed function gaining four branch points.

    Satisfies the flowchart rule and misses the sequence rules (one code file,
    one module, no service, no import edge).
    """
    repo = root / "branch_heavy"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "pipeline.py").write_text(
        "def run(payload):\n    return payload\n", encoding="utf-8"
    )
    (repo / "README.md").write_text("# app\n", encoding="utf-8")
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "init")
    git(repo, "checkout", "-b", "feature")
    (repo / "app" / "pipeline.py").write_text(PIPELINE_PY, encoding="utf-8")
    return _finish(repo)


def build_both_signals_repo(root: Path) -> Path:
    """Cross-module AND branch-heavy: both kinds are eligible in one run.

    Same three files and the same import edge as the cross-module fixture, with
    the branch-heavy ``run`` function added to ``pkg_b/client.py`` so a single
    run exercises the two-kind fan-out.
    """
    repo = root / "both_signals"
    (repo / "pkg_a").mkdir(parents=True)
    (repo / "pkg_b").mkdir(parents=True)
    (repo / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_b" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_a" / "core.py").write_text(
        "def handle(payload):\n    return payload\n", encoding="utf-8"
    )
    (repo / "pkg_a" / "util.py").write_text(
        "def normalize(text):\n    return text\n", encoding="utf-8"
    )
    (repo / "pkg_b" / "client.py").write_text(
        "from pkg_a.core import handle\n\n\ndef call_handle(payload):\n"
        "    return handle(payload)\n",
        encoding="utf-8",
    )
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "init")
    git(repo, "checkout", "-b", "feature")
    (repo / "pkg_a" / "core.py").write_text(CORE_PY, encoding="utf-8")
    (repo / "pkg_a" / "util.py").write_text(
        "def normalize(text):\n    return text.strip()\n", encoding="utf-8"
    )
    (repo / "pkg_b" / "client.py").write_text(
        CLIENT_PY + "\n\n" + BOTH_RUN_PY, encoding="utf-8"
    )
    return _finish(repo)


def build_cross_service_repo(root: Path) -> Path:
    """Monorepo with two manifest-bearing service roots, one changed file each.

    No import edge between them, so only the cross-service rule can fire --
    which is the point: HTTP and queue boundaries are invisible to the import
    graph.
    """
    repo = root / "cross_service"
    for service in ("alpha", "beta"):
        service_root = repo / "services" / service
        service_root.mkdir(parents=True)
        (service_root / "pyproject.toml").write_text(
            f'[project]\nname = "{service}"\n', encoding="utf-8"
        )
        (service_root / "api.py").write_text(
            f'def endpoint():\n    return "{service}"\n', encoding="utf-8"
        )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "cross-service"\n', encoding="utf-8"
    )
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "init")
    git(repo, "checkout", "-b", "feature")
    for service in ("alpha", "beta"):
        (repo / "services" / service / "api.py").write_text(
            f'def endpoint():\n    return "{service}-v2"\n', encoding="utf-8"
        )
    return _finish(repo)


def build_flat_repo(root: Path) -> Path:
    """Two files, one module, zero branch points: no kind is eligible.

    The below-threshold fixture -- the diagram step must record its signals and
    make no backend call at all.
    """
    repo = root / "flat"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "app" / "two.py").write_text("OTHER = 2\n", encoding="utf-8")
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "init")
    git(repo, "checkout", "-b", "feature")
    (repo / "app" / "one.py").write_text("VALUE = 11\n", encoding="utf-8")
    (repo / "app" / "two.py").write_text("OTHER = 22\n", encoding="utf-8")
    return _finish(repo)


# ---------------------------------------------------------------------------
# File bodies (line numbers below are load-bearing for the specs)
# ---------------------------------------------------------------------------

#: ``pkg_a/core.py`` at head. Line 2 calls ``normalize_payload``; line 3
#: returns; line 6 defines ``normalize_payload``.
CORE_PY = (
    "def handle(payload):\n"
    "    cleaned = normalize_payload(payload)\n"
    "    return cleaned\n"
    "\n"
    "\n"
    "def normalize_payload(payload):\n"
    "    return payload\n"
)

#: ``pkg_b/client.py`` at head. Line 6 calls ``normalize``; line 7 calls
#: ``handle``.
CLIENT_PY = (
    "from pkg_a.core import handle\n"
    "from pkg_a.util import normalize\n"
    "\n"
    "\n"
    "def call_handle(payload):\n"
    "    cleaned = normalize(payload)\n"
    "    result = handle(cleaned)\n"
    "    return result\n"
)

#: ``app/pipeline.py`` at head. ``run`` spans lines 1-9 with branch statements
#: on lines 2, 4, 6 and 7; ``fast_path`` is defined on line 12.
PIPELINE_PY = (
    "def run(payload):\n"
    "    if payload is None:\n"
    '        return "empty"\n'
    '    if payload.get("mode") == "fast":\n'
    "        return fast_path(payload)\n"
    '    for item in payload["items"]:\n'
    "        if item:\n"
    "            return item\n"
    '    return "none"\n'
    "\n"
    "\n"
    "def fast_path(payload):\n"
    '    return payload["items"]\n'
)

#: Appended to ``CLIENT_PY`` in the both-signals fixture. ``run`` starts on
#: line 11 (``CLIENT_PY`` is 8 lines plus the two-newline join) with branch
#: statements on lines 12, 14, 16 and 17; ``fast_path`` is defined on line 22.
BOTH_RUN_PY = (
    "def run(payload):\n"
    "    if payload is None:\n"
    '        return "empty"\n'
    '    if payload.get("mode") == "fast":\n'
    "        return fast_path(payload)\n"
    '    for item in payload["items"]:\n'
    "        if item:\n"
    "            return item\n"
    '    return "none"\n'
    "\n"
    "\n"
    "def fast_path(payload):\n"
    '    return payload["items"]\n'
)


# ---------------------------------------------------------------------------
# Canonical grounded specs
# ---------------------------------------------------------------------------


def sequence_spec() -> dict[str, Any]:
    """A fully grounded sequence spec for the cross-module fixture.

    Five messages across three participants: two calls out of
    ``pkg_b/client.py``, their adjacent replies, and one self-call inside
    ``pkg_a/core.py``. Each reply cites its enclosing function's return line
    and immediately follows the reversed call, as required by the sequence
    grounding contract. Five (not the floor's three) so a test that withholds
    the reads for ``pkg_b/client.py`` still leaves a renderable diagram behind,
    making a PARTIAL prune observable.
    """
    return {
        "participants": [
            {
                "name": "Client",
                "kind": "internal",
                "files": ["pkg_b/client.py"],
                "service": None,
            },
            {
                "name": "Core",
                "kind": "internal",
                "files": ["pkg_a/core.py"],
                "service": None,
            },
            {
                "name": "Util",
                "kind": "internal",
                "files": ["pkg_a/util.py"],
                "service": None,
            },
        ],
        "messages": [
            {
                "from": "Client",
                "to": "Util",
                "label": "Normalize payload",
                "kind": "call",
                "changed": True,
                "evidence": {
                    "file": "pkg_b/client.py",
                    "line": 6,
                    "symbol": "normalize",
                },
            },
            {
                "from": "Util",
                "to": "Client",
                "label": "Stripped text",
                "kind": "reply",
                "changed": True,
                "evidence": {
                    "file": "pkg_a/util.py",
                    "line": 2,
                    "symbol": "normalize",
                },
            },
            {
                "from": "Client",
                "to": "Core",
                "label": "Handle cleaned payload",
                "kind": "call",
                "changed": True,
                "evidence": {
                    "file": "pkg_b/client.py",
                    "line": 7,
                    "symbol": "handle",
                },
            },
            {
                "from": "Core",
                "to": "Client",
                "label": "Cleaned payload",
                "kind": "reply",
                "changed": True,
                "evidence": {"file": "pkg_a/core.py", "line": 3, "symbol": "handle"},
            },
            {
                "from": "Core",
                "to": "Core",
                "label": "Normalize inside handler",
                "kind": "self",
                "changed": True,
                "evidence": {
                    "file": "pkg_a/core.py",
                    "line": 2,
                    "symbol": "normalize_payload",
                },
            },
        ],
        "blocks": [],
    }


def flowchart_spec(*, root_file: str = "app/pipeline.py", offset: int = 0) -> dict[str, Any]:
    """A fully grounded flowchart spec for the branch-heavy fixture.

    Seven nodes rooted at ``run``: start, two decisions, one subroutine call to
    ``fast_path``, one process, and two terminal returns.

    Args:
        root_file: The file holding ``run``.
        offset: Line offset to add to every citation, so the both-signals
            fixture (where ``run`` starts on line 11 rather than line 1) reuses
            the same shape.
    """

    def _node(node_id: str, kind: str, label: str, line: int, symbol: str | None) -> dict[str, Any]:
        return {
            "id": node_id,
            "kind": kind,
            "label": label,
            "evidence": {"file": root_file, "line": line + offset, "symbol": symbol},
        }

    return {
        "root": {"file": root_file, "name": "run", "line": 1 + offset},
        "nodes": [
            _node("start", "start", "run", 1, "run"),
            _node("d1", "decision", "payload is None?", 2, None),
            _node("e1", "end", "Return empty", 3, None),
            _node("d2", "decision", "fast mode?", 4, None),
            _node("s1", "subroutine", "fast_path", 5, "fast_path"),
            _node("p1", "process", "Scan items", 6, None),
            _node("e2", "end", "Return none", 9, None),
        ],
        "edges": [
            {"from": "start", "to": "d1", "label": None},
            {"from": "d1", "to": "e1", "label": "yes"},
            {"from": "d1", "to": "d2", "label": "no"},
            {"from": "d2", "to": "s1", "label": "yes"},
            {"from": "d2", "to": "p1", "label": "no"},
            {"from": "s1", "to": "e2", "label": None},
            {"from": "p1", "to": "e2", "label": None},
        ],
    }


def cross_service_sequence_spec() -> dict[str, Any]:
    """A three-message sequence spec for the cross-service fixture."""
    return {
        "participants": [
            {
                "name": "Alpha",
                "kind": "internal",
                "files": ["services/alpha/api.py"],
                "service": "alpha",
            },
            {
                "name": "Beta",
                "kind": "internal",
                "files": ["services/beta/api.py"],
                "service": "beta",
            },
        ],
        "messages": [
            {
                "from": "Alpha",
                "to": "Alpha",
                "label": "Call alpha endpoint",
                "kind": "call",
                "changed": True,
                "evidence": {
                    "file": "services/alpha/api.py",
                    "line": 1,
                    "symbol": "endpoint",
                },
            },
            {
                "from": "Alpha",
                "to": "Alpha",
                "label": "Return alpha body",
                "kind": "reply",
                "changed": True,
                "evidence": {
                    "file": "services/alpha/api.py",
                    "line": 2,
                    "symbol": "endpoint",
                },
            },
            {
                "from": "Beta",
                "to": "Beta",
                "label": "Serve beta endpoint",
                "kind": "self",
                "changed": True,
                "evidence": {
                    "file": "services/beta/api.py",
                    "line": 1,
                    "symbol": "endpoint",
                },
            },
        ],
        "blocks": [],
    }
