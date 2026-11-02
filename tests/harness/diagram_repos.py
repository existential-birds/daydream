"""Real Git repositories and grounded specs shared by both diagram flows.

Cited paths, lines and symbols must match the head tree: fixture drift would
turn a grounded case into an omission case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.harness.git_helpers import commit, git, init_repo


def load_diagram_artifact(target: Path) -> dict[str, Any]:
    """Load ``.daydream/deep/diagram.json`` from a finished run."""
    path = target / ".daydream" / "deep" / "diagram.json"
    assert path.is_file(), f"diagram artifact missing at {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


# Repository builders


def _finish(repo: Path) -> Path:
    """Commit the working tree onto a ``feature`` branch off ``main``."""
    git(repo, "add", ".")
    commit(repo, "change")
    return repo


def _branch_off_main(repo: Path) -> None:
    """Commit the written tree as ``init``, then check out a ``feature`` branch."""
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "init")
    git(repo, "checkout", "-b", "feature")


def _build_cross_module_variant(root: Path, name: str, client_body: str) -> Path:
    """Two-package initial tree with one cross-module import edge, then feature edits."""
    repo = root / name
    (repo / "pkg_a").mkdir(parents=True)
    (repo / "pkg_b").mkdir(parents=True)
    (repo / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_b" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg_a" / "core.py").write_text("def handle(payload):\n    return payload\n", encoding="utf-8")
    (repo / "pkg_a" / "util.py").write_text("def normalize(text):\n    return text\n", encoding="utf-8")
    (repo / "pkg_b" / "client.py").write_text(
        "from pkg_a.core import handle\n\n\ndef call_handle(payload):\n"
        "    return handle(payload)\n",
        encoding="utf-8",
    )
    _branch_off_main(repo)
    (repo / "pkg_a" / "core.py").write_text(CORE_PY, encoding="utf-8")
    (repo / "pkg_a" / "util.py").write_text("def normalize(text):\n    return text.strip()\n", encoding="utf-8")
    (repo / "pkg_b" / "client.py").write_text(client_body, encoding="utf-8")
    return _finish(repo)


def build_cross_module_repo(root: Path) -> Path:
    """Sequence-eligible: three changed files, two modules and one crossing import.
    No changed function gains a branch point, so flowcharts stay ineligible."""
    return _build_cross_module_variant(root, "cross_module", CLIENT_PY)


def build_branch_heavy_repo(root: Path) -> Path:
    """Four new branch points satisfy flowchart eligibility. One code file and
    module without service/import edges keep sequence diagrams ineligible."""
    repo = root / "branch_heavy"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "pipeline.py").write_text("def run(payload):\n    return payload\n", encoding="utf-8")
    (repo / "README.md").write_text("# app\n", encoding="utf-8")
    _branch_off_main(repo)
    (repo / "app" / "pipeline.py").write_text(PIPELINE_PY, encoding="utf-8")
    return _finish(repo)


def build_both_signals_repo(root: Path) -> Path:
    """Combine cross-module imports and PIPELINE_PY for both diagram kinds.

    After the CLIENT_PY prefix and two newlines, run starts at line 11, its branch
    points are 12/14/16/17, and fast_path starts at line 22."""
    return _build_cross_module_variant(root, "both_signals", CLIENT_PY + "\n\n" + PIPELINE_PY)


def build_flat_repo(root: Path) -> Path:
    """Two files in one module with no branches: record signals without calling a backend."""
    repo = root / "flat"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "app" / "two.py").write_text("OTHER = 2\n", encoding="utf-8")
    _branch_off_main(repo)
    (repo / "app" / "one.py").write_text("VALUE = 11\n", encoding="utf-8")
    (repo / "app" / "two.py").write_text("OTHER = 22\n", encoding="utf-8")
    return _finish(repo)


# File bodies (line numbers below are load-bearing for the specs)

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
#: on lines 2, 4, 6 and 7; ``fast_path`` is defined on line 12. Reused by the
#: both-signals fixture (see :func:`build_both_signals_repo`), where the same
#: body is offset by the ``CLIENT_PY`` prefix.
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


# Canonical grounded specs


def _participant(name: str, files: list[str], *, service: str | None = None) -> dict[str, Any]:
    return {"name": name, "kind": "internal", "files": files, "service": service}


def _message(frm: str, to: str, label: str, kind: str, *, file: str, line: int, symbol: str,) -> dict[str, Any]:
    return {"from": frm, "to": to, "label": label, "kind": kind, "changed": True,
        "evidence": {"file": file, "line": line, "symbol": symbol},
    }


def sequence_spec() -> dict[str, Any]:
    """Five messages across three participants, with each reply citing its
    function return and immediately following the reversed call. Five messages
    leave a renderable partial diagram when client.py reads are withheld."""
    return {"participants": [_participant("Client", ["pkg_b/client.py"]), _participant("Core", ["pkg_a/core.py"]),
            _participant("Util", ["pkg_a/util.py"]),
        ],
        "messages": [_message(
                "Client", "Util", "Normalize payload", "call", file="pkg_b/client.py", line=6, symbol="normalize",
            ), _message("Util", "Client", "Stripped text", "reply", file="pkg_a/util.py", line=2, symbol="normalize",),
            _message(
                "Client", "Core", "Handle cleaned payload", "call", file="pkg_b/client.py", line=7, symbol="handle",
            ), _message("Core", "Client", "Cleaned payload", "reply", file="pkg_a/core.py", line=3, symbol="handle",),
            _message("Core", "Core", "Normalize inside handler", "self",
                file="pkg_a/core.py", line=2, symbol="normalize_payload",
            ),
        ], "blocks": [],
    }


def flowchart_spec(*, root_file: str = "app/pipeline.py", offset: int = 0) -> dict[str, Any]:
    """Seven nodes rooted at run, with decisions, fast_path and terminal returns.
    Apply offset to all root_file citations so the combined fixture reuses the shape."""

    def _node(node_id: str, kind: str, label: str, line: int, symbol: str | None) -> dict[str, Any]:
        return {"id": node_id, "kind": kind, "label": label,
            "evidence": {"file": root_file, "line": line + offset, "symbol": symbol},
        }

    return {"root": {"file": root_file, "name": "run", "line": 1 + offset},
        "nodes": [_node("start", "start", "run", 1, "run"), _node("d1", "decision", "payload is None?", 2, None),
            _node("e1", "end", "Return empty", 3, None), _node("d2", "decision", "fast mode?", 4, None),
            _node("s1", "subroutine", "fast_path", 5, "fast_path"), _node("p1", "process", "Scan items", 6, None),
            _node("e2", "end", "Return none", 9, None),
        ], "edges": [{"from": "start", "to": "d1", "label": None}, {"from": "d1", "to": "e1", "label": "yes"},
            {"from": "d1", "to": "d2", "label": "no"}, {"from": "d2", "to": "s1", "label": "yes"},
            {"from": "d2", "to": "p1", "label": "no"}, {"from": "s1", "to": "e2", "label": None},
            {"from": "p1", "to": "e2", "label": None},
        ],
    }


