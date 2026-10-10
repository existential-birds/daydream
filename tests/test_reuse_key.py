"""Pure tests for the deep review reuse-key payload.

The payload separates the review **subject+contract** (``components``, hashed
into the key) from the loop-re-derived **grounding** (recorded, never hashed).
These tests assert the classification in both directions: every component moves
the key, and no grounding input can.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from daydream.deep import reuse_key

_UNSET = object()


def _flip(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "-mutated"
    if isinstance(value, list):
        return [*value, "mutated"]
    if isinstance(value, dict):
        return {**value, "digest": "0" * 64}
    return "mutated"


def _mutate(payload: dict[str, Any], dotted: str, value: Any = _UNSET) -> dict[str, Any]:
    clone = copy.deepcopy(payload)
    cursor = clone
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = _flip(cursor[parts[-1]]) if value is _UNSET else value
    return clone


def _identity() -> reuse_key.PhaseIdentity:
    return reuse_key.PhaseIdentity(backend="claude", model="claude-sonnet-4-5", effort="high", profile_digest="d" * 64,
    )


def _shard_diff(run_id: str) -> str:
    return (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1,2 @@\n"
        " A = 1\n"
        f"+# artifact /runs/{run_id}/notes.txt\n"
    )


def _hunk_index() -> dict[str, Any]:
    return {"a.py": {"hunks": [{"old_start": 1, "old_end": 1, "new_start": 1, "new_end": 2, "added": 1, "removed": 0}],
            "added_total": 1, "removed_total": 0,
        }
    }


def _shard_payload(tmp_path: Path, *, files: list[str], frontier: list[str], blob: bytes,
    run_id: str = "11111111-1111-1111-1111-111111111111",
) -> dict[str, Any]:
    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    for name in [*files, *frontier]:
        target = worktree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob if name in files else b"frontier = 1\n")
    exploration = tmp_path / "exploration"
    exploration.mkdir(parents=True, exist_ok=True)
    (exploration / "files.json").write_text("pre-scan")
    payload = reuse_key.shard_key_payload(stack_name="python", files=files, frontier_files=frontier, docs_only=False,
        diff_path_or_hunks=_shard_diff(run_id), hunk_index=_hunk_index(), exploration_dir=exploration,
        worktree_root=worktree,
        identity=reuse_key.PhaseIdentity(
            backend="claude", model="claude-sonnet-4-5", effort="high", profile_digest="d" * 64,
        ), intent_authoritative=True, include_alternatives=False, prior_commits=None, intent_text=None,
        alternatives_text=None,
    )
    return payload

def test_components_move_the_key_and_grounding_never_does(tmp_path: Path) -> None:
    base = _shard_payload(tmp_path, files=["a.py"], frontier=[], blob=b"A = 1\n")
    key = reuse_key.unit_key(base)
    assert key is not None and len(key) == 64
    assert base["format"] == 2
    obsolete_format = _mutate(_mutate(base, "format", value=1), "components.format", value=1)
    assert reuse_key.unit_key(obsolete_format) != key
    for name in ("format", "hunk_slice", "assigned_files", "assigned_blobs", "frontier_files",
                 "frontier_blobs", "profile", "model", "effort", "intent_authoritative",
                 "include_alternatives", "exploration_present", "docs_only", "schema"):
        assert reuse_key.unit_key(_mutate(base, f"components.{name}")) != key, name
    # MH2/MH16, both directions: every re-derived input is grounding, never a key input.
    for name in ("exploration", "intent", "alternatives", "settled_decisions"):
        moved = _mutate(base, f"grounding.{name}", value={"digest": "f" * 64})
        assert reuse_key.unit_key(moved) == key, f"{name} must not move the key"
        assert reuse_key.grounding_digests(moved)[name] == "f" * 64   # it is still recorded
    # An absent REQUIRED component is a miss, named; absent GROUNDING is a value, not a miss.
    blinded = _mutate(base, "components.assigned_blobs", value=None)
    assert reuse_key.unit_key(blinded) is None
    assert reuse_key.absent_components(blinded) == ["assigned_blobs"]
    assert reuse_key.unit_key(_mutate(base, "grounding.intent", value={"digest": "absent"})) == key
    # Absence of the pre-scan is a CONTRACT flag, not a missing component.
    assert reuse_key.unit_key(_mutate(base, "components.exploration_present", value=False)) != key
    # Run-scoped identifiers are normalized out of every digest input (A9).
    assert reuse_key.unit_key(
        _shard_payload(tmp_path, files=["a.py"], frontier=[], blob=b"A = 1\n", run_id="other-run")
    ) == key
    # Identical inputs from two independent builds agree.
    assert reuse_key.unit_key(_shard_payload(tmp_path, files=["a.py"], frontier=[], blob=b"A = 1\n")) == key


def test_run_scoped_identifiers_are_normalized_out_of_text() -> None:
    assert reuse_key.normalize_run_scoped("see /runs/11111111-1111-1111-1111-111111111111/output.json now"
    ) == "see /output.json now"

def test_blob_map_missing_file_is_none(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    assert reuse_key.blob_map_digest(tmp_path, ["missing.py"]) is None
    assert reuse_key.blob_map_digest(tmp_path, ["a.py"]) is not None
    assert reuse_key.blob_map_digest(tmp_path, []) is not None

def test_grounding_is_recorded_even_when_absent(tmp_path: Path) -> None:
    payload = _shard_payload(tmp_path, files=["a.py"], frontier=[], blob=b"A = 1\n")
    assert set(reuse_key.grounding_digests(payload)) == {"exploration", "intent", "alternatives", "settled_decisions"}
    assert reuse_key.grounding_digests(payload)["exploration"] != "absent"
    assert reuse_key.grounding_digests(payload)["intent"] == "absent"


def _arbiter_payload(*, records: dict[str, bytes | None] | None = None, structural: bytes | None = b'{"issues": []}',
    plan: dict[str, Any] | None = None, precision_mode: bool = False, intent: str = "i" * 64,
) -> dict[str, Any]:
    return reuse_key.arbiter_key_payload(
        contributing_records=(records if records is not None else {"stack-python-records.json": b"{}"}),
        structural_records=structural,
        plan=plan if plan is not None else {"sharded": True, "groups": [["python:1"], ["react:1"]]},
        precision_mode=precision_mode, identity=_identity(),
        grounding={
            "intent": {"digest": intent}, "alternatives": {"digest": "absent"}, "exploration": {"digest": "absent"},
        },
    )

def test_arbiter_key_tracks_records_plan_and_precision_but_not_grounding() -> None:
    """MH8: the arbiter is one content key over its own inputs (all contributing
    records + plan + precision), and the loop-re-derived intent/alternatives/
    pre-scan are recorded grounding that can never move it."""
    base = _arbiter_payload()
    key = reuse_key.unit_key(base)
    assert key is not None and len(key) == 64
    assert reuse_key.unit_key(_arbiter_payload(records={"stack-python-records.json": b'{"issues": [1]}'})) != key
    # A contributing records file that could not be read is a named miss, never
    # a partial key over the files that did read.
    assert reuse_key.unit_key(_arbiter_payload(records={"stack-python-records.json": None})) is None
    # A structural stack appearing or moving is a subject change.
    assert reuse_key.unit_key(_arbiter_payload(structural=None)) != key
    assert reuse_key.unit_key(_arbiter_payload(structural=b'{"issues": [1]}')) != key
    assert reuse_key.unit_key(_arbiter_payload(plan={"sharded": False, "groups": [["python:1"]]})) != key
    assert reuse_key.unit_key(_arbiter_payload(precision_mode=True)) != key
    # Grounding is recorded on every payload, never read by the key (MH2/MH16).
    moved = _arbiter_payload(intent="j" * 64)
    assert reuse_key.unit_key(moved) == key
    assert reuse_key.grounding_digests(moved)["intent"] == "j" * 64
    assert set(reuse_key.grounding_digests(base)) == {"intent", "alternatives", "exploration"}
