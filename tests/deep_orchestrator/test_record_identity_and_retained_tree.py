"""Record Identity And Retained Tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.deep_orchestrator.support import (
    _fresh_uid_run,
    _high_record,
    _item_uids,
    _merged_items,
    _panel_text,
    _prime_source_uid_merge_resume,
    _prime_uid_merge_resume,
    _provenance_item,
    _RejectingArbiterBackend,
    _run_uid_pool,
    _source_uids_by_description,
    _uid_list,
    _uid_records,
    _uncovered_sweep_target,
)
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.test_deep_orchestrator import (
    _TWIN_DESCRIPTION,
    MakeConfig,
    Mute,
    _install_stub_backend,
    _prime_merge_resume,
    _record,
    _record_issues,
    _run_deep,
    _silence,
)


async def test_fresh_multi_stack_run_stamps_record_uid_at_birth(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """#1111 real-path: every record a fresh deep run writes is born identified."""
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    borderline = {"severity": "medium", "confidence": "MEDIUM"}
    stub.parse_by_stack = {
        "python": {
            **borderline,
            "description": "python primary finding",
            "extra": {**borderline, "description": "python second finding"},
        },
        "structure": {
            **borderline,
            "description": "structural primary finding",
            "extra": {**borderline, "description": "structural second finding"},
        },
    }

    assert await _run_deep(multi_stack_target) == 0

    deep = multi_stack_target / ".daydream" / "deep"
    records_files = sorted(deep.glob("stack-*-records.json"))
    assert [p.name for p in records_files] == [
        "stack-generic-records.json",
        "stack-python-records.json",
        "stack-react-records.json",
        "stack-structure-records.json",
    ]
    for path in records_files:
        stack = path.name.removeprefix("stack-").removesuffix("-records.json")
        issues = _record_issues(json.loads(path.read_text()))
        assert issues, f"{path.name} holds no records to identify"
        # Stack half names this very file; ordinals are contiguous from 1.
        assert [issue.get("uid") for issue in issues] == [f"{stack}:{n}" for n in range(1, len(issues) + 1)], (
            f"{path.name} uids are not this stack's contiguous sequence"
        )
    # The two-finding stacks prove the ordinal is a per-stack counter, not a
    # constant: both records carry the reviewer's ``id: 1``.
    assert _uid_list(deep, "python") == ["python:1", "python:2"]
    assert _uid_list(deep, "structure") == ["structure:1", "structure:2"]
    assert {issue["id"] for issue in _uid_records(deep, "python")} == {1, 2}


async def test_uncovered_sweep_stamps_its_own_record_uids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#1111 real-path: the uncovered-file sweep identifies its own records."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    # A second file no reviewer reads, with a hunk large enough to clear the
    # sweep's min-hunk budget, so the sweep has two targets instead of one.
    (target / "extra.txt").write_text("".join(f"extra{i}\n" for i in range(1, 8)))
    _git(target, "add", "extra.txt")
    _commit(target, "test: add a second unread file for the sweep")

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, target)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"notes.txt", "extra.txt"})
    stub.merge_echo_records = True

    assert await run(make_config(target, assume="yes", output_mode="loop")) == 0

    deep = target / ".daydream" / "deep"
    sweep_records = json.loads((deep / "stack-uncovered-records.json").read_text())
    # Records are written in sorted-file order, so the pairing is deterministic.
    assert [(r["file"], r["uid"]) for r in sweep_records] == [
        ("extra.txt", "uncovered:1"),
        ("notes.txt", "uncovered:2"),
    ]
    assert {r["id"] for r in sweep_records} == {1}, (
        "both sweep records share the reviewer's id -- the uid is the only handle that tells them apart"
    )


async def test_merge_resume_backfills_uids_onto_pre_uid_records(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111 real-path: a resume over artifacts written before ``uid`` existed re-derives the identity the
    producing run would have minted."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    deep = _prime_uid_merge_resume(
        multi_stack_target,
        python=[
            _high_record(description="py first"),
            _record(description="py second", line=2, evidence="api.py:2"),
        ],
    )
    primed = json.loads((deep / "stack-python-records.json").read_text())
    assert all("uid" not in record for record in primed), "fixture must predate the uid field"

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    # Backfilled by position, in the file that owns them -- not renumbered
    # globally and not routed elsewhere.
    assert _uid_list(deep, "python") == ["python:1", "python:2"]
    assert _uid_list(deep, "react") == ["react:1"]
    assert _uid_list(deep, "generic") == ["generic:1"]
    assert _uid_list(deep, "structure") == ["structure:1"]
    # ``python:1`` is the record the arbiter was asked about, and its verdict
    # landed on it rather than on its sibling.
    python_records = _uid_records(deep, "python")
    assert python_records[0]["description"] == "ARBITRATED: py first"
    assert python_records[1]["description"] == "py second"
    assert (deep / "merged-items.json").is_file()


async def test_merge_resume_preserves_existing_non_contiguous_uid(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111 real-path: a uid already on disk is never re-minted by position."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    deep = _prime_uid_merge_resume(
        multi_stack_target,
        python=[_high_record(description="py survivor", uid="python:4")],
    )

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    assert _uid_list(deep, "python") == ["python:4"], (
        "the surviving record was renumbered by position instead of keeping the uid the producing run minted"
    )
    assert _uid_records(deep, "python")[0]["description"] == "ARBITRATED: py survivor"


async def test_duplicate_record_uid_stops_the_run_before_merge(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111 real-path: two records sharing one uid is fatal, not a warning."""
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "daydream.deep.review_steps.print_error",
        lambda console, title, message, *a, **k: errors.append((title, message)),
    )
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    deep = _prime_uid_merge_resume(
        multi_stack_target,
        python=[
            _high_record(description="py first", uid="python:1"),
            _high_record(description="py collision", uid="python:1"),
        ],
    )

    assert await _run_deep(multi_stack_target, start_at="merge") == 1

    assert not (deep / "merged-items.json").exists(), "the run merged despite a colliding record identity"
    assert not (multi_stack_target / ".review-output.md").exists()
    titles = [title for title, _message in errors]
    assert "Duplicate Record Identities" in titles, f"no actionable error panel; got {titles!r}"
    message = next(msg for title, msg in errors if title == "Duplicate Record Identities")
    assert "python:1" in message
    assert "stack-python-records.json" in message
    assert "Re-run without --start-at" in message


async def test_arbiter_drop_removes_only_the_named_record_across_stack_files(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111 real-path: an arbiter rejection deletes one record and only that one."""
    _silence(monkeypatch)
    stub = _RejectingArbiterBackend(multi_stack_target, "react:1")
    # Echo the on-disk records as merged items so the report reflects what
    # arbitration actually wrote back, not the stub's fixed payload.
    stub.merge_echo_records = True
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    deep = _prime_merge_resume(
        multi_stack_target,
        python=[_high_record(description="py issue")],
        react=[_high_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_high_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", line=5, evidence="api.py:5")],
    )

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    # Exactly the named record is gone; its file is rewritten empty rather than
    # left holding stale pre-arbitration content.
    assert _uid_records(deep, "react") == []
    # The survivors kept their identities and took their own verdicts.
    assert _uid_list(deep, "python") == ["python:1"]
    assert _uid_list(deep, "generic") == ["generic:1"]
    assert _uid_records(deep, "python")[0]["description"] == "ARBITRATED: py issue"
    assert _uid_records(deep, "generic")[0]["description"] == "ARBITRATED: docs issue"
    # The structural record was never an arbiter target (alone at api.py:5, so
    # uncontested), and the rewrite it rides through left it untouched.
    assert _uid_list(deep, "structure") == ["structure:1"]
    assert _uid_records(deep, "structure")[0]["description"] == "structural issue"
    # The dropped record is absent from the merged output too, and its siblings
    # are not.
    merged = json.loads((deep / "merged-items.json").read_text())["items"]
    descriptions = [item.get("description") for item in merged]
    assert "ARBITRATED: tsx issue" not in descriptions
    assert "tsx issue" not in descriptions
    assert "ARBITRATED: py issue" in descriptions


async def test_unroutable_record_uid_warns_instead_of_erasing_silently(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111 real-path: a record that routes outside the rewritten files is named."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "daydream.deep.merge_steps.print_warning",
        lambda console, msg, *a, **k: warnings.append(msg),
    )
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    deep = _prime_uid_merge_resume(
        multi_stack_target,
        python=[_high_record(description="py issue", uid="ghost:1")],
    )

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    unroutable = [w for w in warnings if "ghost:1" in w]
    assert unroutable, f"the unroutable record was not named in any warning: {warnings!r}"
    assert "stack-python-records.json" in unroutable[0], "the warning must name the record's source"
    assert "stack-ghost-records.json" in unroutable[0], "the warning must name the dest it resolved to"
    assert "will not reach disk" in unroutable[0]
    # The warning is accurate, not decorative: the adjudication genuinely did not
    # reach disk, and no phantom records file was created for the ghost stack.
    assert not (deep / "stack-ghost-records.json").exists()
    assert _uid_records(deep, "python") == []
    # Fail-open: every routable record's adjudication still landed.
    assert _uid_list(deep, "react") == ["react:1"]
    assert _uid_list(deep, "generic") == ["generic:1"]
    assert _uid_list(deep, "structure") == ["structure:1"]


async def test_every_merged_item_carries_source_uids_on_a_multi_stack_run(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """#1111 real-path: every shipped item names the records it derives from."""
    deep = await _fresh_uid_run(multi_stack_target, monkeypatch, mute_side_effects)
    items = _merged_items(deep)
    pool = _run_uid_pool(deep)
    assert pool == {"generic:1", "python:1", "react:1", "structure:1"}, pool

    # (1) No shipped item is unattributed, and none of them cites a record that
    # does not exist. Checked over every item rather than the ones the fixture
    # happens to name, because "3 of 5 had no provenance" was the defect.
    for item in items:
        assert "source_uids" in item, f"shipped item carries no provenance key: {item}"
        assert item["source_uids"], f"shipped item carries no record attribution: {item}"
        assert set(item["source_uids"]) <= pool, (
            f"shipped item cites a record this run never minted: {item['source_uids']} not within {sorted(pool)}"
        )

    # (2) The attribution is the right one, not merely a well-formed one. The two
    # per-stack items name their own stack's record; the cross-stack item names a
    # record from every stack, which is the consolidation the field exists for.
    by_description = _source_uids_by_description(deep)
    assert by_description["Python issue"] == ["python:1"]
    assert by_description["React issue"] == ["react:1"]
    assert by_description["Contract drift between Python handler and React caller"] == [
        "generic:1",
        "python:1",
        "react:1",
    ]

    # (3) The host-appended structural items never see the merge agent, so the
    # host attributes them -- each to the single record it is.
    structural = [item for item in items if item.get("lens") == "structural"]
    assert structural, f"no structural item shipped: {items}"
    assert [item["source_uids"] for item in structural] == [["structure:1"]]


async def test_hallucinated_merge_source_uid_is_dropped_and_reported(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1111 real-path: an invented uid is discarded; the finding is not."""
    from daydream.deep.artifacts import deep_dir

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_source_uid_merge_resume(multi_stack_target)
    stub.merge_items = [
        # One real uid and one invention, on the same item: the real half must
        # survive the drop of the other rather than the whole list being voided.
        _provenance_item(1, "Partly attributed finding", source_uids=["python:1", "python:99"]),
        # Wholly invented, including a stack name this run has no records for.
        _provenance_item(2, "Wholly misattributed finding", file="App.tsx", source_uids=["ghost:7"]),
    ]

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    by_description = _source_uids_by_description(deep)
    # The real uid survives; the invention beside it is gone.
    assert by_description["Partly attributed finding"] == ["python:1"]
    # Fail-open: the wholly misattributed finding SHIPS, with an honest empty
    # provenance. Losing the finding to protect a bookkeeping field would be the
    # bug, not the fix.
    assert by_description["Wholly misattributed finding"] == []
    raw = (deep / "merged-items.json").read_text()
    assert "python:99" not in raw, f"invented uid reached the canonical artifact: {raw}"
    assert "ghost:7" not in raw, f"invented uid reached the canonical artifact: {raw}"

    # ONE aggregate warning for the whole merge (a model that misreads the uid
    # convention misreads it for every item at once), naming both bad uids and
    # how much provenance the run lost.
    out = _panel_text(capsys)
    assert (
        "Merge agent cited 2 source_uid(s) that match no record in this run "
        "(ghost:7, python:99); dropped them from item provenance. "
        "1 of 2 merged item(s) now carry no record attribution." in out
    ), out


async def test_unattributable_merge_source_uids_degrade_to_empty_list(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1111 real-path: null / missing / wrong-typed provenance ships as ``[]``."""
    from daydream.deep.artifacts import deep_dir

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_source_uid_merge_resume(multi_stack_target)
    stub.merge_items = [
        _provenance_item(1, "Explicit null provenance", source_uids=None),
        _provenance_item(2, "Omitted provenance key", file="App.tsx", omit_source_uids=True),
        _provenance_item(3, "Bare-string provenance", file="README.md", source_uids="python:1"),
    ]

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    by_description = _source_uids_by_description(deep)
    for description in (
        "Explicit null provenance",
        "Omitted provenance key",
        "Bare-string provenance",
    ):
        assert description in by_description, f"an unattributable finding was lost: {sorted(by_description)}"
        assert by_description[description] == [], by_description[description]

    # A null is a documented answer, not a bad claim: nothing was cited, so
    # nothing can have been invented, so the unknown-uid warning must stay quiet.
    out = _panel_text(capsys)
    assert "match no record in this run" not in out, out


async def test_single_stack_bypass_attributes_items_to_their_own_records(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """#1111 real-path: the tiny-diff bypass attributes items to their records."""
    from daydream.deep.artifacts import deep_dir

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects()

    assert await _run_deep(tiny_diff_target) == 0

    assert [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()] == [], (
        "the single-stack bypass must not invoke the merge agent"
    )
    deep = deep_dir(tiny_diff_target, allow_standalone=True)
    assert _run_uid_pool(deep) == {"generic:1", "structure:1"}
    by_description = _source_uids_by_description(deep)
    assert by_description["Sample issue"] == ["generic:1"]
    assert by_description["Structural maintainability concern"] == ["structure:1"]
    # Bypass items keep their birth ``uid`` as well, and the two answers agree --
    # the derivation of an item that IS a record is that record.
    for item in _merged_items(deep):
        assert item["source_uids"] == [item["uid"]], item


@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("base_evidenced", [True, False], ids=["base-survives", "structure-survives"])
async def test_structural_fold_survivor_inherits_both_provenances(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_evidenced: bool,
    delegated: bool,
) -> None:
    """Either fold direction preserves both record identities and the structural severity."""
    from daydream.deep.artifacts import deep_dir

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_source_uid_merge_resume(
        multi_stack_target,
        structure=[_record(description=_TWIN_DESCRIPTION, line=5, evidence="api.py:5", uid="structure:1")],
    )
    if delegated:
        from tests.test_finite_delegation_parse import _mark_delegated_artifacts

        _mark_delegated_artifacts(multi_stack_target / ".daydream/deep", {
            "python": ["api.py"], "react": ["App.tsx"], "generic": ["README.md"],
        })
    stub.merge_items = [
        _provenance_item(
            1,
            _TWIN_DESCRIPTION,
            line=5,
            evidence="api.py:5" if base_evidenced else "",
            source_uids=["python:1"],
        ),
        _provenance_item(2, "Unrelated react concern", file="App.tsx", source_uids=["react:1"]),
    ]

    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    deep = deep_dir(multi_stack_target, allow_standalone=True)
    twins = [item for item in _merged_items(deep) if item["description"] == _TWIN_DESCRIPTION]
    assert len(twins) == 1
    survivor = "base" if base_evidenced else "structural"
    source_uids = ["python:1", "structure:1"] if base_evidenced else ["structure:1", "python:1"]
    assert twins[0]["lens"] == ("per-stack" if base_evidenced else "structural")
    assert twins[0]["source_uids"] == source_uids
    assert twins[0]["severity"] == "high"
    assert _source_uids_by_description(deep)["Unrelated react concern"] == ["react:1"]

    folded = json.loads((deep / "folded-structural.json").read_text())
    assert folded["folded_count"] == 1
    assert folded["folded"][0]["survivor"] == survivor
    assert folded["folded"][0]["source_uids"] == source_uids


async def test_dropped_speculative_sidecar_records_item_provenance(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111 real-path: the evidence gate's sidecar records what it deleted."""
    from daydream.deep.artifacts import deep_dir

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_source_uid_merge_resume(multi_stack_target)
    stub.merge_items = [
        _provenance_item(1, "Grounded finding", source_uids=["python:1"]),
        _provenance_item(
            2,
            "Speculative unfounded finding",
            file="App.tsx",
            evidence="",
            rationale="inferred from the diff alone, no exploration evidence",
            source_uids=["react:1", "generic:1"],
        ),
    ]

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    by_description = _source_uids_by_description(deep)
    assert "Speculative unfounded finding" not in by_description, by_description
    assert by_description["Grounded finding"] == ["python:1"]

    dropped = json.loads((deep / "dropped-speculative.json").read_text())
    assert dropped["dropped_count"] == 1, dropped
    assert dropped["dropped_ids"] == [2]
    # Per item, not flattened -- the grouping IS the answer.
    assert dropped["dropped_source_uids"] == [["react:1", "generic:1"]]
    # The dropped object had no uid of its own (the merge agent authored it), so
    # its slot is "" (record_uid's own no-uid sentinel) -- but the array stays
    # positionally aligned with dropped_ids/dropped_source_uids rather than
    # being omitted.
    assert dropped["dropped_uids"] == [""]


@pytest.mark.anyio
async def test_legacy_artifact_backfills_structural_provenance_consistently(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-uid artifact must not yield two different answers in one run (#1111)."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    # Prime WITHOUT uid on disk -- the shape a run from before the field existed
    # left behind. Deliberately not `_prime_source_uid_merge_resume`, which primes
    # uids precisely to avoid this condition.
    _prime_merge_resume(
        multi_stack_target,
        python=[_record(description="py issue", evidence="api.py:1")],
        react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", line=5, evidence="api.py:5")],
    )
    records_path = multi_stack_target / ".daydream" / "deep" / "stack-structure-records.json"
    assert all("uid" not in rec for rec in _record_issues(json.loads(records_path.read_text()))), (
        "fixture must start with no uid on disk or it proves nothing"
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")

    assert exit_code == 0
    items = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())["items"]
    structural = [item for item in items if item.get("lens") == "structural"]
    assert structural, "the structural record must still reach the report"
    assert structural[0]["source_uids"] == ["structure:1"]


async def test_every_shipped_item_carries_a_unique_item_uid_on_a_multi_stack_run(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """#1111 real-path: every merged item ships with a distinct durable handle."""
    deep = await _fresh_uid_run(multi_stack_target, monkeypatch, mute_side_effects)
    items = _merged_items(deep)
    assert len(items) > 1, f"fixture must ship several items or uniqueness proves nothing: {items}"

    for item in items:
        assert "item_uid" in item, f"shipped item carries no durable identity key: {item}"
        assert isinstance(item["item_uid"], str) and item["item_uid"], (
            f"shipped item carries an empty durable identity: {item}"
        )
    uids = _item_uids(deep)
    assert len(set(uids)) == len(uids), f"two shipped items share one identity: {uids}"

    # Nothing in a fresh run arrives pre-stamped, so the minted handles track the
    # display ordinals exactly. That agreement is what makes the value
    # re-derivable for an artifact written before the field existed -- it is a
    # property of THIS case, not the definition of the field (see the preserve
    # tests, where the two deliberately diverge).
    assert uids == [f"item:{item['id']}" for item in items], uids


async def test_shipped_item_carries_id_item_uid_and_provenance_independently(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """#1111 real-path: three fields, three questions, one item dict."""
    deep = await _fresh_uid_run(multi_stack_target, monkeypatch, mute_side_effects)
    items = _merged_items(deep)
    # The display ordinal is dense and 1-based over the SHIPPED list -- the
    # evidence gate deletes items after the merge agent numbered them, so this
    # holds only because ``normalize_items`` renumbers the survivors.
    assert [item["id"] for item in items] == list(range(1, len(items) + 1)), items

    structural = [item for item in items if item.get("lens") == "structural"]
    assert structural, f"no structural item shipped: {items}"
    item = structural[0]
    # The birth record identity: this item WAS a per-stack record of the
    # ``structure`` meta-stack, and it never passed through the merge agent.
    assert item["uid"] == "structure:1", item
    # Its own item identity, from the same namespace every shipped item uses.
    assert item["item_uid"].startswith("item:"), item
    assert item["item_uid"] != item["uid"], f"the item identity collapsed onto the record identity: {item}"
    # Its derivation, which is the record it is -- a list, because an item can
    # be a synthesis of several records even though this one is not.
    assert item["source_uids"] == ["structure:1"], item
    # And the display ordinal, which is none of the above.
    assert isinstance(item["id"], int), item


@pytest.mark.parametrize("scratch_is_related", [False, True])
def test_retained_tree_uses_full_delta_identity_but_authorized_patch(
    tmp_path: Path,
    scratch_is_related: bool,
) -> None:
    """Unrelated/protected bytes invalidate evidence without entering the patch."""
    from daydream import git_ops
    from daydream.deep.fix_steps import FixCycleState, capture_retained_tree
    from daydream.fix_footprint import AuthorizedFixFootprint
    from daydream.workspace import WorkContext

    repo = tmp_path / "retained"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    (repo / "c.py").write_text("C = 1\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    (repo / "scratch.bin").write_bytes(b"\x00user")
    protected = git_ops.snapshot_untracked_paths(repo)
    item = {
        "id": 1,
        "item_uid": "item:1",
        "file": "a.py",
        "related_files": ["scratch.bin"] if scratch_is_related else [],
    }
    footprint = AuthorizedFixFootprint.build(repo, {"a.py"}, [item])
    index = git_ops.snapshot_index(repo)
    state = FixCycleState(
        session_id="s",
        stable_ref=git_ops.head_sha(repo),
        stable_head=git_ops.head_sha(repo),
        initial_index=index,
        preexisting_untracked=protected,
        preexisting_gitlinks=(),
        footprint=footprint,
    )
    (repo / "a.py").write_text("A = 2\n")
    first = capture_retained_tree(
        WorkContext(
            repo=repo,
            source=repo,
            base_branch="main",
            base_sha=state.stable_head,
            head_branch="main",
            head_sha=state.stable_head,
            is_ephemeral=False,
            run_id="s",
        ),
        state,
    )
    (repo / "c.py").write_text("C = 9\n")
    second = capture_retained_tree(
        WorkContext(
            repo=repo,
            source=repo,
            base_branch="main",
            base_sha=state.stable_head,
            head_branch="main",
            head_sha=state.stable_head,
            is_ephemeral=False,
            run_id="s",
        ),
        state,
    )
    assert first.paths == second.paths == frozenset({"a.py"})
    assert b"c.py" not in second.recommended_patch
    assert first.tree_key != second.tree_key


def test_retained_tree_includes_preexisting_authorized_head_delta(tmp_path: Path) -> None:
    """Commit selection is HEAD-relative even though evidence stays run-relative."""
    from daydream import git_ops
    from daydream.deep.fix_steps import FixCycleState, capture_retained_tree
    from daydream.fix_footprint import AuthorizedFixFootprint
    from daydream.workspace import WorkContext

    repo = tmp_path / "preexisting-retained"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 1\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    head = git_ops.head_sha(repo)

    # This reviewed edit predates the fix gate and is therefore part of the
    # stable rollback snapshot, but it still belongs in the eventual commit.
    (repo / "a.py").write_text("A = 2\n")
    stable_ref = git_ops.stash_create(repo) or head
    item = {
        "id": 1,
        "item_uid": "item:1",
        "file": "b.py",
        "related_files": ["a.py"],
    }
    footprint = AuthorizedFixFootprint.build(repo, {"a.py"}, [item])
    state = FixCycleState(
        session_id="s",
        stable_ref=stable_ref,
        stable_head=head,
        initial_index=git_ops.snapshot_index(repo),
        preexisting_untracked={},
        preexisting_gitlinks=(),
        footprint=footprint,
    )
    (repo / "b.py").write_text("B = 2\n")
    work = WorkContext(
        repo=repo,
        source=repo,
        base_branch="main",
        base_sha=head,
        head_branch="main",
        head_sha=head,
        is_ephemeral=False,
        run_id="s",
    )

    snapshot = capture_retained_tree(work, state)

    assert snapshot.paths == frozenset({"a.py", "b.py"})
    assert {path.path for path in snapshot.states} == {"a.py", "b.py"}
    assert b"a.py" in snapshot.recommended_patch
    assert b"b.py" in snapshot.recommended_patch
