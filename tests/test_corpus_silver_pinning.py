"""Real-path proof that gold reads labels/rubric ONLY from the as_of-pinned silver rows.

Builds a real archive (production ``upsert_run`` + ``append_label_observation``)
where the denormalized ``runs.outcome_labels`` / ``runs.rubric_json`` caches are
forced to sentinel values that appear nowhere in the silver history, then runs
``daydream corpus build`` through the production entrypoint (``cli.main`` via
``sys.argv``; the archive resolves through ``DAYDREAM_ARCHIVE_DIR`` exactly as
in production). The emitted JSONL must carry the ``as_of``-pinned silver values
— including the temporal-leakage ``drop_label`` stripping — and the cache
sentinels must appear nowhere in the output bytes.

The ``--as-of`` is deliberately ``Z``-spelled: the config boundary must
canonicalize it to ``+00:00`` before it reaches the lexical ``observed_at <=
as_of`` SQL pin (stored ``observed_at`` is always ``+00:00``-spelled), so a
correct record set also proves the boundary normalization feeds the SQL pin.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from daydream.archive.index import append_label_observation
from tests.test_training_corpus import _seed_run_with_annotation

CACHE_SENTINEL = "cache-sentinel"
SILVER_RUBRIC_PINNED = {"marker": "silver-rubric-pinned"}
SILVER_RUBRIC_NEWER = {"marker": "silver-rubric-newer"}
SILVER_RUBRIC_POSTERIOR = {"marker": "silver-rubric-posterior"}


def _run_cli(argv: list[str]) -> int:
    """Drive ``cli.main`` (the production entrypoint) and return its exit code."""
    from daydream import cli

    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


def _seed_divergent_archive(archive_dir: Path) -> None:
    """Seed two runs whose runs-table caches diverge from the pinned silver rows.

    - ``s-pinned``: silver generation A (observed before the pin, label
      ``accepted``, rubric A) plus a newer generation B (observed after the
      pin, label ``rejected``, rubric B) that wins the denormalized cache.
      The pin must resolve generation A.
    - ``s-posterior``: one silver generation observed before the pin but with
      ``valid_at`` after it (posterior outcome) — the leakage guard must strip
      its label while the rubric (capture-time) survives.

    Both runs then get their ``runs.outcome_labels`` / ``runs.rubric_json``
    caches overwritten with sentinels so any cache read is unmissable in the
    output.
    """
    _seed_run_with_annotation(
        archive_dir, "s-pinned", label="accepted",
        rubric_json=json.dumps(SILVER_RUBRIC_PINNED),
        observed_at="2026-03-01T00:00:00+00:00",
        valid_at="2026-03-01T00:00:00+00:00",
    )
    # Newer silver generation (different label + evidence so the auto-dedup
    # doesn't swallow it); it wins the cache but is observed AFTER the pin.
    append_label_observation(
        archive_dir, "s-pinned",
        labels=["rejected"], pr_state="closed", labeler_version="v2",
        evidence_sha="e2", rubric_json=json.dumps(SILVER_RUBRIC_NEWER),
        valid_at="2026-05-01T00:00:00+00:00", reward_version="r2",
    )
    _seed_run_with_annotation(
        archive_dir, "s-posterior", label="accepted",
        rubric_json=json.dumps(SILVER_RUBRIC_POSTERIOR),
        observed_at="2026-03-01T00:00:00+00:00",
        valid_at="2026-09-01T00:00:00+00:00",  # outcome true only after the pin
    )

    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        # append_label_observation stamps wall clock; pin generation B after
        # the as_of deterministically (same precedent as the seeding helper).
        conn.execute(
            "UPDATE label_observations SET observed_at = ? "
            "WHERE session_id = 's-pinned' AND labels = ?",
            ("2026-05-01T00:00:00+00:00", json.dumps(["rejected"])),
        )
        # Force the denormalized caches to diverge from every silver row.
        conn.execute(
            "UPDATE runs SET outcome_labels = ?, rubric_json = ?",
            (json.dumps([CACHE_SENTINEL]), json.dumps({"marker": CACHE_SENTINEL})),
        )
        conn.commit()
    finally:
        conn.close()


def test_gold_reads_only_the_as_of_pinned_silver_rows(tmp_path, archive_dir):
    _seed_divergent_archive(archive_dir)
    out = tmp_path / "corpus.jsonl"

    rc = _run_cli([
        "corpus", "build",
        "--out", str(out),
        "--include-all-labels",
        "--as-of", "2026-04-01T00:00:00Z",  # Z-spelled: boundary must canonicalize
    ])

    assert rc == 0
    raw = out.read_bytes()
    records = {r["session_id"]: r for r in map(json.loads, raw.decode().splitlines())}
    assert set(records) == {"s-pinned", "s-posterior"}

    # Label and rubric come from the as_of-pinned generation A — not the newer
    # generation B that owns the cache, and never the cache itself.
    assert records["s-pinned"]["outcome_label"] == "accepted"
    assert records["s-pinned"]["rubric"] == SILVER_RUBRIC_PINNED

    # drop_label stripping: the pinned row's outcome is posterior to the pin,
    # so the label is stripped even though the silver row (and the cache) says
    # "accepted"; the capture-time rubric still comes from silver.
    assert records["s-posterior"]["outcome_label"] is None
    assert records["s-posterior"]["rubric"] == SILVER_RUBRIC_POSTERIOR

    # No cache value leaked anywhere in the emitted bytes.
    assert CACHE_SENTINEL.encode() not in raw
    assert SILVER_RUBRIC_NEWER["marker"].encode() not in raw


def test_cli_rejects_invalid_as_of_through_main(tmp_path, archive_dir):
    out = tmp_path / "corpus.jsonl"
    rc = _run_cli(["corpus", "build", "--out", str(out), "--as-of", "2026-04-01T05:00:00+05:00"])
    assert rc == 1
    assert not out.exists()

    rc = _run_cli(["corpus", "build", "--out", str(out), "--as-of", "not-a-timestamp"])
    assert rc == 1
    assert not out.exists()
