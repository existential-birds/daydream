"""Referential identity for per-stack review records (issue #1111).

Per-stack records answer two different questions, and before this module the
pipeline used one kind of answer for both:

1. **"Do these describe the same defect?"** — legitimately content-derived.
   ``pr_review.compute_fingerprint`` and ``deep.dedup.descriptions_match`` both
   answer it, and both stay content-derived. This module does not touch them.
2. **"Which record object is this?"** — must *not* be content-derived, and had
   no representation at all.

The reviewer LLM's ``id`` cannot answer question 2: ``PER_STACK_RECORD_SCHEMA``
carries no ``uniqueItems`` constraint and the numbering restarts at 1 for every
stack, so ``id: 1`` is the norm rather than the exception. The first globally
unique handle a finding used to receive was minted by
``phases.normalize_items`` at the *final merge write* — after dedup,
arbitration, suppression and the structural fold had all already needed one. So
each of those stages invented its own surrogate: a positional list index, an
``id(record)`` object identity, a ``(file, line)`` tuple, an ``(id, file)``
tuple, or the ``source`` string. Every one of them gets *less* discriminating as
two records get more similar — and those sites only ever run on records selected
for being similar, so the key was weakest exactly where it was used.

``uid`` is the host-assigned answer to question 2, minted at record birth.

Format is ``f"{stack_name}:{ordinal}"`` (``python:1``, ``structure:3``), which
is deliberately readable in artifacts when debugging and deterministically
reproducible from ``(stack_name, position)``. That reproducibility is what makes
:func:`stamp_record_uids` safe to run against records written by an older run:
they simply lack the field, and re-deriving it from position restores the same
value the producing run would have minted. A uuid4 would be opaque and could not
be regenerated for pre-existing artifacts.

**Stamping is always host-side and post-validation.** ``run_agent`` validates
structured output against the strict schema *before returning it*, and nothing
re-validates a record dict afterwards, so a host-added key can never be
schema-rejected. This is the same pattern ``normalize_items`` already uses to
overwrite ``id``. None of the strict ``*_SCHEMA`` constants in ``phases.py``
gain a ``uid`` property — ``tests/test_output_schema_strict.py`` requires every
declared property to sit in ``required``, so declaring ``uid`` would force the
*model* to emit a field the host owns.

Scope note: ``uid`` is a **pre-merge** identity. The cross-stack merge agent
re-emits items from scratch under ``MERGED_ITEMS_SCHEMA``, so multi-stack
merged items carry no ``uid`` and do not need one — ``normalize_items`` already
gives every merged item a globally unique ``id``. Records that reach
``merged-items.json`` without passing through the merge agent (the single-stack
bypass, and the host-appended structural items) keep whichever ``uid`` they were
born with, so a ``uid`` on a merged item is a valid handle when present but is
never guaranteed to be there. Read it with :func:`record_uid` and treat the
empty string as "no pre-merge identity".
"""

from __future__ import annotations

from collections import Counter
from typing import Any

#: Record dict key holding the host-assigned referential identity.
RECORD_UID_KEY = "uid"

#: Separator between the stack-name and ordinal halves of a ``uid``. Stack names
#: never contain it, so ``rsplit`` on it recovers the stack name exactly.
_UID_SEPARATOR = ":"

# ``per_stack_records_path`` renders ``stack-{stack_name}-records.json``. The
# ``source`` strings threaded alongside records use that filename on every path
# that loads records off disk, and a bare stack name on the paths that append
# freshly parsed records in memory, so normalization has to accept both forms.
_RECORDS_FILENAME_PREFIX = "stack-"
_RECORDS_FILENAME_SUFFIX = "-records.json"


def mint_record_uid(stack_name: str, ordinal: int) -> str:
    """Return the ``uid`` for the *ordinal*-th record of *stack_name*.

    Args:
        stack_name: Owning stack, e.g. ``python`` or ``structure``.
        ordinal: 1-based position within that stack's record list.

    Returns:
        The ``uid`` string, e.g. ``python:1``.
    """
    return f"{stack_name}{_UID_SEPARATOR}{ordinal}"


def stack_name_from_records_source(source: str) -> str:
    """Normalize a record ``source`` string to a bare stack name.

    ``source`` reaches this function in either of two shapes, because the
    pipeline has two producers: the loop that loads records off disk tags them
    with the records *filename* (``stack-python-records.json``), while the
    uncovered-file sweep appends its records tagged with a bare stack name
    (``uncovered``). Both must normalize to the same key or a record routes to
    the wrong file — or to none at all.

    Args:
        source: A records filename or a bare stack name.

    Returns:
        The bare stack name. An unrecognized shape is returned unchanged, so a
        caller that cannot route it can detect that and say so rather than
        silently dropping the record.
    """
    if source.startswith(_RECORDS_FILENAME_PREFIX) and source.endswith(_RECORDS_FILENAME_SUFFIX):
        return source[len(_RECORDS_FILENAME_PREFIX) : -len(_RECORDS_FILENAME_SUFFIX)]
    return source


def stack_name_from_uid(uid: str) -> str:
    """Return the owning stack name encoded in *uid*.

    Splits on the LAST separator: the ordinal half is always a trailing integer,
    so this stays correct even if a stack name ever gained a separator
    character.

    Args:
        uid: A ``uid`` as minted by :func:`mint_record_uid`.

    Returns:
        The stack-name half, or ``""`` for a value carrying no separator.
    """
    stack_name, separator, _ordinal = uid.rpartition(_UID_SEPARATOR)
    return stack_name if separator else ""


def record_uid(record: dict[str, Any]) -> str:
    """Return *record*'s ``uid``, or ``""`` when it carries none.

    Returning the empty string rather than raising keeps this usable on
    post-merge items, where a ``uid`` is present only for items that bypassed
    the merge agent (see the module docstring).

    Args:
        record: A per-stack record or merged item dict.

    Returns:
        The ``uid`` string, or ``""`` when absent or not a string.
    """
    value = record.get(RECORD_UID_KEY)
    return value if isinstance(value, str) else ""


def stamp_record_uids(records: list[dict[str, Any]], stack_name: str) -> None:
    """Stamp a ``uid`` onto every record in *records* that lacks one, in place.

    Records are mutated in place rather than rebuilt, because the surrogate this
    field replaces was ``id(record)`` object identity: a stage that rebuilt a
    record as a fresh dict silently escaped those sets. Mutating in place means
    a caller holding the same list sees the stamp without having to re-bind
    anything, and no such escape is possible.

    An existing ``uid`` is **preserved**, never overwritten. Two reasons:

    - Records reloaded on a ``--start-at merge`` resume may have been written
      back by ``_rewrite_stack_records`` *after* adjudication dropped some of
      them, so the on-disk list is shorter than the list that produced those
      uids. Re-minting by position would hand out different uids than the run
      that created the artifacts.
    - Idempotence lets this be called at both record birth and record load
      without the second call fighting the first.

    Ordinals count every record in the list, stamped or not, so a partially
    stamped list cannot produce a collision by reusing an ordinal that an
    already-stamped record holds.

    Args:
        records: Record dicts to stamp, mutated in place.
        stack_name: Owning stack name; accepted in either ``source`` shape and
            normalized via :func:`stack_name_from_records_source`.
    """
    normalized = stack_name_from_records_source(stack_name)
    for ordinal, record in enumerate(records, start=1):
        if not record_uid(record):
            record[RECORD_UID_KEY] = mint_record_uid(normalized, ordinal)


def duplicate_record_uids(records: list[dict[str, Any]]) -> list[str]:
    """Return every ``uid`` held by more than one record in *records*.

    A duplicate ``uid`` would silently reintroduce exactly the over-delete this
    field exists to prevent, so callers assembling the cross-stack record pool
    check this and report rather than trusting the invariant. Records with no
    ``uid`` are ignored — absence is a separate, separately reported condition.

    Args:
        records: The assembled record pool to check.

    Returns:
        Sorted list of colliding ``uid`` values; empty when the pool is sound.
    """
    counts = Counter(uid for uid in (record_uid(record) for record in records) if uid)
    return sorted(uid for uid, count in counts.items() if count > 1)
