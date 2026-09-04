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

#: Merged-item key holding the list of record ``uid``s the item derives from.
#: A merged item is a synthesis, possibly of several records, so its provenance
#: is a list where a record's identity is a single value.
RECORD_SOURCE_UIDS_KEY = "source_uids"

#: Merged-item key holding the item's own durable identity.
#:
#: Deliberately NOT :data:`RECORD_UID_KEY`. A merged item can carry both: the
#: host-appended structural items keep the ``uid`` of the record they were born
#: as, and would still need an item identity on top of it. Overloading one key
#: would make ``record_uid`` return an item identity for some items and a record
#: identity for others -- and ``item_source_uids`` falls back to ``record_uid``,
#: so an item identity would silently masquerade as provenance.
ITEM_UID_KEY = "item_uid"

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

    A minted ordinal SKIPS any value an already-stamped record in the list
    holds. Counting positions alone is not sufficient and was a real defect
    here: a list like ``[{uid: python:2}, {}]`` -- a preserved record sitting
    anywhere other than its original position -- would hand the unstamped
    record ``python:2`` as well, emitting the duplicate this field exists to
    make impossible. On a fully unstamped list (the backfill case) skipping
    changes nothing, so ordinals still equal positions and the value stays
    re-derivable from ``(stack_name, position)``.

    Args:
        records: Record dicts to stamp, mutated in place.
        stack_name: Owning stack name; accepted in either ``source`` shape and
            normalized via :func:`stack_name_from_records_source`.
    """
    normalized = stack_name_from_records_source(stack_name)
    taken = {uid for uid in (record_uid(record) for record in records) if uid}
    ordinal = 0
    for record in records:
        if record_uid(record):
            continue
        ordinal += 1
        while (candidate := mint_record_uid(normalized, ordinal)) in taken:
            ordinal += 1
        record[RECORD_UID_KEY] = candidate
        taken.add(candidate)


def stamp_item_uids(items: list[dict[str, Any]]) -> None:
    """Stamp a durable :data:`ITEM_UID_KEY` onto every item lacking one, in place.

    The merged-item counterpart to :func:`stamp_record_uids`, with the same
    preserve-if-present and skip-taken-ordinal semantics and for the same
    reasons. Preserving matters more here than for records: ``normalize_items``
    reassigns ``id`` on every call by design, so an item that already holds an
    identity must keep it precisely *while* its display ordinal changes -- that
    divergence is the whole reason the two fields are not one.

    Args:
        items: Merged item dicts to stamp, mutated in place.
    """
    taken = {uid for uid in (item_uid(item) for item in items) if uid}
    ordinal = 0
    for item in items:
        if item_uid(item):
            continue
        ordinal += 1
        while (candidate := mint_item_uid(ordinal)) in taken:
            ordinal += 1
        item[ITEM_UID_KEY] = candidate
        taken.add(candidate)


def mint_item_uid(ordinal: int) -> str:
    """Return the durable identity for the *ordinal*-th merged item.

    Args:
        ordinal: 1-based position in the merged item list at mint time.

    Returns:
        The item uid, e.g. ``item:3``.
    """
    return f"item{_UID_SEPARATOR}{ordinal}"


def item_uid(item: dict[str, Any]) -> str:
    """Return *item*'s own durable identity, or ``""`` when it carries none.

    This is the post-merge counterpart to :func:`record_uid`, and it answers a
    different question from :func:`item_source_uids`: *which shipped finding is
    this*, not *what was it made of*. Provenance cannot serve as identity --
    nothing stops two merged items citing the same record (the merge agent may
    split one record into two findings), so ``source_uids`` is not unique.

    Args:
        item: A merged finding item.

    Returns:
        The item uid string, or ``""`` when absent or not a string.
    """
    value = item.get(ITEM_UID_KEY)
    return value if isinstance(value, str) else ""


def item_source_uids(item: dict[str, Any]) -> list[str]:
    """Return the pre-merge record uids *item* derives from, in order.

    This is the post-merge counterpart to :func:`record_uid`. A merged item is
    not a record: the cross-stack merge agent synthesizes items, and one item
    may consolidate several records, so its provenance is a *list* rather than a
    single handle. ``source_uids`` carries that list.

    Resolution order, so every path a merged item can arrive by reports the same
    way:

    1. An explicit ``source_uids`` list (the merge agent's attribution, already
       validated host-side against the run's record pool).
    2. Failing that, the item's own ``uid`` — the case for items that never went
       through the merge agent and therefore kept their birth identity: the
       host-appended structural items, the single-stack bypass, and the salvage
       path.
    3. Failing both, empty — the item has no pre-merge provenance. That is a
       real answer, not an error: an item the merge agent declined to attribute
       reports nothing rather than a fabricated link.

    Args:
        item: A merged finding item.

    Returns:
        Deduplicated uids in first-seen order; empty when provenance is unknown.
    """
    raw = item.get(RECORD_SOURCE_UIDS_KEY)
    if isinstance(raw, list):
        attributed = union_source_uids(uid for uid in raw if isinstance(uid, str) and uid)
        if attributed:
            return attributed
    own = record_uid(item)
    return [own] if own else []


def union_source_uids(*uid_groups: Any) -> list[str]:
    """Union uid sequences into one deduplicated, first-seen-order list.

    Used wherever two findings collapse into one and the survivor must inherit
    both provenances -- the structural fold being the case that matters, since a
    folded item represents a defect two lenses reported. Order is first-seen
    rather than sorted so the survivor's own provenance leads, which keeps the
    audit sidecar readable.

    Accepts either a single iterable of uids or several, so callers can union
    two items' lists without pre-flattening.

    Args:
        *uid_groups: Iterables of uid strings (or one such iterable).

    Returns:
        Deduplicated uids in first-seen order, empty strings excluded.
    """
    seen: dict[str, None] = {}
    for group in uid_groups:
        for uid in group:
            if isinstance(uid, str) and uid:
                seen.setdefault(uid, None)
    return list(seen)


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
