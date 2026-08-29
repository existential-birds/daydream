"""Resumable keyboard-driven terminal curation client.

Pure-client UI over :mod:`daydream.benchmark.curation`: every mutating action
``[a/e/n/x/c/r/d/z/i/q]`` maps one-to-one onto a service operation and never
mutates the case YAML/model directly. Rendering is plain-string builders for
deterministic tests; Rich stays available for live styling.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import ValidationError

from daydream.benchmark import curation as cu
from daydream.benchmark.harbor import build
from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict


def parse_indices(spec: str, n: int) -> list[int]:
    """Parse a comma-separated 1-based selector into sorted unique 0-based indices.

    Accepts single numbers and a single ``a-b`` range (a reversed ``b-a`` range
    spans ``a..b`` inclusive). Raises :class:`ValueError` for any index or range
    endpoint outside ``1..n`` (including ``0``), a repeated index or an
    overlapping range, non-numeric or empty tokens, a single-point range, and
    multiple range tokens.
    """
    if not spec or not spec.strip():
        raise ValueError("empty index selector")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    tokens = spec.split(",")
    if sum(1 for t in tokens if "-" in t) > 1:
        raise ValueError(f"multiple ranges not allowed in {spec!r}")
    selected: set[int] = set()
    for token in tokens:
        if not token:
            raise ValueError(f"empty segment in {spec!r}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"malformed range {token!r}")
            if not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"non-numeric range endpoint in {token!r}")
            start, end = int(parts[0]), int(parts[1])
            if start == end:
                raise ValueError(f"range {token!r} is a single point")
            low, high = min(start, end), max(start, end)
            if low < 1 or high > n:
                raise ValueError(f"range endpoint out of range in {token!r}")
            span = set(range(low, high + 1))
            if span & selected:
                raise ValueError(f"range {token!r} overlaps the selection")
            selected |= span
        else:
            if not token.isdigit():
                raise ValueError(f"non-numeric index {token!r}")
            index = int(token)
            if index < 1 or index > n:
                raise ValueError(f"index out of range {token!r}")
            if index in selected:
                raise ValueError(f"repeated index {token!r}")
            selected.add(index)
    return sorted(i - 1 for i in selected)


_INDEX_COLUMNS = (
    "case_id",
    "pr_number",
    "head_prefix",
    "changed_files",
    "changed_lines",
    "evidence_count",
    "state",
    "gold_mode",
    "gold_count",
)


def render_index_table(cases: list[dict[str, Any]]) -> str:
    """Render a plain-text index header + one row per case (every value ``str``)."""
    lines = [" | ".join(_INDEX_COLUMNS)]
    for case in cases:
        lines.append(" | ".join(str(case.get(k, "-")) for k in _INDEX_COLUMNS))
    return "\n".join(lines)


def _prompt(read_line: Callable[[str], str], message: str) -> str:
    """Print *message* to stdout, then read one input line with *read_line*."""
    print(message, end="", flush=True)
    return read_line("")


def _evidence_entries(view: dict[str, Any]) -> list[Any]:
    """The ordered evidence entries of *view* in canonical order.

    Prefers the full import-evidence list and falls back to ``candidates`` only
    for a minimal dict without an ``evidence`` key. The prioritized render and
    every number-based action instead number through the captured view binding
    (:func:`_view_binding`); this canonical order remains the fallback path for
    a minimal dict and the import-order source of ``position``.
    """
    return view.get("evidence") or view.get("candidates") or []


class _ViewBinding(list):  # type: ignore[type-arg]
    """The captured ordered entry→source_id map formed at render time.

    A ``list[str]`` of source_ids in exactly the order :func:`render_case`
    numbered them, carrying the digest of the view it was captured from so
    :func:`_binding_stale` can detect any post-render case mutation.
    """

    def __init__(self, source_ids: list[str], digest: str) -> None:
        super().__init__(source_ids)
        self.digest = digest


def _view_digest(view: dict[str, Any]) -> str:
    """A stable digest of one :func:`cu.get_case` view (never persisted)."""
    return hashlib.sha256(
        json.dumps(view, sort_keys=True, default=str).encode()
    ).hexdigest()


def _view_binding(view: dict[str, Any]) -> _ViewBinding:
    """The captured entry→source_id map for *view*, in render order.

    One source of numbering for render/page/edit/exclude/accept: the
    prioritized entries when the case carries facts (and candidates), else the
    canonical evidence order (the minimal-dict fallback).
    """
    prioritized = (view.get("prioritized_evidence") or {}).get("entries") or []
    if prioritized and view.get("prioritization") is not None and view.get("candidates"):
        source_ids = [e["source_id"] for e in prioritized]
    else:
        source_ids = [ev.get("source_id") or "" for ev in _evidence_entries(view)]
    return _ViewBinding(source_ids, _view_digest(view))


def _resolve_number(number: int, binding: list[str]) -> str | None:
    """Resolve a 1-based displayed *number* through the captured *binding*.

    Returns the source_id the render displayed at that number, or ``None``
    when out of range — never a fresh re-derivation.
    """
    index = number - 1
    if not (0 <= index < len(binding)):
        return None
    return binding[index] or None


def _binding_stale(root: Path, case_id: str, binding: list[str]) -> bool:
    """True when the case changed after *binding* was captured at render time.

    Compares the render-time view digest against a fresh read, so any case-doc
    mutation (finding, exclusion, state transition) invalidates the captured
    numbering instead of letting an old number reinterpret fresh content.
    """
    fresh = _view_binding(cu.get_case(root, case_id))
    if isinstance(binding, _ViewBinding):
        return binding.digest != fresh.digest
    return list(binding) != list(fresh)


def _check_fresh(root: Path, case_id: str, binding: list[str]) -> bool:
    """Print the rerender prompt and return ``False`` when *binding* is stale."""
    if not _binding_stale(root, case_id, binding):
        return True
    print("view changed — rerendering")
    return False


def _entry_block(
    ev: dict[str, Any],
    candidates: list[dict[str, Any]],
    number: int,
    entry: dict[str, Any] | None,
) -> str:
    """The numbered detail block for one evidence record.

    *entry* is the prioritized projection entry (band/reasons/disposition) when
    the case renders sectioned, else ``None`` for the canonical fallback.
    """
    cand = None
    cand_index = ev.get("candidate_index")
    if cand_index is not None and 0 <= cand_index < len(candidates):
        cand = candidates[cand_index]
    elif isinstance(ev.get("title"), str):
        # minimal candidate-only fallback stores its own metadata.
        cand = ev
    author = ev.get("author") or {}
    login = author.get("login") or "-"
    if author.get("type") == "Bot":
        login = f"{login}[bot]"
    kind = ev.get("kind") or "-"
    commit = (ev.get("commit_id") or "")[:12] or "-"
    rec_state = ev.get("state")
    state_tag = f" {rec_state}" if rec_state else ""
    loc_path = ev.get("path")
    start = ev.get("start_line") or ev.get("line")
    if cand is not None:
        cand_loc = cand.get("location") or {}
        loc_path = loc_path or cand_loc.get("path")
        start = start or cand_loc.get("start_line")
    loc_path = loc_path or "-"
    anchor = f"{loc_path}:{start}" if loc_path != "-" and start else loc_path
    markers = ""
    if ev.get("resolved"):
        markers += " [resolved]"
    if ev.get("outdated"):
        markers += " [outdated]"
    authoring = ""
    auth_anchor = ev.get("authoring_anchor")
    if isinstance(auth_anchor, dict) and auth_anchor.get("commit_id"):
        authoring = f" auth:{auth_anchor['commit_id'][:12]}"
    reason = ""
    if cand is not None and not cand.get("exact_acceptable"):
        reason_tag = cand.get("not_exact_reason")
        if reason_tag:
            reason = f" [{reason_tag}]"
    advisory = ""
    if entry is not None:
        parts = []
        if entry.get("reasons"):
            parts.append("reasons: " + ",".join(entry["reasons"]))
        parts.append(f"disp: {entry.get('disposition') or '-'}")
        advisory = " (" + "; ".join(parts) + ")"
    preview = (ev.get("body") or "").replace("\n", " ")[:120]
    src = ev.get("source_id") or ""
    src_tag = f" {src}" if src else ""
    line_parts = [
        f"  {number}. [{kind}] {login} commit:{commit}{authoring}{reason}{state_tag}"
        f" {anchor}{markers}{advisory}{src_tag}",
        f"      {preview or '-'}",
    ]
    if cand is not None:
        line_parts.insert(1, f"      {cand.get('title') or '-'}")
    return "\n".join(line_parts)


_LEGEND = (
    "legend: resolved/outdated = already addressed; "
    "anchor-delta-changed/deleted/renamed/binary = anchor drifted since authoring; "
    "pr-author-reply = PR author replied; "
    "commit-non-ancestor/commit-unavailable/anchor-unavailable/facts-missing = could not verify at head; "
    "dismissed/decided-by-finding/decided-by-exclusion/decided-by-conflict/non-candidate = classification"
)


def render_case(case: dict[str, Any]) -> str:
    """Render the snapshot header plus the numbered evidence view.

    When the case carries prioritization facts, evidence renders in labeled
    band sections (:data:`cu.BAND_RANK` order, only non-empty bands), each
    entry keeping its detail line plus advisory reason/disposition markers,
    closed by a legend line decoding the reason codes. Numbering comes from the
    captured view binding (:func:`_view_binding`) — the same order every
    number-based action resolves through. When every candidate lacks facts,
    the canonical order renders with a ``needs_judgment`` note instead; a
    minimal dict without an ``evidence`` key falls back to ``candidates``.

    Every evidence entry shows its number, kind, author login (+ ``[bot]`` for a
    bot), the re-anchored ``commit_id`` prefix (labeled ``commit:``) and — where
    the record carries a strict ``authoring_anchor`` — the authoring commit
    prefix (labeled ``auth:``), the fixed ``not_exact_reason`` whenever exact
    acceptance is unavailable, and — where the record carries one — its review
    ``state`` (e.g. ``APPROVED``/``COMMENTED``/``CHANGES_REQUESTED``), the
    ``path:line`` anchor, resolved/outdated markers, and a body preview (first
    ~120 chars). Candidate records additionally show their title.
    """
    snapshot = case.get("snapshot") or {}
    curation = case.get("curation") or {}
    head = snapshot.get("original_head_sha") or "-"
    policy = snapshot.get("policy") or "-"
    state = curation.get("state") or "-"
    candidates = case.get("candidates") or []
    prioritized = (case.get("prioritized_evidence") or {}).get("entries") or []
    facts = case.get("prioritization")
    lines = [
        f"case {case.get('case_id')}: state={state} policy={policy} head={head}",
    ]
    if prioritized and facts is not None and candidates:
        records = case.get("evidence") or []
        binding_ids: list[str] = []
        for band in sorted(cu.BAND_RANK, key=cu.BAND_RANK.__getitem__):
            group = [e for e in prioritized if e["band"] == band]
            if not group:
                continue
            lines.append(f"-- {band} --")
            for entry in group:
                rec = records[entry["position"]]
                number = len(binding_ids) + 1
                binding_ids.append(entry["source_id"])
                lines.append(_entry_block(rec, candidates, number, entry))
        lines.append(_LEGEND)
        return "\n".join(lines)
    entries = _evidence_entries(case)
    if prioritized and candidates and facts is None:
        lines.append("note: needs_judgment — no prioritization facts for any candidate")
    for i, ev in enumerate(entries, start=1):
        lines.append(_entry_block(ev, candidates, i, None))
    return "\n".join(lines)


_ACTIONS = frozenset("aenxcrdziq")
_ACTION_PROMPT = "action [a/e/n/x/c/r/d/z/i/q]: "


def _pick_editor() -> str:
    """Deterministic editor resolution: ``$VISUAL`` -> ``$EDITOR`` -> ``vi``."""
    for var in ("VISUAL", "EDITOR"):
        value = os.environ.get(var)
        if value:
            return value
    return "vi"


def _launch_editor(initial: str) -> str | None:
    """Open *initial* in the user's editor; return edited text or ``None``.

    A ``0600`` temp ``.yaml`` buffer is created and removed in a ``finally`` so
    an editor interrupt cannot leak it. A nonzero editor exit removes the buffer
    and returns ``None`` (no mutation).
    """
    editor = _pick_editor()
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(initial)
        try:
            proc = subprocess.run([*shlex.split(editor), path], text=False)
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode != 0:
            return None
        return Path(path).read_text()
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _service_error(exc: BaseException, outcome: str = "continue") -> str:
    """Print a curation error message and return the action-loop *outcome*.

    Shared by every ``_action_*`` handler so a typed ``cu.CurationError``
    (or a ``ValidationError`` from a malformed atom) renders as a one-line
    message plus a verdict, never a bare traceback.
    """
    print(str(exc))
    return outcome


def _editor_fragment_new() -> str:
    """One blank template finding as an editable YAML fragment."""
    return yaml.safe_dump({"findings": [{
        "title": "", "body": "", "severity": None,
        "location": None, "source_ids": [],
    }]}, sort_keys=False)


def _editor_fragment_edit(finding: dict[str, Any]) -> str:
    """The existing *finding* as one editable YAML fragment atom."""
    atom = {
        "title": finding.get("title"),
        "body": finding.get("body"),
        "severity": finding.get("severity"),
        "location": finding.get("location"),
        "source_ids": (finding.get("provenance") or {}).get("source_ids") or [],
    }
    return yaml.safe_dump({"findings": [atom]}, sort_keys=False)


def _editor_fragment_authored(source_ids: list[str]) -> str:
    """One blank atom pre-filled with the selected evidence *source_ids*."""
    return yaml.safe_dump({"findings": [{
        "title": "", "body": "", "severity": None,
        "location": None, "source_ids": source_ids,
    }]}, sort_keys=False)


def _parse_fragment(text: str) -> list[dict[str, Any]] | None:
    """Parse edited YAML into a list of non-blank finding atoms, or ``None``.

    Requires a dict with a non-empty ``findings`` list of atoms each carrying a
    non-blank ``title``/``body`` string."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    findings = data.get("findings")
    if not isinstance(findings, list) or not findings:
        return None
    atoms: list[dict[str, Any]] = []
    for item in findings:
        if not isinstance(item, dict):
            return None
        title = item.get("title")
        body = item.get("body")
        if not isinstance(title, str) or not title.strip():
            return None
        if not isinstance(body, str) or not body.strip():
            return None
        atoms.append(item)
    return atoms


def _edit_and_stage_fragment(
    initial: str,
    stage: Callable[[list[dict[str, Any]]], None],
    *,
    success: Callable[[int], str],
    err_outcome: str = "continue",
) -> str:
    """The shared edit-a-fragment -> stage-through-service scaffold.

    Launches the editor over *initial*, parses its non-blank atoms, then stages
    them through *stage*. Editor cancellation, an invalid fragment, a curation
    error or a validation error all print a one-line message, mutate nothing,
    and return *err_outcome* (``continue`` by default; ``rerender`` for the
    ``[n]`` add path). On success reports *success(len(atoms))* and returns
    ``"rerender"``.
    """
    text = _launch_editor(initial)
    if text is None:
        print("editor cancelled or failed; nothing written")
        return "continue"
    atoms = _parse_fragment(text)
    if atoms is None:
        print("invalid fragment; nothing written")
        return "continue"
    try:
        stage(atoms)
    except (cu.CurationError, ValidationError) as exc:
        return _service_error(exc, err_outcome)
    print(success(len(atoms)))
    return "rerender"


def _action_new(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[n]`` author action: edit a blank fragment, add every atom."""
    del binding
    return _edit_and_stage_fragment(
        _editor_fragment_new(),
        lambda atoms: cu.add_findings(root, case_id, findings=atoms),
        success=lambda n: f"added {n} authored finding(s)",
        err_outcome="rerender",
    )


def _action_edit(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[e]`` edit-or-author action.

    Prompts for a finding to rewrite (the existing finding-rewrite path) or
    ``a`` to author edited finding(s) from selected evidence via
    :func:`add_edited_findings`. The evidence selector resolves through the
    captured view binding; a stale binding rerenders instead of acting.
    """
    findings = (view.get("curation") or {}).get("findings") or []
    first = _prompt(
        read_line,
        "edit finding (number) or author from evidence [a] (0 to cancel): ",
    ).strip()
    if first.lower() == "a":
        if not _check_fresh(root, case_id, binding):
            return "rerender"
        return _edit_author_evidence(root, case_id, binding, read_line)
    text = first
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(findings))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    if len(indices) != 1:
        print(f"edit takes exactly one finding (got {len(indices)})")
        return "continue"
    finding = findings[indices[0]]
    return _edit_and_stage_fragment(
        _editor_fragment_edit(finding),
        lambda atoms: cu.replace_findings(root, case_id, finding["finding_id"], replacements=atoms),
        success=lambda n: f"replaced finding with {n} atom(s)",
    )


def _edit_author_evidence(
    root: Path, case_id: str, binding: list[str], read_line: Callable[[str], str]
) -> str:
    """The ``[e]``\u2192[author-from-evidence] sub-flow.

    Parses one 1-based evidence selector (a number or a single ``a-b`` range)
    against the captured view binding, pins the selected source_ids into a
    blank atom, opens the editor, then stages the atoms through
    :func:`add_edited_findings`. Editor cancellation, an invalid fragment, a
    curation error or a validation error all mutate nothing.
    """
    text = _prompt(read_line, "evidence (number or range, 0 to cancel): ").strip()
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(binding))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    source_ids = [binding[i] for i in indices]
    if any(not sid for sid in source_ids):
        print("selected entry has no source_id")
        return "continue"
    return _edit_and_stage_fragment(
        _editor_fragment_authored(source_ids),
        lambda atoms: cu.add_edited_findings(root, case_id, atoms=atoms),
        success=lambda n: f"authored {n} edited finding(s)",
    )


# Single source of truth lives on the curation service; re-export here so a
# reason added on one side cannot drift apart from the other.
_EVIDENCE_REASONS = cu._EVIDENCE_REASONS
_CASE_EXCLUSION_REASONS = cu._CASE_EXCLUSION_REASONS


def _action_exclude_case(
    root: Path,
    case_id: str,
    view: dict[str, Any],
    read_line: Callable[[str], str],
    binding: list[str],
) -> str:
    """The ``[z]`` case-exclusion action (4 fixed reasons + optional note)."""
    del binding
    reason = _prompt(
        read_line, f"reason ({'|'.join(_CASE_EXCLUSION_REASONS)}): "
    ).strip()
    if reason not in _CASE_EXCLUSION_REASONS:
        print(f"invalid case exclusion reason {reason!r}")
        return "continue"
    note: str | None = None
    if reason == "other":
        value = _prompt(read_line, "note: ").strip()
        if value in ("q", "Q"):
            return "quit"
        if not value:
            print("case exclusion reason 'other' requires a note")
            return "continue"
        note = value
    try:
        cu.exclude_case(root, case_id, reason, note=note)
    except cu.CurationError as exc:
        return _service_error(exc)
    print(f"excluded case {case_id}")
    return "rerender"


def _action_reinclude(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[i]`` re-include action for an excluded case."""
    del binding
    try:
        cu.reinclude_case(root, case_id)
    except cu.CurationError as exc:
        return _service_error(exc)
    print(f"re-included case {case_id}")
    return "rerender"


def _action_clean(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[c]`` clean-attest action (requires an empty gold findings set)."""
    del binding
    findings = (view.get("curation") or {}).get("findings") or []
    if findings:
        print("[c] requires an empty gold findings set")
        return "continue"
    answer = _prompt(
        read_line, f"Mark {case_id} as reviewed clean with zero expected findings? [y/N] "
    ).strip()
    if answer.lower() != "y":
        return "continue"
    try:
        cu.attest_clean(root, case_id)
    except cu.CurationError as exc:
        return _service_error(exc)
    print(f"attested {case_id} clean")
    return "rerender"


def _action_ready(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[r]`` mark-ready action: pages the exact Task Spec and asks one combined question.

    Renders the byte-deterministic Task.md for the case (the same bytes the
    compiler verifies and writes), prints it, then asks the single combined
    approve-spec + attest-review question. Only a literal ``y`` proceeds: the
    digest is derived by :func:`mark_ready` **under the workspace lock** from
    the exact case being marked ready, so the approved digest is never a stale
    pre-lock render and cannot abort a later whole-workspace compile. Anything
    else is a no-op leaving the case ``draft``.
    """
    del binding
    head = (view.get("snapshot") or {}).get("original_head_sha") or ""
    raw = load_yaml_strict(Path(root) / "cases" / f"{case_id}.yaml")
    spec_bytes = build.render_task_spec(raw, instruction=build.ASSIGNMENT_TEXT)
    print(spec_bytes.decode("utf-8"))
    answer = _prompt(
        read_line,
        f"Approve this Task Spec and attest that this golden review is valid "
        f"against head {head} and mark {case_id} ready? [y/N] ",
    ).strip()
    if answer.lower() != "y":
        return "continue"
    try:
        cu.mark_ready(root, case_id, head_sha=head)
    except cu.CurationError as exc:
        return _service_error(exc)
    print(f"marked {case_id} ready")
    return "rerender"


def _action_exclude(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[x]`` evidence-exclusion action over the captured view binding.

    Parses one 1-based selector (a single index or one ``a-b`` range) against
    the binding the render numbered. A stale binding prints the rerender prompt
    and re-renders without acting. The reason/note contract is validated
    **before** any mutation, then every selected evidence source is excluded
    with the same reason/note. A bad range, an invalid reason, or a missing
    ``other`` note mutates nothing. Single-index selections keep the original
    note prompt (a stray note on a non-``other`` reason is still rejected); a
    range with reason ``other`` prompts for the single note applied to the
    whole range, so a range exclusion is never a dead-end the service
    immediately rejects.
    """
    if not _check_fresh(root, case_id, binding):
        return "rerender"
    text = _prompt(read_line, "evidence (number or range, 0 to cancel): ").strip()
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(binding))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    source_ids = [binding[i] for i in indices]
    if any(not sid for sid in source_ids):
        print("selected entry has no source_id")
        return "continue"
    reason = _prompt(
        read_line, f"reason ({'|'.join(_EVIDENCE_REASONS)}): "
    ).strip()
    if reason not in _EVIDENCE_REASONS:
        print(f"invalid evidence reason {reason!r}")
        return "continue"
    note: str | None = None
    if reason == "other":
        note = _prompt(read_line, "note: ").strip() or None
        if not note:
            print("reason 'other' requires a note")
            return "continue"
    elif len(indices) == 1:
        note = _prompt(read_line, "note: ").strip() or None
    try:
        cu.exclude_evidence_batch(root, case_id, source_ids, reason=reason, note=note)
    except cu.CurationError as exc:
        return _service_error(exc)
    print(f"excluded {len(indices)} evidence source(s)")
    return "rerender"


def _action_accept(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """The ``[a]`` accept-candidate action: one exact-acceptable candidate.

    The candidate number resolves through the captured view binding — the same
    numbers the render displayed — and a stale binding rerenders instead of
    re-interpreting the number against fresh content.
    """
    if not _check_fresh(root, case_id, binding):
        return "rerender"
    candidates = view.get("candidates") or []
    text = _prompt(read_line, "candidate (number, 0 to cancel): ").strip()
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(binding))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    if len(indices) != 1:
        print(f"accept takes exactly one candidate (got {len(indices)})")
        return "continue"
    sid = binding[indices[0]]
    cand = next((c for c in candidates if c.get("source_id") == sid), None)
    if cand is None:
        print(f"no candidate number {text}")
        return "continue"
    if not cand.get("exact_acceptable"):
        print(f"{sid} is not exactly acceptable \u2014 use [e] to edit it")
        return "continue"
    try:
        cu.accept_candidate(root, case_id, sid)
    except cu.CurationError as exc:
        return _service_error(exc)
    print(f"accepted {sid} as a historical finding")
    return "rerender"


# Dispatch table: one stable binding per action char. Kept as a module-level dict
# (rather than an 11-branch if/elif ladder) so it cannot drift from the _ACTIONS
# frozenset and adding a binding is a single-key change.
_ACTION_DISPATCH: dict[str, Callable[[Path, str, dict[str, Any], Callable[[str], str], list[str]], str]] = {
    "a": _action_accept,
    "n": _action_new,
    "e": _action_edit,
    "x": _action_exclude,
    "c": _action_clean,
    "r": _action_ready,
    "z": _action_exclude_case,
    "i": _action_reinclude,
}


def _run_action(
    action: str, root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str], binding: list[str]
) -> str:
    """Dispatch one recognized action; returns the next action-loop outcome."""
    handler = _ACTION_DISPATCH.get(action)
    if handler is not None:
        return handler(root, case_id, view, read_line, binding)
    if action == "q":
        print("saving; run curate again to resume")
        return "quit"
    if action == "d":
        print(f"case {case_id} deferred \u2014 nothing persisted")
        return "done"
    print(f"action [{action}]: not yet wired")
    return "continue"



def _launch_pager(text: str) -> None:
    """Display *text* in the platform pager (default ``less -R``)."""
    try:
        subprocess.run(["less", "-R"], input=text, text=True, check=False)
    except (subprocess.SubprocessError, OSError):
        print(text)


def _run_case(root: Path, case_id: str, read_line: Callable[[str], str]) -> str:
    """Run one case session; returns ``"quit"`` or ``"done"``.

    Renders once and captures the view binding the render numbered; every
    number-based action resolves through that binding. When the case changed
    after render (stale binding), the loop prints a rerender prompt and
    re-renders instead of re-interpreting the number against fresh content.
    """
    view = cu.get_case(root, case_id)
    print(render_case(view))
    binding = _view_binding(view)
    while True:
        try:
            action = _prompt(read_line, _ACTION_PROMPT).strip().lower()
            if action not in _ACTIONS:
                if action.isdigit():
                    if not _check_fresh(root, case_id, binding):
                        view = cu.get_case(root, case_id)
                        print(render_case(view))
                        binding = _view_binding(view)
                        continue
                    sid = _resolve_number(int(action), binding)
                    if sid is None:
                        print(f"no evidence number {action}")
                        continue
                    view = cu.get_case(root, case_id)
                    rec = next(
                        (r for r in (view.get("evidence") or [])
                         if r.get("source_id") == sid),
                        None,
                    )
                    if rec is None:
                        print(f"no evidence number {action}")
                    else:
                        _launch_pager(rec.get("body") or "")
                    continue
                print(f"unknown action: {action}")
                continue
            outcome = _run_action(action, root, case_id, cu.get_case(root, case_id), read_line, binding)
            if outcome in ("quit", "done"):
                return outcome
            if outcome == "rerender":
                view = cu.get_case(root, case_id)
                print(render_case(view))
                binding = _view_binding(view)
        except KeyboardInterrupt:
            print("interrupted \u2014 prior actions preserved")








def run_curate_tui(
    root: Path,
    case_id: str | None = None,
    *,
    read_line: Callable[[str], str] | None = None,
) -> int:
    """Drive the resumable curation terminal client.

    Queue mode (``case_id is None``) lists cases, renders the index, and prompts
    for a case (id, or 1-based row number) or ``q`` to quit. Single-case mode
    opens *case_id* once and returns its outcome.
    """
    read = read_line or input
    try:
        if case_id is not None:
            _run_case(root, case_id, read)
            return 0
        while True:
            try:
                cases = cu.list_cases(root)
                print(render_index_table(cases))
                text = _prompt(read, "case (id or number), or q: ")
                if text in ("a", "A"):
                    continue
                stripped = text.strip()
                if stripped in ("q", "Q"):
                    return 0
                selected: str = stripped
                if stripped.isdigit():
                    index = int(stripped) - 1
                    if not (0 <= index < len(cases)):
                        print(f"no case at row {stripped}; try again")
                        continue
                    selected = cases[index]["case_id"]
                elif stripped not in {case["case_id"] for case in cases}:
                    print(f"unknown case {stripped}; try again")
                    continue
                if _run_case(root, selected, read) == "quit":
                    return 0
            except KeyboardInterrupt:
                print("interrupted \u2014 prior actions preserved")
    except (
        cu.CurationError,
        WorkspaceCorrupt,
        ValidationError,
        KeyError,
        TypeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1

