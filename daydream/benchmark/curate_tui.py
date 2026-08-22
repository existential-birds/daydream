"""Resumable keyboard-driven terminal curation client.

Pure-client UI over :mod:`daydream.benchmark.curation`: every mutating action
``[a/e/n/x/c/r/d/z/i/q]`` maps one-to-one onto a service operation and never
mutates the case YAML/model directly. Rendering is plain-string builders for
deterministic tests; Rich stays available for live styling.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import ValidationError

from daydream.benchmark import curation as cu
from daydream.benchmark.storage import WorkspaceCorrupt


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
    return read_line(message)


def render_case(case: dict[str, Any]) -> str:
    """Render a plain-text snapshot header + numbered evidence list.

    Each evidence entry shows its number, kind, author login (+ ``[bot]`` for a
    bot), commit prefix, ``path:line`` anchor, resolved/outdated markers, the
    candidate title, and a body preview (first ~120 chars). Candidates without
    an ``evidence`` projection render those fields as ``-``.
    """
    snapshot = case.get("snapshot") or {}
    curation = case.get("curation") or {}
    head = snapshot.get("original_head_sha") or "-"
    policy = snapshot.get("policy") or "-"
    state = curation.get("state") or "-"
    lines = [
        f"case {case.get('case_id')}: state={state} policy={policy} head={head}",
    ]
    for i, cand in enumerate(case.get("candidates") or [], start=1):
        ev = cand.get("evidence") or {}
        author = ev.get("author") or {}
        login = author.get("login") or "-"
        if author.get("type") == "Bot":
            login = f"{login}[bot]"
        kind = ev.get("kind") or "-"
        commit = (ev.get("commit_id") or "")[:12] or "-"
        location = cand.get("location") or {}
        loc_path = location.get("path") or "-"
        start = location.get("start_line")
        anchor = f"{loc_path}:{start}" if loc_path != "-" and start else loc_path
        markers = ""
        if ev.get("resolved"):
            markers += " [resolved]"
        if ev.get("outdated"):
            markers += " [outdated]"
        preview = (cand.get("body") or "").replace("\n", " ")[:120]
        line_parts = [
            f"  {i}. [{kind}] {login} {commit} {anchor}{markers}",
            f"      {cand.get('title') or '-'}",
            f"      {preview or '-'}",
        ]
        lines.append("\n".join(line_parts))
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
            proc = subprocess.run([editor, path], text=False)
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


def _action_new(root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]) -> str:
    """The ``[n]`` author action: edit a blank fragment, add every atom."""
    text = _launch_editor(_editor_fragment_new())
    if text is None:
        print("editor cancelled or failed; nothing written")
        return "continue"
    atoms = _parse_fragment(text)
    if atoms is None:
        print("invalid fragment; nothing written")
        return "continue"
    try:
        for atom in atoms:
            cu.add_finding(
                root, case_id,
                title=atom["title"], body=atom["body"],
                severity=atom.get("severity"),
                location=atom.get("location"),
                source_ids=atom.get("source_ids") or [],
            )
    except (cu.CurationError, ValidationError) as exc:
        print(str(exc))
        return "continue"
    print(f"added {len(atoms)} authored finding(s)")
    return "rerender"


def _action_edit(root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]) -> str:
    """The ``[e]`` edit action: replace one gold finding with its re-written atoms."""
    findings = (view.get("curation") or {}).get("findings") or []
    text = _prompt(read_line, "finding (number, 0 to cancel): ").strip()
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(findings))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    finding = findings[indices[0]]
    frag = _launch_editor(_editor_fragment_edit(finding))
    if frag is None:
        print("editor cancelled or failed; nothing written")
        return "continue"
    atoms = _parse_fragment(frag)
    if atoms is None:
        print("invalid fragment; nothing written")
        return "continue"
    try:
        cu.replace_findings(root, case_id, finding["finding_id"], replacements=atoms)
    except (cu.CurationError, ValidationError) as exc:
        print(str(exc))
        return "continue"
    print(f"replaced finding with {len(atoms)} atom(s)")
    return "rerender"


# Single source of truth lives on the curation service; re-export here so a
# reason added on one side cannot drift apart from the other.
_EVIDENCE_REASONS = cu._EVIDENCE_REASONS
_CASE_EXCLUSION_REASONS = cu._CASE_EXCLUSION_REASONS


def _action_exclude_case(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]
) -> str:
    """The ``[z]`` case-exclusion action (4 fixed reasons + optional note)."""
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
        print(str(exc))
        return "continue"
    print(f"excluded case {case_id}")
    return "rerender"


def _action_reinclude(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]
) -> str:
    """The ``[i]`` re-include action for an excluded case."""
    try:
        cu.reinclude_case(root, case_id)
    except cu.CurationError as exc:
        print(str(exc))
        return "continue"
    print(f"re-included case {case_id}")
    return "rerender"


def _action_clean(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]
) -> str:
    """The ``[c]`` clean-attest action (requires an empty gold findings set)."""
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
        print(str(exc))
        return "continue"
    print(f"attested {case_id} clean")
    return "rerender"


def _action_ready(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]
) -> str:
    """The ``[r]`` mark-ready action (exact-head-SHA confirmation)."""
    head = (view.get("snapshot") or {}).get("original_head_sha") or ""
    answer = _prompt(
        read_line,
        f"Attest that this golden review is valid against head {head} "
        f"and mark {case_id} ready? [y/N] ",
    ).strip()
    if answer.lower() != "y":
        return "continue"
    try:
        cu.mark_ready(root, case_id, head_sha=head)
    except cu.CurationError as exc:
        print(str(exc))
        return "continue"
    print(f"marked {case_id} ready")
    return "rerender"


def _action_exclude(
    root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]
) -> str:
    """The ``[x]`` evidence-exclusion action (7 fixed reasons + optional note)."""
    candidates = view.get("candidates") or []
    text = _prompt(read_line, "evidence (number, 0 to cancel): ").strip()
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(candidates))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    source_id = candidates[indices[0]]["source_id"]
    reason = _prompt(
        read_line, f"reason ({'|'.join(_EVIDENCE_REASONS)}): "
    ).strip()
    if reason not in _EVIDENCE_REASONS:
        print(f"invalid evidence reason {reason!r}")
        return "continue"
    note = _prompt(read_line, "note: ").strip() or None
    if reason == "other" and not note:
        print("reason 'other' requires a note")
        return "continue"
    try:
        cu.exclude_evidence(root, case_id, source_id, reason=reason, note=note)
    except cu.CurationError as exc:
        print(str(exc))
        return "continue"
    print(f"excluded {source_id}")
    return "rerender"


def _action_accept(root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]) -> str:
    """The ``[a]`` accept-candidate action: one exact-acceptable candidate."""
    candidates = view.get("candidates") or []
    text = _prompt(read_line, "candidate (number, 0 to cancel): ").strip()
    if text == "0":
        return "continue"
    try:
        indices = parse_indices(text, len(candidates))
    except ValueError as exc:
        print(str(exc))
        return "continue"
    cand = candidates[indices[0]]
    src = cand["source_id"]
    if not cand.get("exact_acceptable"):
        print(f"{src} is not exactly acceptable \u2014 use [e] to edit it")
        return "continue"
    try:
        cu.accept_candidate(root, case_id, src)
    except cu.CurationError as exc:
        print(str(exc))
        return "continue"
    print(f"accepted {src} as a historical finding")
    return "rerender"


def _run_action(action: str, root: Path, case_id: str, view: dict[str, Any], read_line: Callable[[str], str]) -> str:
    """Dispatch one recognized action; returns the next action-loop outcome."""
    if action == "a":
        return _action_accept(root, case_id, view, read_line)
    if action == "n":
        return _action_new(root, case_id, view, read_line)
    if action == "e":
        return _action_edit(root, case_id, view, read_line)
    if action == "x":
        return _action_exclude(root, case_id, view, read_line)
    if action == "c":
        return _action_clean(root, case_id, view, read_line)
    if action == "r":
        return _action_ready(root, case_id, view, read_line)
    if action == "z":
        return _action_exclude_case(root, case_id, view, read_line)
    if action == "i":
        return _action_reinclude(root, case_id, view, read_line)
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
    """Run one case session; returns ``"quit"`` or ``"done"``."""
    print(render_case(cu.get_case(root, case_id)))
    while True:
        try:
            action = _prompt(read_line, _ACTION_PROMPT).strip().lower()
            if action not in _ACTIONS:
                if action.isdigit():
                    candidates = cu.get_case(root, case_id).get("candidates") or []
                    index = int(action) - 1
                    if not (0 <= index < len(candidates)):
                        print(f"no evidence number {action}")
                    else:
                        _launch_pager(candidates[index].get("body") or "")
                    continue
                print(f"unknown action: {action}")
                continue
            outcome = _run_action(action, root, case_id, cu.get_case(root, case_id), read_line)
            if outcome in ("quit", "done"):
                return outcome
            if outcome == "rerender":
                print(render_case(cu.get_case(root, case_id)))
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
    except (cu.CurationError, WorkspaceCorrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 1

