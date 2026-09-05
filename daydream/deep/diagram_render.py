"""Pure mermaid renderers for the grounded-diagram phase (issue #1113).

The LLM never writes mermaid. It proposes a structured spec whose every element
carries ``file:line`` evidence; a deterministic grounding pass prunes and caps
that spec; this module turns the surviving ``spec_final`` into mermaid text and
into the folded ``<details>`` blocks that land in the Code Review Summary and in
``review-output.md``.

Everything here is pure and byte-deterministic: same spec in, same bytes out. No
I/O, no LLM, no clock, no set iteration. Model-authored strings only ever reach
the output through :func:`sanitize_label` (mermaid labels) or ``_md_text``
(markdown cells), so a label can never introduce a new mermaid statement, break
out of a markdown table cell, or close the surrounding ``<details>`` wrapper.

The render caps are enforced upstream, in the grounding pass, *before* the
omission floor is evaluated -- that is what keeps the floor honest. The
renderers assert them again and raise ``ValueError`` on an over-cap spec:
defense in depth against a hand-written or corrupted artifact.

``SEQUENCE_LINE_GRAMMAR`` / ``FLOWCHART_LINE_GRAMMAR`` are the exhaustive line
grammars for each kind -- every line either renderer can emit ``fullmatch``es
its kind's grammar, and nothing else does. They are exported so the integration
tests can re-assert that property against real pipeline output.

Exports:
    render_sequence_mermaid: spec_final -> mermaid ``sequenceDiagram`` text
    render_flowchart_mermaid: spec_final -> mermaid ``flowchart TD`` text
    render_diagram_blocks: per-kind results -> folded markdown blocks
    render_omission_notice: kind + result -> one-paragraph omission text
    sanitize_label: model text + cap -> mermaid-safe label
    SEQUENCE_LINE_GRAMMAR / FLOWCHART_LINE_GRAMMAR: per-kind line grammars
"""

from __future__ import annotations

import re
from typing import Any

from daydream.config import (
    DIAGRAM_KINDS,
    DIAGRAM_LABEL_CAP_EDGE,
    DIAGRAM_LABEL_CAP_MESSAGE,
    DIAGRAM_LABEL_CAP_NODE,
    DIAGRAM_LABEL_CAP_PARTICIPANT,
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_EDGES,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_NODES,
    DIAGRAM_MAX_PARTICIPANTS,
)

# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

# Characters dropped outright from a mermaid label. Each one is either a mermaid
# statement terminator (``;``), a comment opener (``%%``, handled separately), a
# markdown code-span opener (backtick), an entity opener (``#``), a table-cell
# separator (``|``), a node-shape delimiter (brackets/braces/parens), or the
# mermaid escape/shape character ``\`` (``[\text\]``). Dropping ``#`` *before*
# the escapes below is what makes ``#lt;`` unforgeable from model text.
_DROPPED_LABEL_CHARS = ";`#|[]{}()\\"

# Applied after the drop pass, so the ``#`` they introduce is always ours.
_LABEL_ESCAPES: tuple[tuple[str, str], ...] = (("<", "#lt;"), (">", "#gt;"), ('"', "#quot;"))

# Markdown-cell sanitizer: strip the three characters that could escape a table
# cell or an HTML wrapper, and fold every whitespace run to one space.
_MD_DROPPED_CHARS = "`|<>"

# Defensive cap for markdown cells (paths, symbols, reason codes). Long enough
# never to bite a real repository path, short enough that a corrupted artifact
# cannot emit a megabyte-wide table row.
_MD_CAP = 200

# Label substituted when sanitization leaves nothing. An empty mermaid label
# would render as a nameless box and, for ``alt``/``opt``/``loop``, as a bare
# keyword line that the line grammar would still accept but a reader could not.
_EMPTY_LABEL = "unlabeled"


def sanitize_label(text: str, cap: int) -> str:
    """Reduce model-authored text to a mermaid-safe, length-capped label.

    Pipeline, in this exact order (the order is load-bearing):

    1. drop non-printable characters (control bytes, ANSI escapes),
    2. drop every character in ``_DROPPED_LABEL_CHARS`` -- including ``#``,
    3. remove ``%%`` sequences repeatedly until none remain (a single ``%`` is
       harmless and is preserved),
    4. collapse every whitespace run (newlines included) to a single space and
       strip the ends,
    5. truncate to ``cap`` characters and right-strip,
    6. escape ``<``/``>``/``"`` as ``#lt;``/``#gt;``/``#quot;``.

    The ``%%`` pass runs after the drop pass (dropping a character between two
    percent signs would otherwise fuse them) and before the collapse pass
    (which shrinks whitespace runs and so can never fuse them). Truncation
    happens *before* escaping, so a cut can never bisect an escape and leave a
    stray ``#``; the returned string may therefore be slightly longer than
    ``cap`` when it contains escapes.

    Args:
        text: The model-authored label. A non-string is treated as empty.
        cap: Maximum number of pre-escape characters to keep. ``<= 0`` disables
            truncation.

    Returns:
        The sanitized label. May be empty; callers substitute ``_EMPTY_LABEL``.
    """
    raw = text if isinstance(text, str) else ""
    raw = "".join(ch for ch in raw if ch.isprintable() or ch.isspace())
    for char in _DROPPED_LABEL_CHARS:
        raw = raw.replace(char, "")
    while "%%" in raw:
        raw = raw.replace("%%", "")
    raw = " ".join(raw.split())
    if cap > 0:
        raw = raw[:cap].rstrip()
    for src, dst in _LABEL_ESCAPES:
        raw = raw.replace(src, dst)
    return raw


def _label(text: Any, cap: int) -> str:
    """Sanitize ``text`` for mermaid, substituting a placeholder when empty."""
    return sanitize_label(text if isinstance(text, str) else "", cap) or _EMPTY_LABEL


def _md_text(value: Any, cap: int = _MD_CAP) -> str:
    """Reduce ``value`` to safe inline markdown-cell text (may be empty)."""
    raw = value if isinstance(value, str) else ""
    raw = "".join(ch for ch in raw if ch.isprintable() or ch.isspace())
    for char in _MD_DROPPED_CHARS:
        raw = raw.replace(char, "")
    return " ".join(raw.split())[:cap]


def _code_span(value: Any) -> str:
    """Render ``value`` as a markdown code span, or ``""`` when it is empty."""
    text = _md_text(value)
    return f"`{text}`" if text else ""


# ---------------------------------------------------------------------------
# Line grammars
# ---------------------------------------------------------------------------

# One label character: anything the sanitizer cannot remove, plus our three
# escapes. ``>`` is excluded, so no label can contain ``->>`` or ``-->>``; ``|``
# is excluded, so an edge label cannot close its own delimiter; the shape
# delimiters are excluded, so a node label cannot close its own shape.
_LABEL_CHAR = r"[^\n;`#<>|\[\]{}()]"
_LABEL_ATOM = rf"(?:#lt;|#gt;|#quot;|{_LABEL_CHAR})"
_LABEL_RE = rf"{_LABEL_ATOM}*"

#: Every line :func:`render_sequence_mermaid` can emit ``fullmatch``es this.
SEQUENCE_LINE_GRAMMAR: re.Pattern[str] = re.compile(
    "|".join(
        (
            r"sequenceDiagram",
            rf"    participant P\d+ as {_LABEL_RE}",
            rf"(?:    |        )P\d+(?:->>|-->>)P\d+: {_LABEL_RE}",
            rf"    (?:alt|else|opt|loop) {_LABEL_RE}",
            r"    end",
        )
    )
)

_SHAPE_RE = (
    rf"(?:\(\[{_LABEL_RE}\]\)|\[\[{_LABEL_RE}\]\]|\[/{_LABEL_RE}/\]|\[{_LABEL_RE}\]|\{{{_LABEL_RE}\}})"
)
_NODE_REF_RE = rf"N\d+{_SHAPE_RE}?"

#: Every line :func:`render_flowchart_mermaid` can emit ``fullmatch``es this.
FLOWCHART_LINE_GRAMMAR: re.Pattern[str] = re.compile(
    "|".join(
        (
            r"flowchart TD",
            rf"    N\d+{_SHAPE_RE}",
            rf"    {_NODE_REF_RE} --> {_NODE_REF_RE}",
            rf"    {_NODE_REF_RE} -->\|{_LABEL_RE}\| {_NODE_REF_RE}",
        )
    )
)


# ---------------------------------------------------------------------------
# Small typed readers over the untyped spec/result dicts
# ---------------------------------------------------------------------------


def _dicts(value: Any) -> list[dict[str, Any]]:
    """Return the dict members of ``value`` when it is a list, else ``[]``."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    """Return ``value`` as an int when it is a non-bool integer, else ``0``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _key(value: Any) -> str | None:
    """Return ``value`` when it is a usable dict key (a string), else ``None``."""
    return value if isinstance(value, str) and value else None


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


def _was(count: int) -> str:
    return "was" if count == 1 else "were"


def _evidence_location(evidence: Any) -> str:
    """Render an evidence block as ``path:line``, or ``""`` when unusable."""
    block = _mapping(evidence)
    path = _md_text(block.get("file"))
    line = block.get("line")
    if not path or isinstance(line, bool) or not isinstance(line, int):
        return path
    return f"{path}:{line}"


# ---------------------------------------------------------------------------
# Sequence renderer
# ---------------------------------------------------------------------------

_REPLY_ARROW = "-->>"
_CALL_ARROW = "->>"
_BLOCK_KINDS = ("alt", "opt", "loop")


def _message_line(message: dict[str, Any], ids: dict[str, str], indent: str) -> str | None:
    """Render one message arrow, or ``None`` when an endpoint is not a participant.

    Grounding drops participants with no remaining messages and messages whose
    endpoints did not survive, so an unresolvable endpoint can only come from a
    hand-written or corrupted spec. Skipping the line is the fail-open answer --
    the evidence table still lists the message.
    """
    src = ids.get(_key(message.get("from")) or "")
    dst = ids.get(_key(message.get("to")) or "")
    if src is None or dst is None:
        return None
    arrow = _REPLY_ARROW if message.get("kind") == "reply" else _CALL_ARROW
    label = _label(message.get("label"), DIAGRAM_LABEL_CAP_MESSAGE)
    return f"{indent}{src}{arrow}{dst}: {label}"


def _block_ownership(
    blocks: list[dict[str, Any]], message_count: int
) -> tuple[dict[int, tuple[int, int]], dict[int, str], dict[tuple[int, int], str]]:
    """Map each message index to the (block, branch) that owns it.

    First claim wins, so two blocks citing the same message cannot both wrap it.
    Out-of-range and non-integer indices are ignored. Returns the ownership map,
    the per-block mermaid keyword, and the per-branch sanitized condition.
    """
    owner: dict[int, tuple[int, int]] = {}
    keywords: dict[int, str] = {}
    conditions: dict[tuple[int, int], str] = {}
    for block_index, block in enumerate(blocks):
        kind = block.get("kind")
        keywords[block_index] = kind if kind in _BLOCK_KINDS else "opt"
        for branch_index, branch in enumerate(_dicts(block.get("branches"))):
            conditions[(block_index, branch_index)] = _label(
                branch.get("condition"), DIAGRAM_LABEL_CAP_MESSAGE
            )
            indices = branch.get("messages")
            if not isinstance(indices, list):
                continue
            for raw in indices:
                if isinstance(raw, bool) or not isinstance(raw, int):
                    continue
                if 0 <= raw < message_count and raw not in owner:
                    owner[raw] = (block_index, branch_index)
    return owner, keywords, conditions


def render_sequence_mermaid(spec_final: dict[str, Any]) -> str:
    """Render a grounded sequence spec as mermaid ``sequenceDiagram`` text.

    Participants are declared in spec order as ``P1..Pn``; ``call`` and ``self``
    messages use ``->>``, ``reply`` uses ``-->>``. Blocks are emitted by walking
    the messages in order and opening/closing the owning block as ownership
    changes, so no message is ever dropped or reordered by a malformed block and
    an ``alt`` whose branches are interleaved simply closes and reopens.

    Args:
        spec_final: The pruned + capped sequence spec.

    Returns:
        The mermaid text, without a trailing newline.

    Raises:
        ValueError: A collection exceeds its render cap. The grounding pass
            enforces the caps before the omission floor; reaching this means the
            spec did not come from that pass.
    """
    participants = _dicts(spec_final.get("participants"))
    messages = _dicts(spec_final.get("messages"))
    blocks = _dicts(spec_final.get("blocks"))
    _assert_cap("participants", len(participants), DIAGRAM_MAX_PARTICIPANTS)
    _assert_cap("messages", len(messages), DIAGRAM_MAX_MESSAGES)
    _assert_cap("blocks", len(blocks), DIAGRAM_MAX_BLOCKS)

    lines = ["sequenceDiagram"]
    ids: dict[str, str] = {}
    for index, participant in enumerate(participants):
        pid = f"P{index + 1}"
        name = _key(participant.get("name"))
        if name is not None and name not in ids:
            ids[name] = pid
        lines.append(f"    participant {pid} as {_label(participant.get('name'), DIAGRAM_LABEL_CAP_PARTICIPANT)}")

    owner, keywords, conditions = _block_ownership(blocks, len(messages))
    open_block: int | None = None
    open_branch: int | None = None
    for index, message in enumerate(messages):
        claim = owner.get(index)
        if claim is None:
            if open_block is not None:
                lines.append("    end")
                open_block = None
                open_branch = None
        else:
            block_index, branch_index = claim
            condition = conditions[(block_index, branch_index)]
            if open_block is None:
                lines.append(f"    {keywords[block_index]} {condition}")
            elif block_index != open_block:
                lines.append("    end")
                lines.append(f"    {keywords[block_index]} {condition}")
            elif branch_index != open_branch:
                # ``else`` is only legal inside ``alt``; for ``opt``/``loop`` a
                # second branch is malformed, so close and reopen instead.
                if keywords[block_index] == "alt":
                    lines.append(f"    else {condition}")
                else:
                    lines.append("    end")
                    lines.append(f"    {keywords[block_index]} {condition}")
            open_block, open_branch = block_index, branch_index
        line = _message_line(message, ids, "        " if open_block is not None else "    ")
        if line is not None:
            lines.append(line)
    if open_block is not None:
        lines.append("    end")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flowchart renderer
# ---------------------------------------------------------------------------

_BARE_NODE_KINDS = ("start", "end", "process", "decision", "io")


def _node_shape(node: dict[str, Any]) -> str:
    """Render a node's mermaid shape + label, e.g. ``([Start])`` or ``{Ready?}``.

    An unknown kind falls back to the ``process`` rectangle. The label cannot
    contain a shape delimiter (the sanitizer drops all of them), so a shape can
    never be closed early from model text.
    """
    label = _label(node.get("label"), DIAGRAM_LABEL_CAP_NODE)
    kind = node.get("kind")
    if kind in ("start", "end"):
        return f"([{label}])"
    if kind == "decision":
        return f"{{{label}}}"
    if kind == "subroutine":
        return f"[[{label}]]"
    if kind == "io":
        # ``/`` is legal inside the label (paths are common) but a leading or
        # trailing one would sit against the shape's own ``/`` delimiter and
        # confuse the mermaid lexer, so trim just the ends.
        return f"[/{label.strip('/') or _EMPTY_LABEL}/]"
    return f"[{label}]"


def render_flowchart_mermaid(spec_final: dict[str, Any]) -> str:
    """Render a grounded flowchart spec as mermaid ``flowchart TD`` text.

    Nodes are numbered ``N1..Nn`` in spec order. A node carries its shape and
    label at its **first mention** in the edge list and is referenced bare
    afterwards; nodes that no edge mentions are declared on their own lines after the
    edges, in spec order. Edges whose endpoints are not declared nodes are
    skipped (grounding drops those; a corrupted spec must not crash the report).

    Args:
        spec_final: The pruned + capped flowchart spec.

    Returns:
        The mermaid text, without a trailing newline.

    Raises:
        ValueError: ``nodes`` or ``edges`` exceeds its render cap.
    """
    nodes = _dicts(spec_final.get("nodes"))
    edges = _dicts(spec_final.get("edges"))
    _assert_cap("nodes", len(nodes), DIAGRAM_MAX_NODES)
    _assert_cap("edges", len(edges), DIAGRAM_MAX_EDGES)

    ids: dict[str, str] = {}
    shapes: dict[str, str] = {}
    order: list[str] = []
    for index, node in enumerate(nodes):
        nid = f"N{index + 1}"
        order.append(nid)
        shapes[nid] = _node_shape(node)
        key = _key(node.get("id"))
        if key is not None and key not in ids:
            ids[key] = nid

    lines = ["flowchart TD"]
    mentioned: set[str] = set()

    def reference(nid: str) -> str:
        if nid in mentioned:
            return nid
        mentioned.add(nid)
        return f"{nid}{shapes[nid]}"

    for edge in edges:
        src = ids.get(_key(edge.get("from")) or "")
        dst = ids.get(_key(edge.get("to")) or "")
        if src is None or dst is None:
            continue
        head = reference(src)
        tail = reference(dst)
        raw_label = edge.get("label")
        # ``label`` is nullable in the spec; an absent or empty-after-sanitizing
        # label degrades to an unlabeled edge, never to ``-->||``.
        label = sanitize_label(raw_label if isinstance(raw_label, str) else "", DIAGRAM_LABEL_CAP_EDGE)
        if label:
            lines.append(f"    {head} -->|{label}| {tail}")
        else:
            lines.append(f"    {head} --> {tail}")

    lines.extend(f"    {nid}{shapes[nid]}" for nid in order if nid not in mentioned)
    return "\n".join(lines)


def _assert_cap(collection: str, size: int, cap: int) -> None:
    """Raise when ``size`` exceeds the render cap for ``collection``."""
    if size > cap:
        raise ValueError(
            f"diagram spec exceeds the {collection} render cap: {size} > {cap}; "
            "caps are enforced by the grounding pass before the omission floor"
        )


# ---------------------------------------------------------------------------
# Markdown blocks
# ---------------------------------------------------------------------------

_KIND_TITLES = {"sequence": "Sequence Diagram", "flowchart": "Flowchart"}
_KIND_PHRASES = {"sequence": "sequence diagram", "flowchart": "flowchart"}


def _capped_total(grounding: dict[str, Any]) -> int:
    """Total number of elements dropped by a render cap across all collections."""
    return sum(_int(value) for value in _mapping(grounding.get("capped")).values())


def _element_checks(grounding: dict[str, Any], element: str) -> list[dict[str, Any]]:
    """Every ``ElementCheck`` dict of the given element type, in report order."""
    return [c for c in _dicts(grounding.get("elements")) if c.get("element") == element]


def _defined_at_by_final_index(grounding: dict[str, Any], element: str) -> dict[int, str]:
    """Map ``final_index`` -> ``defined_at`` for the given element type."""
    out: dict[int, str] = {}
    for check in _element_checks(grounding, element):
        index = check.get("final_index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        defined_at = _md_text(check.get("defined_at"))
        if defined_at:
            out[index] = defined_at
    return out


def _defined_at_by_ref(grounding: dict[str, Any], element: str) -> dict[str, str]:
    """Map ``ref`` -> ``defined_at`` for the given element type."""
    out: dict[str, str] = {}
    for check in _element_checks(grounding, element):
        ref = _key(check.get("ref"))
        defined_at = _md_text(check.get("defined_at"))
        if ref is not None and defined_at and ref not in out:
            out[ref] = defined_at
    return out


def _sequence_sub_line(spec: dict[str, Any], grounding: dict[str, Any]) -> str:
    """The ``<sub>`` grounding line for a rendered sequence diagram."""
    messages = len(_dicts(spec.get("messages")))
    participants = len(_dicts(spec.get("participants")))
    parts = [
        f"{messages} {_plural(messages, 'interaction')} across "
        f"{participants} {_plural(participants, 'component')}, "
        "each grounded to a cited call site."
    ]
    pruned = _int(_mapping(grounding.get("summary")).get("pruned"))
    if pruned:
        parts.append(
            f"{pruned} proposed {_plural(pruned, 'interaction')} {_was(pruned)} dropped as ungrounded."
        )
    capped = _capped_total(grounding)
    if capped:
        parts.append(
            f"{capped} further {_plural(capped, 'interaction')} {_was(capped)} "
            "trimmed to fit the diagram cap."
        )
    return " ".join(parts)


def _flowchart_sub_line(spec: dict[str, Any], grounding: dict[str, Any]) -> str:
    """The ``<sub>`` grounding line for a rendered flowchart."""
    root = _mapping(spec.get("root"))
    name = _md_text(root.get("name")) or "the changed function"
    path = _md_text(root.get("file"))
    line = root.get("line")
    start = "" if isinstance(line, bool) or not isinstance(line, int) else str(line)
    location = f"{path}:{start}" if path and start else path
    root_range = grounding.get("root_range")
    if location and start and isinstance(root_range, (list, tuple)) and len(root_range) == 2:
        end = root_range[1]
        if not isinstance(end, bool) and isinstance(end, int):
            location = f"{path}:{start}-{end}"
    nodes = len(_dicts(spec.get("nodes")))
    where = f" (`{location}`)" if location else ""
    parts = [
        f"Control flow of `{name}`{where}: {nodes} {_plural(nodes, 'node')}, "
        "each grounded to a statement inside that function."
    ]
    pruned = _int(_mapping(grounding.get("summary")).get("pruned"))
    if pruned:
        parts.append(f"{pruned} proposed {_plural(pruned, 'node')} {_was(pruned)} dropped as ungrounded.")
    capped = _capped_total(grounding)
    if capped:
        parts.append(
            f"{capped} further {_plural(capped, 'node')} {_was(capped)} trimmed to fit the diagram cap."
        )
    return " ".join(parts)


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table. Cells are already sanitized by their builders."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _sequence_evidence_table(spec: dict[str, Any], grounding: dict[str, Any]) -> str:
    """One row per message: interaction text, call site, callee definition site."""
    participants = _dicts(spec.get("participants"))
    labels = {
        name: _label(participant.get("name"), DIAGRAM_LABEL_CAP_PARTICIPANT)
        for participant in participants
        if (name := _key(participant.get("name"))) is not None
    }
    defined = _defined_at_by_final_index(grounding, "message")
    rows: list[list[str]] = []
    for index, message in enumerate(_dicts(spec.get("messages"))):
        source = _key(message.get("from")) or ""
        target = _key(message.get("to")) or ""
        src = labels.get(source) or _label(source, DIAGRAM_LABEL_CAP_PARTICIPANT)
        dst = labels.get(target) or _label(target, DIAGRAM_LABEL_CAP_PARTICIPANT)
        label = _label(message.get("label"), DIAGRAM_LABEL_CAP_MESSAGE)
        rows.append(
            [
                str(index + 1),
                f"{src} → {dst}: {label}",
                _code_span(_evidence_location(message.get("evidence"))),
                _code_span(defined.get(index, "")),
            ]
        )
    return _table(["#", "Interaction", "Call site", "Callee defined at"], rows)


def _flowchart_evidence_table(spec: dict[str, Any], grounding: dict[str, Any]) -> str:
    """One row per node: generated id, statement description, source location."""
    defined = _defined_at_by_ref(grounding, "node")
    rows: list[list[str]] = []
    for index, node in enumerate(_dicts(spec.get("nodes"))):
        nid = f"N{index + 1}"
        location = _evidence_location(node.get("evidence"))
        if node.get("kind") == "subroutine":
            symbol = _code_span(_mapping(node.get("evidence")).get("symbol"))
            statement = f"subroutine {symbol}" if symbol else "subroutine"
            if location:
                statement = f"{statement}, called at {_code_span(location)}"
            definition = defined.get(_key(node.get("id")) or "", "")
            if definition:
                statement = f"{statement}, defined at {_code_span(definition)}"
            rows.append([nid, statement, ""])
            continue
        kind = node.get("kind")
        statement = kind if kind in _BARE_NODE_KINDS else "process"
        rows.append([nid, statement, _code_span(location)])
    return _table(["Node", "Statement", "Location"], rows)


def _wrap_block(title: str, mermaid: str, sub_line: str, table: str) -> str:
    """Wrap one kind's mermaid + grounding line + evidence table in the folds."""
    return "\n".join(
        (
            f"<details><summary><h3>{title}</h3></summary>",
            "",
            "```mermaid",
            mermaid,
            "```",
            "",
            f"<sub>{sub_line}</sub>",
            "",
            "<details><summary>Evidence</summary>",
            "",
            table,
            "</details>",
            "</details>",
        )
    )


def render_diagram_blocks(results: dict[str, dict[str, Any] | None]) -> str:
    """Render the folded ``<details>`` block for every kind that rendered.

    The mermaid is **always re-rendered from** ``spec_final``; a stored
    ``mermaid`` string is never read, so no model-authored markdown can reach a
    PR comment even if the artifact carries some. Kinds are emitted in
    ``config.DIAGRAM_KINDS`` order (sequence first) and joined with one blank
    line, with no leading or trailing blank line.

    A kind is skipped when its result is missing, is not ``status ==
    "rendered"``, or lacks a usable ``spec_final``/``grounding`` -- a combination
    only a malformed artifact can produce, and never a reason to raise.

    Args:
        results: The per-kind result dicts, keyed by diagram kind.

    Returns:
        The joined markdown blocks, or ``""`` when nothing rendered.
    """
    blocks: list[str] = []
    for kind in DIAGRAM_KINDS:
        result = results.get(kind)
        if not isinstance(result, dict) or result.get("status") != "rendered":
            continue
        spec = result.get("spec_final")
        grounding = result.get("grounding")
        if not isinstance(spec, dict) or not isinstance(grounding, dict):
            continue
        if kind == "sequence":
            block = _wrap_block(
                _KIND_TITLES[kind],
                render_sequence_mermaid(spec),
                _sequence_sub_line(spec, grounding),
                _sequence_evidence_table(spec, grounding),
            )
        else:
            block = _wrap_block(
                _KIND_TITLES[kind],
                render_flowchart_mermaid(spec),
                _flowchart_sub_line(spec, grounding),
                _flowchart_evidence_table(spec, grounding),
            )
        blocks.append(block)
    return "\n\n".join(blocks)


def render_omission_notice(kind: str, result: dict[str, Any]) -> str:
    """Render the one-paragraph "no diagram" text for an explicitly-requested kind.

    Used only on an explicit request (``--diagram-only`` / a mention command),
    where silence would be indistinguishable from a broken run. States the kind,
    the floor reason codes or the skip/failure reason, and the grounding counts.

    Args:
        kind: ``"sequence"`` or ``"flowchart"``.
        result: That kind's result dict.

    Returns:
        A single-line paragraph, or ``""`` when the kind did render.
    """
    if result.get("status") == "rendered":
        return ""
    phrase = _KIND_PHRASES.get(kind, _md_text(kind) or "diagram")
    parts = [f"No {phrase} was rendered for this pull request."]
    codes = [text for raw in _list(result.get("omit_reasons")) if (text := _md_text(raw))]
    if codes:
        parts.append("Grounding floor not met: " + ", ".join(codes) + ".")
    reason = _md_text(result.get("reason"))
    if reason:
        parts.append(f"Reason: {reason}.")
    grounding = result.get("grounding")
    if isinstance(grounding, dict):
        summary = _mapping(grounding.get("summary"))
        proposed = _int(summary.get("proposed"))
        parts.append(
            f"{proposed} {_plural(proposed, 'element')} proposed, "
            f"{_int(summary.get('grounded_first_pass'))} grounded on the first pass, "
            f"{_int(summary.get('repaired'))} repaired, "
            f"{_int(summary.get('pruned'))} dropped as ungrounded."
        )
        capped = _capped_total(grounding)
        if capped:
            parts.append(
                f"{capped} {_plural(capped, 'element')} {_was(capped)} trimmed to fit the diagram cap."
            )
    return " ".join(parts)


def _list(value: Any) -> list[Any]:
    """Return ``value`` when it is a list, else ``[]``."""
    return value if isinstance(value, list) else []
