"""Pure mermaid renderers for the grounded-diagram phase (issue #1113).

The LLM never writes mermaid. It proposes a structured spec whose every element
carries ``file:line`` evidence; a deterministic grounding pass prunes and caps
that spec; this module turns the surviving ``spec_final`` into mermaid text and
into the folded ``<details>`` blocks that land in the Code Review Summary and in
``review-output.md``.

Everything here is pure and byte-deterministic: same spec in, same bytes out. No
I/O, no LLM, no clock, no set iteration. Model-authored strings only ever reach
the output through :func:`sanitize_label` (mermaid labels) or ``_md_text``
(omission text), so a label can never introduce a new mermaid statement or
close the surrounding ``<details>`` wrapper.

The render caps are enforced upstream, in the grounding pass, *before* the
omission floor is evaluated -- that is what keeps the floor honest. The
renderers assert them again and raise ``ValueError`` on an over-cap spec:
defense in depth against a hand-written or corrupted artifact.

Exports:
    render_sequence_mermaid: spec_final -> mermaid ``sequenceDiagram`` text
    render_flowchart_mermaid: spec_final -> mermaid ``flowchart TD`` text
    render_diagram_blocks: per-kind results -> folded markdown blocks
    render_omission_notice: kind + result -> one-paragraph omission text
    sanitize_label: model text + cap -> mermaid-safe label
"""

from __future__ import annotations

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
from daydream.deep.diagram_types import BLOCK_KINDS, as_dict, as_list, as_optional_str

_mapping, _list, _key = as_dict, as_list, as_optional_str

# Sanitization

# Characters dropped outright from a mermaid label. Each one is either a mermaid
# statement terminator (``;``), a comment opener (``%%``, handled separately), a
# markdown code-span opener (backtick), an entity opener (``#``), a table-cell
# separator (``|``), a node-shape delimiter (brackets/braces/parens), or the
# mermaid escape/shape character ``\`` (``[\text\]``). Dropping ``#`` *before*
# the escapes below is what makes ``#lt;`` unforgeable from model text.
_DROPPED_LABEL_CHARS = ";`#|[]{}()\\"

# Applied after the drop pass, so the ``#`` they introduce is always ours.
_LABEL_ESCAPES: tuple[tuple[str, str], ...] = (("<", "#lt;"), (">", "#gt;"), ('"', "#quot;"))

# Omission-text sanitizer: strip markdown delimiters and HTML brackets,
# and fold every whitespace run to one space.
_MD_DROPPED_CHARS = "`|<>"

# Bound omission reasons from malformed artifacts.
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


# Small typed readers over the untyped spec/result dicts


def _dicts(value: Any) -> list[dict[str, Any]]:
    """Return the dict members of ``value`` when it is a list, else ``[]``."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# Sequence renderer

_REPLY_ARROW = "-->>"
_CALL_ARROW = "->>"


def _message_line(message: dict[str, Any], ids: dict[str, str], indent: str) -> str | None:
    """Render one message arrow, or ``None`` when an endpoint is not a participant.

    Grounding drops participants with no remaining messages and messages whose
    endpoints did not survive, so an unresolvable endpoint can only come from a
    hand-written or corrupted spec. Skipping the line is the fail-open answer.
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
        keywords[block_index] = kind if kind in BLOCK_KINDS else "opt"
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


# Flowchart renderer



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


# Markdown blocks

_KIND_TITLES = {"sequence": "Sequence Diagram", "flowchart": "Flowchart"}
_KIND_PHRASES = {"sequence": "sequence diagram", "flowchart": "flowchart"}


def _wrap_block(title: str, mermaid: str) -> str:
    """Wrap a diagram in one collapsible block."""
    return "\n".join(
        (
            f"<details><summary><h3>{title}</h3></summary>",
            "",
            "```mermaid",
            mermaid,
            "```",
            "",
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
    "rendered"``, or lacks a usable ``spec_final`` -- a combination
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
        if not isinstance(spec, dict):
            continue
        if kind == "sequence":
            block = _wrap_block(
                _KIND_TITLES[kind],
                render_sequence_mermaid(spec),
            )
        else:
            block = _wrap_block(
                _KIND_TITLES[kind],
                render_flowchart_mermaid(spec),
            )
        blocks.append(block)
    return "\n\n".join(blocks)


def render_omission_notice(kind: str, result: dict[str, Any]) -> str:
    """Render the one-paragraph "no diagram" text for an explicitly-requested kind.

    Used only on an explicit request (``--diagram-only`` / a mention command),
    where silence would be indistinguishable from a broken run. States the kind,
    the floor reason codes or the skip/failure reason.

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
    return " ".join(parts)
