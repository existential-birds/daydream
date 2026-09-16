"""Dependency-neutral prompt-size, sanctioned-input, and advisory-selection policy."""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from daydream.artifact_visibility import ArtifactVisibilityError
from daydream.backends import AUDIT_ROOT_ISOLATION

# Upper bound for inlined diff text. Above this bound, prompts retain an
# on-disk diff pointer rather than embedding the diff.
INLINE_DIFF_BUDGET_BYTES = 12_288
SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES = INLINE_DIFF_BUDGET_BYTES
SANCTIONED_EXACT_INPUT_MAX_FILES = 512
SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES = 1_048_576
SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES = 4_194_304

_SANCTIONED_INLINE_HEADER = "Sanctioned phase inputs (captured verbatim):"
_SANCTIONED_INLINE_CLOSE_TAG = "</sanctioned-input>"


class SanctionedInputUnavailable(ArtifactVisibilityError):
    """A declared model input could not be captured without widening access."""


class SanctionedInputTransport(str, Enum):
    """How one immutable sanctioned-input set reaches a backend."""

    EXACT_PATHS = "exact_paths"
    INLINE = "inline"


@dataclass(frozen=True)
class PreparedSanctionedInput:
    """One stable, UTF-8 artifact captured before a model attempt."""

    label: str
    path: Path
    text: str | None
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class PreparedSanctionedInputs:
    """Closed inputs bound to one backend, cwd, and access mode."""

    transport: SanctionedInputTransport
    inputs: tuple[PreparedSanctionedInput, ...]
    backend_identity: object
    cwd: Path
    read_only: bool

    def render(self) -> str:
        """Render deterministic path pointers or captured inline bytes."""
        if not self.inputs:
            return ""
        if self.transport is SanctionedInputTransport.EXACT_PATHS:
            lines = ["Sanctioned phase inputs (read only these exact files):"]
            lines.extend(f"- {item.label}: {item.path}" for item in self.inputs)
            return "\n".join(lines)
        blocks = [_SANCTIONED_INLINE_HEADER]
        for item in self.inputs:
            blocks += [_sanctioned_inline_open_tag(item.label), item.text or "", _SANCTIONED_INLINE_CLOSE_TAG]
        return "\n".join(blocks)

    def render_prompt(self, prompt: str) -> str:
        """Append the inputs, hiding private pathnames from inline transports.

        Builders suppress the pointers they own, but one still names artifacts
        it did not sanction (a sibling under the same private root), so the
        inputs' common parent is scrubbed too.

        Idempotent: a caller may render the prompt it shapes and then let the
        agent layer re-apply this renderer on its way to the backend. An
        already-appended section is returned unchanged rather than duplicated;
        the scrub still runs first, so a path that survived an earlier render
        is still hidden. The appended section is scrubbed too, because captured
        content can itself embed a private pathname, and scrubbing both sides
        keeps the idempotence check stable across re-renders.
        """
        rendered = self.render()
        if self.transport is SanctionedInputTransport.INLINE and self.inputs:
            for item in self.inputs:
                prompt = prompt.replace(str(item.path), f"sanctioned input '{item.label}'")
                rendered = rendered.replace(str(item.path), f"sanctioned input '{item.label}'")
            common_parent = os.path.commonpath([str(item.path.parent) for item in self.inputs])
            if common_parent and common_parent != os.path.sep:
                prompt = prompt.replace(common_parent, "sanctioned artifact storage")
                rendered = rendered.replace(common_parent, "sanctioned artifact storage")
        if not rendered:
            return prompt
        if prompt.endswith(rendered):
            return prompt
        return f"{prompt}\n\n{rendered}"

    def revalidate(self, backend: object, cwd: Path, read_only: bool) -> None:
        """Fail closed if call identity or any captured file changed."""
        if backend is not self.backend_identity:
            raise SanctionedInputUnavailable("sanctioned input backend changed")
        canonical_cwd = _canonical_cwd(cwd)
        if canonical_cwd != self.cwd or read_only is not self.read_only:
            raise SanctionedInputUnavailable("sanctioned input call mode changed")
        if _sanctioned_transport(backend, canonical_cwd, read_only=read_only) is not self.transport:
            raise SanctionedInputUnavailable("sanctioned input transport mode changed before model execution")
        aggregate = 0
        for item in self.inputs:
            if _unchanged_since_capture(item):
                # The capture already streamed and hashed these exact bytes,
                # so the re-read/re-hash is redundant — but the aggregate cap
                # stays enforced exactly as a fresh capture would enforce it.
                max_bytes, _, _ = _transport_allowance(self.transport, aggregate)
                if item.size > max_bytes:
                    raise SanctionedInputUnavailable(
                        f"sanctioned input {item.label!r} exceeds remaining aggregate input budget"
                    )
                aggregate += item.size
                continue
            current = _capture_input(item.label, item.path, self.transport, aggregate)
            aggregate += current.size
            if current != item:
                raise SanctionedInputUnavailable(f"sanctioned input {item.label!r} changed before model execution")


@dataclass(frozen=True)
class AdvisoryCandidate:
    """One declared advisory input that is admitted whole or omitted whole.

    ``size`` is filled by :func:`select_advisory_inputs` from the file it sized;
    callers declare only a label and a path.
    """

    label: str
    path: Path
    size: int = 0


@dataclass(frozen=True)
class OmittedAdvisoryInput:
    """One advisory input that did not fit the active transport's allowance."""

    label: str
    size: int
    reason: str


@dataclass(frozen=True)
class AdvisorySelection:
    """The whole-artifact split one advisory set resolves to on one transport."""

    transport: SanctionedInputTransport
    admitted: tuple[AdvisoryCandidate, ...]
    omitted: tuple[OmittedAdvisoryInput, ...]
    admitted_bytes: int
    allowance_bytes: int

    def selected_paths(self) -> dict[str, Path]:
        """The admitted mapping :func:`prepare_sanctioned_inputs` consumes."""
        return {candidate.label: candidate.path for candidate in self.admitted}

    def to_dict(self) -> dict[str, object]:
        """JSON-safe omission diagnostic, admitted entries in declared order."""
        return {
            "transport": self.transport.value,
            "allowance_bytes": self.allowance_bytes,
            "admitted_bytes": self.admitted_bytes,
            "admitted": [{"label": c.label, "bytes": c.size} for c in self.admitted],
            "omitted": [
                {"label": o.label, "bytes": o.size, "reason": o.reason} for o in self.omitted
            ],
        }


def _canonical_cwd(cwd: Path) -> Path:
    try:
        return cwd.resolve(strict=True)
    except OSError as exc:
        raise SanctionedInputUnavailable("sanctioned input cwd is unavailable") from exc


def _unchanged_since_capture(item: PreparedSanctionedInput) -> bool:
    """Whether *item*'s file still carries the identity the capture attested.

    The ``(dev, ino, size, mtime_ns)`` tuple is the same identity
    :func:`_capture_input` validates internally; matching it re-validates file
    identity without a per-attempt re-read. A missing, unreadable, or replaced
    file returns ``False`` so the caller performs the full capture and
    surfaces its original fail-closed errors.
    """
    try:
        metadata = item.path.lstat()
    except OSError:
        return False
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
    return identity == (item.device, item.inode, item.size, item.mtime_ns)


def _transport_allowance(transport: SanctionedInputTransport, aggregate: int) -> tuple[int, bool, str]:
    """Remaining per-item byte allowance under *transport* after *aggregate* bytes."""
    if transport is SanctionedInputTransport.INLINE:
        return max(SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - aggregate, 0), False, "byte budget"
    remaining = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES - aggregate
    aggregate_limit = remaining < SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES
    return max(min(SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES, remaining), 0), aggregate_limit, "file byte limit"


def _capture_input(
    label: str, path: Path, transport: SanctionedInputTransport, aggregate: int
) -> PreparedSanctionedInput:
    """Capture one no-follow file within the transport's remaining allowance.

    Reading stops one byte past the allowance, so a refusal never hashes more
    than the aggregate ceiling admits. ``fstat`` before and after bounds the
    captured bytes to one immutable revision of one inode.
    """
    max_bytes, aggregate_limit, limit_name = _transport_allowance(transport, aggregate)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    fd = -1
    try:
        lexical = Path(os.path.abspath(path))
        lexical_stat = lexical.lstat()
        if not stat.S_ISREG(lexical_stat.st_mode):
            raise SanctionedInputUnavailable(f"sanctioned input {label!r} must be a regular file")
        fd = os.open(lexical, flags)
        # An identical (dev, ino) is the same inode, hence still a regular file.
        before = os.fstat(fd)
        if (lexical_stat.st_dev, lexical_stat.st_ino) != (before.st_dev, before.st_ino):
            raise SanctionedInputUnavailable(f"sanctioned input {label!r} changed while being opened")
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        chunks: list[bytes] | None = [] if transport is SanctionedInputTransport.INLINE else None
        total = 0
        while chunk := os.read(fd, min(65_536, max_bytes + 1 - total)):
            total += len(chunk)
            if total > max_bytes:
                if aggregate_limit:
                    raise SanctionedInputUnavailable("sanctioned input aggregate byte limit exceeded")
                raise SanctionedInputUnavailable(f"sanctioned input {label!r} exceeds the {limit_name}")
            digest.update(chunk)
            decoder.decode(chunk, final=False)
            if chunks is not None:
                chunks.append(chunk)
        decoder.decode(b"", final=True)
        after = os.fstat(fd)
    except SanctionedInputUnavailable:
        raise
    except UnicodeDecodeError as exc:
        raise SanctionedInputUnavailable(f"sanctioned input {label!r} is not valid UTF-8") from exc
    except OSError as exc:
        raise SanctionedInputUnavailable(f"sanctioned input {label!r} is unavailable") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    # One fd cannot change device or inode, so size and mtime are the whole check.
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SanctionedInputUnavailable(f"sanctioned input {label!r} changed while being captured")
    payload = b"".join(chunks) if chunks is not None else None
    return PreparedSanctionedInput(
        label=label,
        path=lexical,
        text=payload.decode("utf-8") if payload is not None else None,
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


def _sanctioned_transport(backend: object, canonical_cwd: Path, *, read_only: bool) -> SanctionedInputTransport:
    """Inline the inputs for a backend that cannot read exact host paths.

    "Can this backend read this host path" is not on the ``Backend`` protocol,
    so it is read off the three concrete backends that declare it.
    """
    strict_audit = getattr(backend, "audit_root_isolation", None) == AUDIT_ROOT_ISOLATION
    if strict_audit:
        audit_root = getattr(backend, "audit_root", None)
        try:
            matched = isinstance(audit_root, Path) and audit_root.resolve(strict=True) == canonical_cwd
        except OSError as exc:
            raise SanctionedInputUnavailable("sanctioned input audit root is unavailable") from exc
        if not matched:
            raise SanctionedInputUnavailable("sanctioned input audit root does not match the model cwd")
    if (
        strict_audit
        or (read_only and bool(getattr(backend, "read_only_disposable_clone", False)))
        or bool(getattr(backend, "sandbox", False))
    ):
        return SanctionedInputTransport.INLINE
    return SanctionedInputTransport.EXACT_PATHS


def sanctioned_transport_for(
    backend: object, cwd: Path, *, read_only: bool
) -> SanctionedInputTransport:
    """Return the sanctioned-input transport a capture from *cwd* would use.

    Public wrapper around the transport decision so phase builders can size
    advisory inputs (e.g. exploration context) for the INLINE budget *before*
    capture, mirroring how the diff is excluded when it is inlined.
    """
    return _sanctioned_transport(backend, cwd, read_only=read_only)


def select_advisory_inputs(
    backend: object,
    cwd: Path,
    candidates: Sequence[AdvisoryCandidate],
    *,
    read_only: bool,
) -> AdvisorySelection:
    """Split advisory inputs whole by the transport's real remaining allowance.

    Resolves the transport once with the same resolver and canonical cwd the
    capture path uses, then walks ``candidates`` in the order given, admitting a
    whole candidate while its cost fits and omitting it whole otherwise. INLINE
    costs account for the exact bytes :meth:`PreparedSanctionedInputs.render`
    emits for that candidate, so an admitted set always renders inside the
    shared aggregate; EXACT_PATHS reuses the capture path's per-file/aggregate
    arithmetic. A missing or unreadable candidate is omitted as ``unavailable``;
    transport-resolution failures propagate untouched (fail closed).
    """
    canonical_cwd = _canonical_cwd(cwd)
    transport = _sanctioned_transport(backend, canonical_cwd, read_only=read_only)
    inline = transport is SanctionedInputTransport.INLINE
    allowance_bytes = (
        SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES if inline else SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES
    )
    admitted: list[AdvisoryCandidate] = []
    omitted: list[OmittedAdvisoryInput] = []
    admitted_bytes = 0
    aggregate = 0
    for candidate in candidates:
        try:
            size = candidate.path.lstat().st_size
        except OSError:
            omitted.append(OmittedAdvisoryInput(candidate.label, 0, "unavailable"))
            continue
        if inline:
            entries = [(item.label, item.size) for item in admitted]
            entries.append((candidate.label, size))
            emitted = inline_section_emitted_bytes(entries)
            if emitted <= allowance_bytes:
                admitted.append(AdvisoryCandidate(candidate.label, candidate.path, size))
                admitted_bytes += size
                aggregate = emitted
            else:
                omitted.append(OmittedAdvisoryInput(candidate.label, size, "exceeds-byte-budget"))
            continue
        max_bytes, aggregate_limit, _ = _transport_allowance(transport, aggregate)
        if size <= max_bytes:
            admitted.append(AdvisoryCandidate(candidate.label, candidate.path, size))
            admitted_bytes += size
            aggregate += size
        else:
            reason = "exceeds-byte-budget" if aggregate_limit else "exceeds-file-limit"
            omitted.append(OmittedAdvisoryInput(candidate.label, size, reason))
    return AdvisorySelection(
        transport=transport,
        admitted=tuple(admitted),
        omitted=tuple(omitted),
        admitted_bytes=admitted_bytes,
        allowance_bytes=allowance_bytes,
    )


def prepare_sanctioned_inputs(
    backend: object,
    cwd: Path,
    inputs: Mapping[str, Path],
    *,
    read_only: bool,
) -> PreparedSanctionedInputs:
    """Capture a closed logical-label mapping for one backend call."""
    canonical_cwd = _canonical_cwd(cwd)
    transport = _sanctioned_transport(backend, canonical_cwd, read_only=read_only)
    if transport is SanctionedInputTransport.EXACT_PATHS and len(inputs) > SANCTIONED_EXACT_INPUT_MAX_FILES:
        raise SanctionedInputUnavailable("sanctioned input file count limit exceeded")
    prepared: list[PreparedSanctionedInput] = []
    aggregate = 0
    for label, path in sorted(inputs.items()):
        if not label or len(label) > 128 or any(ord(c) < 32 or ord(c) == 127 or c in '<>"' for c in label):
            raise SanctionedInputUnavailable("sanctioned input label is invalid")
        item = _capture_input(label, path, transport, aggregate)
        aggregate += item.size
        prepared.append(item)
    return PreparedSanctionedInputs(
        transport=transport,
        inputs=tuple(prepared),
        backend_identity=backend,
        cwd=canonical_cwd,
        read_only=read_only,
    )


def fits_inline_diff_budget(text: str) -> bool:
    """Whether ``text`` fits the UTF-8 byte budget for an inlined diff."""
    return len(text.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES


def _sanctioned_inline_open_tag(label: str) -> str:
    return f'<sanctioned-input label="{label}">'


def inline_section_emitted_bytes(entries: Sequence[tuple[str, int]]) -> int:
    """Exact UTF-8 byte length of the INLINE render for these captured inputs.

    ``entries`` pairs each ``(label, content_bytes)`` exactly as
    :meth:`PreparedSanctionedInputs.render` receives them, so callers can size
    the emitted block — header, tags, newline separators, and content — before
    capturing anything. ``()`` matches the renderer's empty result: zero bytes.
    """
    if not entries:
        return 0
    sizes = [len(_SANCTIONED_INLINE_HEADER.encode("utf-8"))]
    for label, content_bytes in entries:
        sizes.append(len(_sanctioned_inline_open_tag(label).encode("utf-8")))
        sizes.append(content_bytes)
        sizes.append(len(_SANCTIONED_INLINE_CLOSE_TAG.encode("utf-8")))
    return sum(sizes) + max(len(sizes) - 1, 0)


def truncate_utf8_to_budget(text: str, budget_bytes: int, marker: str = "") -> str:
    """Byte-exact prefix of ``text`` that, with ``marker``, fits ``budget_bytes``.

    Returns ``text`` unchanged when it plus ``marker`` already fits. Otherwise
    slices ``text.encode("utf-8")`` (never ``str`` indices) and decodes the
    prefix with ``errors="ignore"`` so a split multibyte sequence is dropped
    rather than replaced, keeping the result ``<= budget_bytes`` UTF-8 bytes.
    A marker larger than the budget is itself truncated rather than raising.
    """
    marker_bytes = marker.encode("utf-8")
    encoded = text.encode("utf-8")
    if len(encoded) + len(marker_bytes) <= budget_bytes:
        return text
    if len(marker_bytes) >= budget_bytes:
        return marker_bytes[:budget_bytes].decode("utf-8", errors="ignore")
    keep = budget_bytes - len(marker_bytes)
    return encoded[:keep].decode("utf-8", errors="ignore") + marker
