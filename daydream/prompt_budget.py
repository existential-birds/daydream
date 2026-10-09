"""Dependency-neutral prompt-size, sanctioned-input, and advisory-selection policy."""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from daydream.artifact_visibility import ArtifactVisibilityError
from daydream.backends import AUDIT_ROOT_ISOLATION
from daydream.review_source import SourceRecipe, SourceWindow

# Upper bound for inlined diff text. Above this bound, prompts retain an
# on-disk diff pointer rather than embedding the diff.
INLINE_DIFF_BUDGET_BYTES = 12_288
SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES = INLINE_DIFF_BUDGET_BYTES
SANCTIONED_EXACT_INPUT_MAX_FILES = 512
SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES = 1_048_576
SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES = 8_388_608
# Durable references cost streaming I/O, not model context. Bound that resource
# independently of captured prompt inputs (one required diff per input set).
SANCTIONED_DIFF_REFERENCE_MAX_BYTES = 128 * 1024 * 1024

_SANCTIONED_INLINE_HEADER = "Sanctioned phase inputs (captured verbatim):"
_SANCTIONED_INLINE_CLOSE_TAG = "</sanctioned-input>"


class SanctionedInputUnavailable(ArtifactVisibilityError):
    """A declared model input could not be captured without widening access."""


class SourceAccessUnavailable(SanctionedInputUnavailable):
    """Frozen source could not be served or independently revalidated."""


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
    pointer_only: bool = False
    source: SourceWindow | None = None
    prompt_visible: bool = True
    inline_advisory: bool = False


@dataclass(frozen=True)
class PreparedSanctionedInputs:
    """Closed inputs bound to one backend, cwd, and access mode."""

    transport: SanctionedInputTransport
    inputs: tuple[PreparedSanctionedInput, ...]
    backend_identity: object
    cwd: Path
    read_only: bool
    source_recipe: SourceRecipe | None = None

    def render(self) -> str:
        """Render deterministic path pointers or captured inline bytes."""
        if not self.inputs:
            return ""
        if self.transport is SanctionedInputTransport.EXACT_PATHS:
            lines = [
                "The exact-file restriction below applies to host phase artifacts only. "
                "It does not restrict repository source reads permitted by the assigned task; "
                "do not browse other host artifacts.",
                "Sanctioned phase inputs (read only these exact files):",
            ]
            lines.extend(f"- {item.label}: {item.path}" for item in self.inputs
                         if item.source is None and item.prompt_visible and not item.inline_advisory)
            for item in self.inputs:
                if item.inline_advisory:
                    lines.extend([_sanctioned_inline_open_tag(item.label), item.text or "",
                                  _SANCTIONED_INLINE_CLOSE_TAG])
            return "\n".join(lines)
        blocks = [_SANCTIONED_INLINE_HEADER]
        for item in self.inputs:
            blocks += [_sanctioned_inline_open_tag(item.label), item.text or "", _SANCTIONED_INLINE_CLOSE_TAG]
        return "\n".join(blocks)

    def render_prompt(self, prompt: str) -> str:
        """Append sanctioned inputs once, scrubbing inline private paths before comparison.

        Scrub both prompt and captured content, including their common parent: builders
        may mention unsanctioned sibling artifacts under that same private root."""
        rendered = self.render()
        if self.transport is SanctionedInputTransport.INLINE and self.inputs:
            prompt = self._scrub_private_paths(prompt)
            rendered = self._scrub_private_paths(rendered)
        if not rendered:
            return prompt
        if prompt.endswith(rendered):
            return prompt
        return f"{prompt}\n\n{rendered}"

    def _scrub_private_paths(self, text: str) -> str:
        """Replace input paths with labels, then their common parent with a storage marker."""
        for item in self.inputs:
            text = text.replace(str(item.path), f"sanctioned input '{item.label}'")
        common_parent = os.path.commonpath([str(item.path.parent) for item in self.inputs])
        if common_parent and common_parent != os.path.sep:
            text = text.replace(common_parent, "sanctioned artifact storage")
        return text

    def finalization_text(
        self, backend: object, cwd: Path, read_only: bool, *, input_priority: tuple[str, ...] = (),
    ) -> str:
        """Read bounded prefixes for tool-less output, validating the whole input first.

        Exact-path captures retain identity/hash; recheck that identity before exposing
        any newly captured bytes."""
        self.revalidate(backend, cwd, read_only)
        blocks: list[str] = []
        remaining = 24000
        aggregate = 0
        priorities = {label: index for index, label in enumerate(input_priority)}
        inputs = sorted(self.inputs, key=lambda item: priorities.get(item.label, len(priorities)))
        for item in inputs:
            if item.pointer_only:
                blocks.append(
                    f"Reference {item.label!r}: {item.path} (not captured; use completed investigation evidence)"
                )
                continue
            if remaining <= 0:
                blocks.append("[remaining sanctioned inputs omitted; coverage incomplete]")
                break
            limit = min(remaining, 12000)
            current = _capture_input(item.label, item.path, self.transport, aggregate, text_budget=limit)
            aggregate += current.size
            if replace(current, text=item.text, source=item.source, prompt_visible=item.prompt_visible,
                       inline_advisory=item.inline_advisory) != item:
                raise SanctionedInputUnavailable(f"sanctioned input {item.label!r} changed before finalization")
            text = current.text or ""
            block = f"Input {item.label!r} (sha256={item.sha256}):\n{text}"
            remaining -= len(block.encode("utf-8"))
            blocks.append(block)
            if len(text.encode("utf-8")) < item.size:
                blocks.append("[input truncated; missing bytes do not establish coverage]")
        rendered = truncate_utf8_to_budget("\n\n".join(blocks), 24000, "[sanctioned context truncated]")
        if self.transport is SanctionedInputTransport.INLINE and self.inputs:
            rendered = self._scrub_private_paths(rendered)
        return rendered

    def revalidate(self, backend: object, cwd: Path, read_only: bool) -> None:
        """Fail closed if call identity or any captured file changed."""
        if backend is not self.backend_identity:
            raise SanctionedInputUnavailable("sanctioned input backend changed")
        if self.source_recipe is not None:
            self.source_recipe.revalidate()
        canonical_cwd = _canonical_cwd(cwd)
        if canonical_cwd != self.cwd or read_only is not self.read_only:
            raise SanctionedInputUnavailable("sanctioned input call mode changed")
        if _sanctioned_transport(backend, canonical_cwd, read_only=read_only) is not self.transport:
            raise SanctionedInputUnavailable("sanctioned input transport mode changed before model execution")
        aggregate = 0
        for item in self.inputs:
            if not item.pointer_only and _unchanged_since_capture(item):
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
            current = _capture_input(
                item.label, item.path, self.transport, aggregate, pointer_only=item.pointer_only,
                text_budget=item.size if item.inline_advisory else None,
            )
            if not item.pointer_only:
                aggregate += current.size
            if replace(current, source=item.source, prompt_visible=item.prompt_visible,
                       inline_advisory=item.inline_advisory) != item:
                raise SanctionedInputUnavailable(f"sanctioned input {item.label!r} changed before model execution")


@dataclass(frozen=True)
class AdvisoryCandidate:
    """An advisory input admitted or omitted whole; selection fills its measured size."""

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
    """Check captured (dev, ino, size, mtime_ns) without rereading file content.

    Missing, unreadable, or changed files trigger full capture and its original
    fail-closed errors."""
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
    label: str, path: Path, transport: SanctionedInputTransport, aggregate: int, *,
    text_budget: int | None = None, pointer_only: bool = False,
) -> PreparedSanctionedInput:
    """Capture one no-follow inode within the remaining transport allowance.

    Read at most allowance + 1 bytes; fstat before/after binds one immutable revision."""
    max_bytes, aggregate_limit, limit_name = (
        (SANCTIONED_DIFF_REFERENCE_MAX_BYTES, False, "durable diff resource limit")
        if pointer_only else _transport_allowance(transport, aggregate)
    )
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
        keep_text = not pointer_only and (transport is SanctionedInputTransport.INLINE or text_budget is not None)
        chunks: list[bytes] | None = [] if keep_text else None
        retained = 0
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
                prefix = chunk if text_budget is None else chunk[:max(text_budget - retained, 0)]
                chunks.append(prefix)
                retained += len(prefix)
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
        text=(
            payload.decode("utf-8", errors="ignore" if text_budget is not None else "strict")
            if payload is not None else None
        ),
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        pointer_only=pointer_only,
    )


def _sanctioned_transport(backend: object, canonical_cwd: Path, *, read_only: bool) -> SanctionedInputTransport:
    """Choose inline transport unless a concrete backend declares exact host-path access."""
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
    """Resolve the capture transport early so builders can size advisory inputs."""
    return _sanctioned_transport(backend, cwd, read_only=read_only)


def uses_diff_reference(backend: object, cwd: Path, *, read_only: bool) -> bool:
    """Pi discovery reads its durable diff through an admitted host path."""
    from daydream.backends.pi import PiBackend

    return isinstance(backend, PiBackend) and sanctioned_transport_for(
        backend, cwd, read_only=read_only,
    ) is SanctionedInputTransport.EXACT_PATHS


def select_advisory_inputs(
    backend: object,
    cwd: Path,
    candidates: Sequence[AdvisoryCandidate],
    *,
    read_only: bool,
) -> AdvisorySelection:
    """Admit candidates whole, in order, within the resolved transport’s allowance.

    INLINE accounts for exact rendered bytes; EXACT_PATHS uses capture limits.
    Unavailable candidates are omitted; transport resolution errors propagate."""
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
    source_recipe: SourceRecipe | None = None,
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
        pointer_only = label == "diff" and uses_diff_reference(backend, canonical_cwd, read_only=read_only)
        item = _capture_input(label, path, transport, aggregate, pointer_only=pointer_only)
        if not pointer_only:
            aggregate += item.size
        source = next((window for window in source_recipe.windows if window.projection == item.path), None) if (
            source_recipe is not None) else None
        prepared.append(replace(item, source=source))
    return PreparedSanctionedInputs(
        transport=transport,
        inputs=tuple(prepared),
        backend_identity=backend,
        cwd=canonical_cwd,
        read_only=read_only,
        source_recipe=source_recipe,
    )


def fits_inline_diff_budget(text: str) -> bool:
    """Whether ``text`` fits the UTF-8 byte budget for an inlined diff."""
    return len(text.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES


def _sanctioned_inline_open_tag(label: str) -> str:
    return f'<sanctioned-input label="{label}">'


def inline_section_emitted_bytes(entries: Sequence[tuple[str, int]]) -> int:
    """Count exact UTF-8 bytes for (label, content_bytes) entries, including all wrappers."""
    if not entries:
        return 0
    sizes = [len(_SANCTIONED_INLINE_HEADER.encode("utf-8"))]
    for label, content_bytes in entries:
        sizes.append(len(_sanctioned_inline_open_tag(label).encode("utf-8")))
        sizes.append(content_bytes)
        sizes.append(len(_SANCTIONED_INLINE_CLOSE_TAG.encode("utf-8")))
    return sum(sizes) + max(len(sizes) - 1, 0)


def inline_context_file(path: Path, budget_bytes: int = 4096) -> str | None:
    """Inline a whole bounded host-context artifact, else use its pointer; never a source receipt."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(budget_bytes + 1)
        if len(raw) <= budget_bytes:
            return raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        pass
    return None


def truncate_utf8_to_budget(text: str, budget_bytes: int, marker: str = "") -> str:
    """Fit text plus marker within a UTF-8 byte budget without splitting characters.

    Keep fitting text unchanged. Oversized markers are also truncated; incomplete
    multibyte suffixes are dropped rather than replaced."""
    marker_bytes = marker.encode("utf-8")
    encoded = text.encode("utf-8")
    if len(encoded) + len(marker_bytes) <= budget_bytes:
        return text
    if len(marker_bytes) >= budget_bytes:
        return marker_bytes[:budget_bytes].decode("utf-8", errors="ignore")
    keep = budget_bytes - len(marker_bytes)
    return encoded[:keep].decode("utf-8", errors="ignore") + marker
