"""Dependency-neutral prompt-size and sanctioned-input policy."""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from daydream.artifact_visibility import ArtifactVisibilityError
from daydream.backends import AUDIT_ROOT_ISOLATION_V1

# Upper bound for inlined diff text. Above this bound, prompts retain an
# on-disk diff pointer rather than embedding the diff.
INLINE_DIFF_BUDGET_BYTES = 12_288
SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES = INLINE_DIFF_BUDGET_BYTES
SANCTIONED_EXACT_INPUT_MAX_FILES = 512
SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES = 1_048_576
SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES = 4_194_304


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
        blocks = ["Sanctioned phase inputs (captured verbatim):"]
        for item in self.inputs:
            blocks += [f'<sanctioned-input label="{item.label}">', item.text or "", "</sanctioned-input>"]
        return "\n".join(blocks)

    def render_prompt(self, prompt: str) -> str:
        """Append the inputs, hiding private pathnames from inline transports.

        Builders suppress the pointers they own, but one still names artifacts
        it did not sanction (a sibling under the same private root), so the
        inputs' common parent is scrubbed too.
        """
        if self.transport is SanctionedInputTransport.INLINE and self.inputs:
            for item in self.inputs:
                prompt = prompt.replace(str(item.path), f"sanctioned input '{item.label}'")
            common_parent = os.path.commonpath([str(item.path.parent) for item in self.inputs])
            if common_parent and common_parent != os.path.sep:
                prompt = prompt.replace(common_parent, "sanctioned artifact storage")
        rendered = self.render()
        return f"{prompt}\n\n{rendered}" if rendered else prompt

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
            current = _capture_input(item.label, item.path, self.transport, aggregate)
            aggregate += current.size
            if current != item:
                raise SanctionedInputUnavailable(f"sanctioned input {item.label!r} changed before model execution")


def _canonical_cwd(cwd: Path) -> Path:
    try:
        return cwd.resolve(strict=True)
    except OSError as exc:
        raise SanctionedInputUnavailable("sanctioned input cwd is unavailable") from exc


def _capture_input(
    label: str, path: Path, transport: SanctionedInputTransport, aggregate: int
) -> PreparedSanctionedInput:
    """Capture one no-follow file within the transport's remaining allowance.

    Reading stops one byte past the allowance, so a refusal never hashes more
    than the aggregate ceiling admits. ``fstat`` before and after bounds the
    captured bytes to one immutable revision of one inode.
    """
    aggregate_limit = False
    limit_name = "byte budget"
    if transport is SanctionedInputTransport.INLINE:
        max_bytes = SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - aggregate
    else:
        remaining = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES - aggregate
        max_bytes = min(SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES, remaining)
        aggregate_limit = remaining < SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES
        limit_name = "file byte limit"
    max_bytes = max(max_bytes, 0)
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
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


def _sanctioned_transport(backend: object, canonical_cwd: Path, *, read_only: bool) -> SanctionedInputTransport:
    """Inline the inputs for a backend that cannot read exact host paths.

    "Can this backend read this host path" is not on the ``Backend`` protocol,
    so it is read off the three concrete backends that declare it.
    """
    strict_audit = getattr(backend, "audit_root_isolation", None) == AUDIT_ROOT_ISOLATION_V1
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
