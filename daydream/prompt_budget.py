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
SANCTIONED_PHASE_INPUT_BUDGET_BYTES = INLINE_DIFF_BUDGET_BYTES
SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES = INLINE_DIFF_BUDGET_BYTES
SANCTIONED_EXACT_INPUT_MAX_FILES = 512
SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES = 1_048_576
SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES = 4_194_304
SANCTIONED_INPUT_UNAVAILABLE = "sanctioned_input_unavailable"


class SanctionedInputUnavailable(ArtifactVisibilityError):
    """A declared model input could not be captured without widening access."""

    reason = SANCTIONED_INPUT_UNAVAILABLE


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
    payload_bytes: bytes | None
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
        blocks = ["Sanctioned phase inputs (captured verbatim):"]
        for item in self.inputs:
            blocks.extend(
                (
                    f'<sanctioned-input label="{item.label}">',
                    item.text or "",
                    "</sanctioned-input>",
                )
            )
        return "\n".join(blocks)

    def render_prompt(self, prompt: str) -> str:
        """Append the inputs, hiding private pathnames from inline transports."""
        if self.transport is SanctionedInputTransport.INLINE:
            for item in self.inputs:
                prompt = prompt.replace(str(item.path), f"sanctioned input '{item.label}'")
            parents = [str(item.path.parent) for item in self.inputs]
            common_parent = os.path.commonpath(parents) if parents else ""
            if common_parent and common_parent != os.path.sep:
                prompt = prompt.replace(common_parent, "sanctioned artifact storage")
        rendered = self.render()
        return f"{prompt}\n\n{rendered}" if rendered else prompt

    def revalidate(self, backend: object, cwd: Path, read_only: bool) -> None:
        """Fail closed if call identity or any captured file changed."""
        if backend is not self.backend_identity:
            raise SanctionedInputUnavailable("sanctioned input backend changed")
        try:
            canonical_cwd = cwd.resolve(strict=True)
        except OSError as exc:
            raise SanctionedInputUnavailable("sanctioned input cwd is unavailable") from exc
        if canonical_cwd != self.cwd or read_only is not self.read_only:
            raise SanctionedInputUnavailable("sanctioned input call mode changed")
        if _sanctioned_transport(backend, canonical_cwd, read_only=read_only) is not self.transport:
            raise SanctionedInputUnavailable(
                "sanctioned input transport mode changed before model execution"
            )
        aggregate = 0
        for item in self.inputs:
            aggregate_limit = False
            if self.transport is SanctionedInputTransport.INLINE:
                max_bytes = SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - aggregate
            else:
                remaining = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES - aggregate
                max_bytes = min(SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES, remaining)
                aggregate_limit = remaining < SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES
            if _unchanged_since_capture(item):
                # The (dev, ino, size, mtime_ns) identity is unchanged since
                # the capture that already streamed and hashed these exact
                # bytes, so the re-read/re-hash is redundant on this attempt.
                # The aggregate budget stays enforced exactly as a fresh
                # capture would enforce it. Invariant: unchanged items carry
                # their capture-time sizes, whose sum capture already checked
                # against the cap, and a re-captured item either matches
                # item (same size) or fails closed below — so this guard
                # pins the cap fail-closed even if that invariant drifts.
                if item.size > max_bytes:
                    raise SanctionedInputUnavailable(
                        f"sanctioned input {item.label!r} exceeds remaining "
                        "aggregate input budget"
                    )
                aggregate += item.size
                continue
            current = _capture_input(
                item.label,
                item.path,
                transport=self.transport,
                max_bytes=max(max_bytes, 0),
                aggregate_limit=aggregate_limit,
            )
            aggregate += current.size
            if current != item:
                raise SanctionedInputUnavailable(
                    f"sanctioned input {item.label!r} changed before model execution"
                )


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


def _capture_input(
    label: str,
    path: Path,
    *,
    transport: SanctionedInputTransport,
    max_bytes: int,
    aggregate_limit: bool = False,
) -> PreparedSanctionedInput:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    fd = -1
    try:
        lexical = Path(os.path.abspath(path))
        lexical_stat = lexical.lstat()
        if not stat.S_ISREG(lexical_stat.st_mode):
            raise SanctionedInputUnavailable(
                f"sanctioned input {label!r} must be a regular file"
            )
        fd = os.open(lexical, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SanctionedInputUnavailable(
                f"sanctioned input {label!r} must be a regular file"
            )
        if (lexical_stat.st_dev, lexical_stat.st_ino) != (before.st_dev, before.st_ino):
            raise SanctionedInputUnavailable(
                f"sanctioned input {label!r} changed while being opened"
            )
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        chunks: list[bytes] | None = (
            [] if transport is SanctionedInputTransport.INLINE else None
        )
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                if aggregate_limit:
                    raise SanctionedInputUnavailable(
                        "sanctioned input aggregate byte limit exceeded"
                    )
                limit = (
                    "byte budget"
                    if transport is SanctionedInputTransport.INLINE
                    else "file byte limit"
                )
                raise SanctionedInputUnavailable(
                    f"sanctioned input {label!r} exceeds the {limit}"
                )
            digest.update(chunk)
            decoder.decode(chunk, final=False)
            if chunks is not None:
                chunks.append(chunk)
        decoder.decode(b"", final=True)
        after = os.fstat(fd)
    except SanctionedInputUnavailable:
        raise
    except UnicodeDecodeError as exc:
        raise SanctionedInputUnavailable(
            f"sanctioned input {label!r} is not valid UTF-8"
        ) from exc
    except OSError as exc:
        raise SanctionedInputUnavailable(
            f"sanctioned input {label!r} is unavailable"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SanctionedInputUnavailable(
            f"sanctioned input {label!r} changed while being captured"
        )
    payload = b"".join(chunks) if chunks is not None else None
    text = payload.decode("utf-8") if payload is not None else None
    return PreparedSanctionedInput(
        label=label,
        path=lexical,
        text=text,
        payload_bytes=payload,
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


def _sanctioned_transport(
    backend: object, cwd: Path, *, read_only: bool
) -> SanctionedInputTransport:
    try:
        canonical_cwd = cwd.resolve(strict=True)
    except OSError as exc:
        raise SanctionedInputUnavailable("sanctioned input cwd is unavailable") from exc
    audit_root = getattr(backend, "audit_root", None)
    audit_capability = getattr(backend, "audit_root_isolation", None)
    try:
        if audit_capability == AUDIT_ROOT_ISOLATION_V1:
            if not isinstance(audit_root, Path) or audit_root.resolve(strict=True) != canonical_cwd:
                raise SanctionedInputUnavailable(
                    "sanctioned input audit root does not match the model cwd"
                )
            strict_audit = True
        else:
            strict_audit = False
    except OSError as exc:
        raise SanctionedInputUnavailable(
            "sanctioned input audit root is unavailable"
        ) from exc
    disposable_read_only = read_only and bool(
        getattr(backend, "read_only_disposable_clone", False)
    )
    sandboxed_osprey = bool(getattr(backend, "sandbox", False))
    if strict_audit or disposable_read_only or sandboxed_osprey:
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
    try:
        canonical_cwd = cwd.resolve(strict=True)
    except OSError as exc:
        raise SanctionedInputUnavailable("sanctioned input cwd is unavailable") from exc
    transport = _sanctioned_transport(backend, canonical_cwd, read_only=read_only)
    if (
        transport is SanctionedInputTransport.EXACT_PATHS
        and len(inputs) > SANCTIONED_EXACT_INPUT_MAX_FILES
    ):
        raise SanctionedInputUnavailable("sanctioned input file count limit exceeded")
    prepared: list[PreparedSanctionedInput] = []
    aggregate = 0
    for label, path in sorted(inputs.items()):
        if (
            not label
            or len(label) > 128
            or any(ord(char) < 32 or ord(char) == 127 for char in label)
            or any(char in label for char in '<>"')
        ):
            raise SanctionedInputUnavailable("sanctioned input label is invalid")
        aggregate_limit = False
        if transport is SanctionedInputTransport.INLINE:
            per_file_limit = SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - aggregate
        else:
            remaining = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES - aggregate
            per_file_limit = min(SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES, remaining)
            aggregate_limit = remaining < SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES
        item = _capture_input(
            label,
            path,
            transport=transport,
            max_bytes=max(per_file_limit, 0),
            aggregate_limit=aggregate_limit,
        )
        aggregate += item.size
        if (
            transport is SanctionedInputTransport.EXACT_PATHS
            and aggregate > SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES
        ):
            raise SanctionedInputUnavailable(
                "sanctioned input aggregate byte limit exceeded"
            )
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
