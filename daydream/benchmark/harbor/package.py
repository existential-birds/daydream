"""Package compiled benchmark tasks for Harbor 0.21."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from daydream.benchmark.harbor.build import CompileError

GENERATION_COMMAND = "uv export --frozen --no-dev --no-emit-project --format requirements-txt"


class PackageError(CompileError):
    """Raised when a compiled Harbor package cannot be produced or validated."""

    def __init__(self, message: str, *, remediation: str = "") -> None:
        self.remediation = remediation
        suffix = f" Remediation: {remediation}" if remediation else ""
        super().__init__(message + suffix)


@dataclass(frozen=True)
class WheelInfo:
    """Validated Daydream wheel identity and content digest."""

    distribution: str
    version: str
    sha256: str


def validate_wheel(wheel_path: Path, *, daydream_version: str) -> WheelInfo:
    """Validate the exact wheel filename for the running Daydream release."""
    wheel_path = Path(wheel_path)
    expected = f"daydream-{daydream_version}-py3-none-any.whl"
    if not wheel_path.is_file() or wheel_path.name != expected:
        raise PackageError(
            f"wheel {wheel_path.name!r} is missing or mismatched; expected {expected!r}",
            remediation="run `uv build --wheel` and pass the wheel for the running Daydream version",
        )
    if not re.fullmatch(r"daydream-[^-]+-py3-none-any\.whl", wheel_path.name):
        raise PackageError(
            f"wheel {wheel_path.name!r} is not the expected Daydream wheel {expected!r}",
            remediation="run `uv build --wheel`",
        )
    try:
        digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PackageError(
            f"cannot read wheel {wheel_path}: {exc}", remediation="run `uv build --wheel`"
        ) from exc
    return WheelInfo(distribution="daydream", version=daydream_version, sha256=digest)


def render_lock_header(
    daydream_version: str,
    source_sha256: str,
    gen_command: str,
    template_version: str,
) -> str:
    """Render deterministic provenance for the packaged runtime lock."""
    return (
        "# Daydream Harbor runtime requirements (generated; do not edit)\n"
        f"# daydream_version: {daydream_version}\n"
        f"# source_uv_lock_sha256: {source_sha256}\n"
        f"# generation_command: {gen_command}\n"
        f"# template_version: {template_version}\n"
        "\n"
    )


def _strip_uv_header(text: str) -> str:
    """Strip uv's leading generated-comment block while preserving requirement comments."""
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines) and (lines[index].startswith("#") or not lines[index].strip()):
        index += 1
    return "".join(lines[index:])


def _uv_export_body(uv_lock_path: Path) -> str:
    """Export hash-pinned runtime requirements from *uv_lock_path*."""
    command = GENERATION_COMMAND.split()
    try:
        result = subprocess.run(
            command,
            cwd=Path(uv_lock_path).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise PackageError(f"cannot run `{GENERATION_COMMAND}`: {exc}") from exc
    if result.returncode != 0:
        raise PackageError(
            f"`{GENERATION_COMMAND}` failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    body = _strip_uv_header(result.stdout)
    if "--hash=sha256:" not in body:
        raise PackageError(f"`{GENERATION_COMMAND}` did not produce hash-pinned requirements")
    return body


def render_runtime_lock(uv_lock_path: Path, *, daydream_version: str) -> tuple[str, str]:
    """Return the deterministic header and exported requirements body."""
    from daydream.benchmark.harbor.build import TEMPLATE_VERSION

    uv_lock_path = Path(uv_lock_path)
    try:
        source_sha256 = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PackageError(f"cannot read source uv lock {uv_lock_path}: {exc}") from exc
    body = _uv_export_body(uv_lock_path)
    if f"daydream=={daydream_version}" in body:
        raise PackageError("runtime lock unexpectedly includes the Daydream project")
    header = render_lock_header(daydream_version, source_sha256, GENERATION_COMMAND, TEMPLATE_VERSION)
    return header, body


def generate_runtime_lock(uv_lock_path: Path, *, daydream_version: str) -> bytes:
    """Generate complete packaged lock bytes."""
    header, body = render_runtime_lock(uv_lock_path, daydream_version=daydream_version)
    return (header + body).encode("utf-8")


def write_runtime_lock(output: str | Path, uv_lock_path: Path, daydream_version: str) -> None:
    """Generate and write the packaged runtime lock."""
    Path(output).write_bytes(generate_runtime_lock(uv_lock_path, daydream_version=daydream_version))
