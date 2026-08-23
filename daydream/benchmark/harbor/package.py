"""Package compiled benchmark tasks for Harbor 0.21."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from daydream.benchmark.harbor.build import CompileError

GENERATION_COMMAND = "uv export --frozen --no-dev --no-emit-project --format requirements-txt"
_BASE_DIGEST = "sha256:876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4"
ENV_BASE_IMAGE = f"python:3.12-slim@{_BASE_DIGEST}"
VERIFIER_BASE_IMAGE = ENV_BASE_IMAGE


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


def resolve_harbor() -> str:
    """Resolve a compatible Harbor executable from this Python environment."""
    remediation = "pip install 'daydream[benchmark]'"
    try:
        version = importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PackageError(
            "Harbor is not installed in the running Daydream interpreter", remediation=remediation
        ) from exc
    try:
        parts = tuple(int(part) for part in version.split(".")[:2])
    except ValueError as exc:
        raise PackageError(
            f"Harbor version {version!r} is invalid; supported range is [0.21, 0.22)",
            remediation=remediation,
        ) from exc
    if parts != (0, 21):
        raise PackageError(
            f"Harbor version {version} is outside supported range [0.21, 0.22)",
            remediation=remediation,
        )
    executable = Path(sys.executable).parent / "harbor"
    if not executable.is_file():
        raise PackageError(
            f"Harbor metadata is present but {executable} is missing; install Harbor into the Daydream interpreter",
            remediation=remediation,
        )
    return str(executable)


def render_verifier_dockerfile(*, base_image: str) -> bytes:
    """Render and validate the entrypoint-free separate verifier image."""
    template = Path(__file__).parent.joinpath("templates/tests/Dockerfile").read_text()
    text = template.replace("__BASE_IMAGE__", base_image)
    forbidden = ("ENTRYPOINT", "CMD", "/verifier", "httpx>=")
    violations = [token for token in forbidden if token in text]
    if violations or "httpx==0.28.1" not in text:
        raise PackageError(
            f"verifier Dockerfile template violates pinned entrypoint-free contract: {violations}"
        )
    return text.encode("utf-8")


def render_environment_dockerfile(*, base_image: str, daydream_version: str) -> bytes:
    """Render and validate the isolated agent environment image."""
    template = Path(__file__).parent.joinpath("templates/environment/Dockerfile").read_text()
    text = template.replace("__BASE_IMAGE__", base_image).replace(
        "__DAYDREAM_VERSION__", daydream_version
    )
    required = (
        "git clone",
        "repository.bundle",
        "/workspace/repo",
        "checkout head",
        "rev-parse --verify base",
        "rev-parse --verify head",
        "remote remove",
        "WORKDIR /workspace/repo",
        "--require-hashes",
        "--no-deps",
    )
    missing = [directive for directive in required if directive not in text]
    if missing:
        raise PackageError(f"environment Dockerfile template is missing directives: {missing}")
    return text.encode("utf-8")


def render_task_toml(opaque_key: str) -> bytes:
    """Render the fixed Harbor schema-1.4 task configuration."""
    # TOML is intentionally rendered directly: fixed ordering and no timestamp
    # make these bytes part of the deterministic compiled-tree contract.
    return f'''schema_version = "1.4"

[metadata]
benchmark_case_key = "{opaque_key}"
source_kind = "historic-github-pr"

[agent]
timeout_sec = 1800.0
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com"]

[environment]
network_mode = "no-network"
build_timeout_sec = 1200.0
workdir = "/workspace/repo"
cpus = 2
memory_mb = 4096
storage_mb = 10240

[verifier]
timeout_sec = 900.0
environment_mode = "separate"

[verifier.environment]
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com"]
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 2048
storage_mb = 4096
'''.encode("utf-8")


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
