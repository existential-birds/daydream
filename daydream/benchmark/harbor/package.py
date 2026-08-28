"""Package compiled benchmark tasks for Harbor 0.22."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import importlib.resources
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from daydream.benchmark import schema
from daydream.benchmark.harbor.build import CompileError

GENERATION_COMMAND = "uv export --frozen --no-dev --no-emit-project --format requirements-txt"
_BASE_DIGEST = "sha256:876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4"
ENV_BASE_IMAGE = f"python:3.12-slim@{_BASE_DIGEST}"
VERIFIER_BASE_IMAGE = ENV_BASE_IMAGE
PI_BASE_IMAGE = (
    "node:24.18.1-bookworm-slim@"
    "sha256:a09aabc645e86e81e23dab78e0c0f2eaa233cab4277c7188232181a1a8bd5d39"
)
PI_PACKAGE = "@earendil-works/pi-coding-agent@0.84.3"

# Marker comment pair delimiting the wheel-install block of the environment
# Dockerfile template. render_environment_dockerfile strips the delimited block
# verbatim when a wheel is not baked, so a wheel-less compile never emits a
# COPY/install referencing a wheel that was never written into environment/.
_WHEEL_BEGIN = "# __ENV_WHEEL_BEGIN__"
_WHEEL_END = "# __ENV_WHEEL_END__"


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


@dataclass(frozen=True)
class DockerNetworkPolicyCapability:
    """Result of loading Harbor's real Docker egress-control ruleset."""

    supported: bool
    reason: str = ""
    image_name: str | None = None


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
            f"Harbor version {version!r} is invalid; supported range is [0.22, 0.23)",
            remediation=remediation,
        ) from exc
    if parts != (0, 22):
        raise PackageError(
            f"Harbor version {version} is outside supported range [0.22, 0.23)",
            remediation=remediation,
        )
    executable = Path(sys.executable).parent / "harbor"
    if not executable.is_file():
        raise PackageError(
            f"Harbor metadata is present but {executable} is missing; install Harbor into the Daydream interpreter",
            remediation=remediation,
        )
    return str(executable)


def _ensure_harbor_egress_sidecar_image() -> str:
    """Build Harbor's content-addressed sidecar image through its own helper."""
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.environments.docker.utils import (
        default_docker_platform,
        ensure_docker_image_built,
    )

    async def ensure() -> str:
        platform = await default_docker_platform()
        return str(
            await ensure_docker_image_built(
                docker_name=DockerEnvironment._EGRESS_CONTROL_SIDECAR_DOCKER_NAME,
                docker_build_context=DockerEnvironment._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
                dockerfile_path=DockerEnvironment._egress_control_sidecar_dockerfile_path(),
                build_args={},
                platform=platform,
            )
        )

    return asyncio.run(ensure())


def _probe_harbor_egress_sidecar_image(
    image_name: str,
) -> subprocess.CompletedProcess[str]:
    """Load Harbor's actual nftables rules in a disposable isolated container."""
    return subprocess.run(
        [
            "docker",
            "container",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-add",
            "NET_ADMIN",
            "--cap-add",
            "NET_RAW",
            "--entrypoint",
            "/bin/sh",
            image_name,
            "-c",
            "network-policy deny-all >/dev/null && "
            "nft list table inet gost_egress >/dev/null",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _bounded_probe_error(result: subprocess.CompletedProcess[str]) -> str:
    detail = "\n".join(
        part.strip() for part in (result.stderr or "", result.stdout or "") if part.strip()
    )
    return detail[:1000] or f"probe exited {result.returncode} without output"


def docker_network_policy_capability() -> DockerNetworkPolicyCapability:
    """Return whether Harbor's real Docker allowlist backend works now.

    Harbor 0.22 fixes daemon-kernel capability detection, but a config-bit
    check alone is not sufficient for Daydream's paid-run preflight.  Build
    the exact content-addressed sidecar Harbor will use and load its complete
    nftables ruleset.  Any import, build, Docker, timeout, or ruleset failure
    is reported as unsupported; there is no public-networking fallback.
    """
    image_name: str | None = None
    try:
        resolve_harbor()
        image_name = _ensure_harbor_egress_sidecar_image()
        result = _probe_harbor_egress_sidecar_image(image_name)
    except Exception as exc:  # noqa: BLE001 - every probe failure must fail closed
        return DockerNetworkPolicyCapability(
            supported=False,
            reason=f"Harbor Docker egress sidecar live probe failed: {str(exc)[:1000]}",
            image_name=image_name,
        )
    if result.returncode != 0:
        return DockerNetworkPolicyCapability(
            supported=False,
            reason=(
                "Harbor Docker egress sidecar rejected its nftables rules: "
                f"{_bounded_probe_error(result)}"
            ),
            image_name=image_name,
        )
    return DockerNetworkPolicyCapability(supported=True, image_name=image_name)


def _read_packaged_resource(
    module: str, rel: str, fallback: Path, label: str
) -> str:
    """Read a packaged resource through the installed-release resource seam.

    Reads *rel* from the installed package *module*, falling back to the local
    source-tree *fallback* and raising ``PackageError`` when neither resolves.
    """
    try:
        resource = importlib.resources.files(module)
        for part in Path(rel).parts:
            resource = resource.joinpath(part)
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError) as resource_error:
        try:
            return fallback.read_text(encoding="utf-8")
        except OSError as exc:
            raise PackageError(
                f"missing packaged {label} {rel}: {resource_error}; {exc}"
            ) from exc


def template_text(rel: str) -> str:
    """Read a packaged template through the installed-release resource seam."""
    return _read_packaged_resource(
        "daydream.benchmark.harbor.templates",
        rel,
        Path(__file__).parent / "templates" / rel,
        label="template resource",
    )


def build_harbor(root: Path, *, wheel: Path) -> dict[str, Any]:
    """Validate all preconditions, then atomically compile a runnable dataset."""
    from daydream.benchmark.harbor import build
    from daydream.benchmark.workspace import validate_workspace

    code, label = validate_workspace(Path(root))
    if code != 0:
        raise PackageError(
            f"workspace must validate ready before build-harbor (validation: {label})",
            remediation="run `daydream benchmark validate <workspace>` and resolve every finding",
        )
    resolve_harbor()
    # compile_workspace validates the wheel up front on its own; there is no
    # preflight-benefit to hashing the (large) wheel twice in one build.
    return build.compile_workspace(Path(root), wheel=Path(wheel))


def _validate_compiled_local(root: Path) -> Path:
    """Verify compiled inventory hashes, exact file set, and control-plane leakage."""
    import json

    from daydream.benchmark.harbor import build

    compiled = root / "harbor"
    if not compiled.is_dir():
        raise PackageError(f"compiled dataset is missing: {compiled}")
    lock_path = compiled / "benchmark.lock.json"
    try:
        lock = json.loads(lock_path.read_text())
        inventory = lock["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PackageError(f"compiled lock is missing or invalid: {lock_path}: {exc}") from exc
    if not isinstance(inventory, dict):
        raise PackageError(f"compiled lock files inventory is invalid: {lock_path}")
    actual = {
        str(path.relative_to(compiled))
        for path in compiled.rglob("*")
        if path.is_file() and path != lock_path
    }
    expected = set(inventory)
    if actual != expected:
        raise PackageError(
            f"compiled file inventory mismatch (missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    for rel, expected_sha in sorted(inventory.items()):
        path = compiled / rel
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise PackageError(f"compiled file hash mismatch for {rel}: {actual_sha} != {expected_sha}")

    daydream = lock.get("daydream")
    if daydream and daydream.get("version") != importlib.metadata.version("daydream"):
        raise PackageError(
            f"compiled wheel version {daydream.get('version')} does not match running Daydream "
            f"{importlib.metadata.version('daydream')}",
            remediation="run `uv build --wheel` and rebuild with `daydream benchmark build-harbor`",
        )
    text_names = {
        "README.md", "instruction.md", "Task.md", "task.toml", "Dockerfile",
        "runtime-requirements.lock", "verifier-metadata.json", "harbor-job.yaml", "harbor-oracle.yaml",
    }
    control_plane = {
        rel: (compiled / rel).read_text(errors="replace")
        for rel in sorted(inventory)
        if Path(rel).name in text_names
    }
    cases = lock.get("cases") or {}
    repository_slug = next(iter(cases.values())).get("repository", "") if cases else ""
    build.leakage_scan(control_plane, repository_slug=repository_slug)
    return compiled


def validate_compiled(root: Path | None) -> int:
    """Validate local authoring/compiled state, all Harbor models, and the custom agent.

    Runs the authoring preflight, the compiled-local inventory/leakage scan, the
    same-interpreter Harbor resolution, the Harbor Task/JobConfig model checks, and
    the custom-agent import preflight (``daydream.benchmark.harbor.agent`` must
    import in this interpreter)."""
    if root is None:
        resolve_harbor()
        raise PackageError("compiled workspace path is required")
    from daydream.benchmark.workspace import validate_workspace

    root = Path(root)
    code, label = validate_workspace(root)
    if code != 0:
        raise PackageError(f"compiled workspace authoring validation failed: {label}")
    compiled = _validate_compiled_local(root)
    resolve_harbor()

    try:
        from harbor.models.job.config import JobConfig
        try:
            from harbor.models.task import Task
        except ImportError:  # Harbor exposes task as a namespace package in some wheels.
            from harbor.models.task.task import Task
    except ImportError as exc:
        raise PackageError(
            f"cannot import Harbor 0.22 models from the Daydream interpreter: {exc}",
            remediation="pip install 'daydream[benchmark]'",
        ) from exc

    for task_toml in sorted(compiled.glob("case-*/task.toml")):
        try:
            Task(str(task_toml.parent), disable_verification=True)
        except Exception as exc:
            raise PackageError(f"Harbor rejected task {task_toml.parent.name}: {exc}") from exc
    for name in ("harbor-job.yaml", "harbor-oracle.yaml"):
        path = compiled / name
        try:
            JobConfig.model_validate(yaml.safe_load(path.read_text()))
        except Exception as exc:
            raise PackageError(f"Harbor rejected {name}: {exc}") from exc

    # Custom-agent preflight: import the exact runnable agent path in the same
    # interpreter Harbor/validation share, so a missing or separate-environment
    # class fails before any trial is consumed.
    try:
        agent_mod = importlib.import_module("daydream.benchmark.harbor.agent")
        getattr(agent_mod, "DaydreamReviewAgent")
    except (ImportError, AttributeError) as exc:
        raise PackageError(
            f"cannot import custom Harbor agent "
            f"daydream.benchmark.harbor.agent:DaydreamReviewAgent from the "
            f"Daydream interpreter: {exc}",
            remediation="pip install 'daydream[benchmark]'",
        ) from exc
    return 0


def lock_text() -> str:
    """Read the packaged runtime lock through the installed-release resource seam."""
    return _read_packaged_resource(
        "daydream.benchmark.harbor",
        "runtime-requirements.lock",
        Path(__file__).parent / "runtime-requirements.lock",
        label="runtime lock",
    )


def runtime_lock_header_fields(text: str) -> dict[str, str]:
    """Extract deterministic provenance fields from a packaged runtime lock."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("# ") or ": " not in line:
            continue
        key, value = line[2:].split(": ", 1)
        if key in {"daydream_version", "source_uv_lock_sha256", "template_version"}:
            fields[key] = value
    required = {"daydream_version", "source_uv_lock_sha256", "template_version"}
    if fields.keys() != required:
        raise PackageError(f"runtime lock header is missing fields: {sorted(required - fields.keys())}")
    return fields


def render_job_config(*, oracle: bool) -> bytes:
    """Render a deterministic Harbor job or Oracle configuration."""
    agents: list[dict[str, Any]] = [{"name": "oracle"}] if oracle else [{
        "import_path": "daydream.benchmark.harbor.agent:DaydreamReviewAgent",
        # DAYDREAM_REVIEW_BACKEND is a pi|claude selection passed through verbatim;
        # supported values are validated downstream by the agent/entrypoint allowlist.
        "env": {
            "DAYDREAM_REVIEW_BACKEND": "${DAYDREAM_REVIEW_BACKEND:-pi}",
            "DAYDREAM_REVIEW_MODEL": "${DAYDREAM_REVIEW_MODEL}",
            "DAYDREAM_REVIEW_API_KEY": "${DAYDREAM_REVIEW_API_KEY}",
            "DAYDREAM_REVIEW_BASE_URL": "${DAYDREAM_REVIEW_BASE_URL}",
            "DAYDREAM_REVIEW_PROFILE_CANDIDATE": "${DAYDREAM_REVIEW_PROFILE_CANDIDATE:-}",
            # ANTHROPIC_* carries claude-backend credentials into the container
            # (agent.build_child_env keep-set, entrypoint claude branch); the
            # API key and auth token are alternatives, the base URL optional.
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
            "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN:-}",
            "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL:-}",
        },
    }]
    document = {
        "jobs_dir": "jobs",
        "n_attempts": 1,
        "n_concurrent_trials": 4,
        "environment": {"type": "docker", "delete": True},
        "agents": agents,
        "verifier": {"env": {
            "DAYDREAM_JUDGE_PROVIDER": "${DAYDREAM_JUDGE_PROVIDER}",
            "DAYDREAM_JUDGE_MODEL": "${DAYDREAM_JUDGE_MODEL}",
            "DAYDREAM_JUDGE_API_KEY": "${DAYDREAM_JUDGE_API_KEY}",
            "DAYDREAM_JUDGE_BASE_URL": "${DAYDREAM_JUDGE_BASE_URL}",
            # CLAUDE_CODE_* feeds the keyless claude-cli judge client; the
            # nonessential-traffic gate defaults on (fail-safe direction).
            "CLAUDE_CODE_OAUTH_TOKEN": "${CLAUDE_CODE_OAUTH_TOKEN}",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": (
                "${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}"
            ),
        }},
        "datasets": [{"path": "."}],
        "metrics": [{"type": "uv-script", "kwargs": {"script_path": "metric.py"}}],
    }
    try:
        return yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    except yaml.YAMLError as exc:
        raise PackageError(f"cannot serialize Harbor job config: {exc}") from exc


def render_verifier_dockerfile(*, base_image: str) -> bytes:
    """Render and validate the entrypoint-free separate verifier image."""
    text = _render_and_check(
        "tests/Dockerfile",
        replaces={"__BASE_IMAGE__": base_image},
        required=("httpx==0.28.1", "WORKDIR /tests", "verifier-metadata.json"),
        forbidden=("ENTRYPOINT", "CMD", "/verifier", "httpx>="),
        error_label="verifier Dockerfile template",
    )
    return text.encode("utf-8")


_ENV_REQUIRED: tuple[str, ...] = (
    PI_PACKAGE,
    "node --version",
    "pi --version",
    "git clone",
    "repository.bundle",
    "/workspace/repo",
    "checkout head",
    "rev-parse --verify base",
    "rev-parse --verify head",
    "remote remove",
    "WORKDIR /workspace/repo",
    "--require-hashes",
)


def _strip_wheel_block(text: str, *, wheel: bool) -> str:
    """Strip/keep the delimited wheel-install block of the environment template."""
    if _WHEEL_BEGIN not in text or _WHEEL_END not in text:
        raise PackageError("environment Dockerfile template is missing the wheel block markers")
    if wheel:
        return text.replace(_WHEEL_BEGIN, "").replace(_WHEEL_END, "")
    start = text.index(_WHEEL_BEGIN)
    end = text.index(_WHEEL_END) + len(_WHEEL_END)
    return text[:start] + text[end:]


def render_environment_dockerfile(*, base_image: str, daydream_version: str, wheel: bool = False) -> bytes:
    """Render and validate the isolated agent environment image.

    *wheel* selects whether the image installs the packaged Daydream wheel:
    True keeps the ``COPY``/install block (with the wheel copied into
    ``environment/`` by the compile path), False strips it so a wheel-less
    compile cannot emit a self-referential ``COPY`` of a wheel that is never
    written into the environment.
    """
    text = _render_and_check(
        "environment/Dockerfile",
        replaces={
            "__BASE_IMAGE__": base_image,
            "__PI_BASE_IMAGE__": PI_BASE_IMAGE,
            "__PI_PACKAGE__": PI_PACKAGE,
            "__DAYDREAM_VERSION__": daydream_version,
        },
        required=_ENV_REQUIRED + (("--no-deps",) if wheel else ()),
        error_label="environment Dockerfile template",
        transform=lambda rendered: _strip_wheel_block(rendered, wheel=wheel),
    )
    return text.encode("utf-8")


def _render_and_check(
    rel: str,
    *,
    replaces: dict[str, str],
    required: Sequence[str],
    forbidden: Sequence[str] = (),
    error_label: str,
    transform: Callable[[str], str] | None = None,
) -> str:
    """Read a Dockerfile template, substitute placeholders, then guard output.

    Every rendered Dockerfile surface goes through this one render+validate
    seam so their required/forbidden guard sets stay consistent: read the
    packaged template, apply the *replaces*, run the optional *transform*
    (e.g. wheel-block stripping), then assert the *required* and *forbidden*
    tokens are present/absent, raising ``PackageError`` on violation.
    """
    text = template_text(rel)
    for marker, value in replaces.items():
        text = text.replace(marker, value)
    if transform is not None:
        text = transform(text)
    missing = [token for token in required if token not in text]
    if missing:
        raise PackageError(f"{error_label} is missing directives: {missing}")
    violations = [token for token in forbidden if token in text]
    if violations:
        raise PackageError(f"{error_label} violates pinned contract: {violations}")
    return text


def _normalized_allowed_hosts(hosts: list[str] | None, label: str) -> list[str]:
    """Normalize and sort an allowlist, failing closed on empty/invalid input.

    Each host goes through ``schema.normalize_hostname`` (drops a
    ``<scheme>://`` prefix, ``user:pass@`` credentials, a ``:port`` suffix, and
    a trailing ``/path``; rejects wildcards, whitespace, empties, and dot-less
    segments). The normalized list is sorted so the rendered TOML bytes stay
    deterministic; a missing/empty list or a single bad host is a hard
    ``PackageError`` -- there is no silent fallback host.
    """
    if not hosts:
        raise PackageError(f"task network policy requires a non-empty {label} host list")
    normalized: list[str] = []
    for host in hosts:
        try:
            normalized.append(schema.normalize_hostname(host))
        except ValueError as exc:
            raise PackageError(f"invalid {label} host: {exc}") from exc
    return sorted(normalized)


def render_task_toml(
    opaque_key: str,
    *,
    reviewer_hosts: list[str] | None = None,
    judge_hosts: list[str] | None = None,
) -> bytes:
    """Render the Harbor schema-1.4 task configuration with an explicit network policy.

    ``reviewer_hosts`` populate ``[agent].allowed_hosts`` (the boundary the
    reviewing agent may reach) and ``judge_hosts`` populate the verifier's
    separate ``[verifier.environment].allowed_hosts`` boundary -- the two
    policies stay independent. Both lists are normalized and sorted before
    rendering, and compilation fails closed (``PackageError``) on a missing,
    empty, or invalid list rather than silently defaulting to a host.

    The ``[environment.env]`` block threads the opaque per-case task key and the
    deterministic ``base``/``head`` ref names into the agent container (Harbor
    natively injects ``[environment].env`` into the environment). No judge/
    credential/archive configuration is ever rendered onto the agent surface.
    """
    reviewer = _normalized_allowed_hosts(reviewer_hosts, "reviewer")
    judge = _normalized_allowed_hosts(judge_hosts, "judge")
    # TOML is intentionally rendered directly: fixed ordering and no timestamp
    # make these bytes part of the deterministic compiled-tree contract.
    # json.dumps renders each list as a double-quoted TOML inline array.
    return f'''schema_version = "1.4"

[metadata]
benchmark_case_key = "{opaque_key}"
source_kind = "historic-github-pr"

[agent]
timeout_sec = 1800.0
network_mode = "allowlist"
allowed_hosts = {json.dumps(reviewer)}

[environment]
network_mode = "no-network"
build_timeout_sec = 1200.0
workdir = "/workspace/repo"
cpus = 2
memory_mb = 4096
storage_mb = 10240

[environment.env]
DAYDREAM_REVIEW_CASE_ID = "{opaque_key}"
DAYDREAM_REVIEW_BASE_REF = "base"
DAYDREAM_REVIEW_HEAD_REF = "head"

[verifier]
timeout_sec = 900.0
environment_mode = "separate"

[verifier.environment]
network_mode = "allowlist"
allowed_hosts = {json.dumps(judge)}
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
