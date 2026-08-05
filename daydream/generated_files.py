import re
from fnmatch import fnmatchcase
from pathlib import Path

GENERATED_FILE_GLOBS: tuple[str, ...] = (
    "migrations/*.sql",
    "*/migrations/*.sql",
    "alembic/versions/*.py",
    "migrations/*.py",
    "*/migrations/*.py",
    "*_generated.go",
    "*_generated.py",
    "*.pb.go",
    "*.pb.py",
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "go.sum",
)

_GENERATED_MARKER = re.compile(r"@generated|Code generated.*DO NOT EDIT\.", re.IGNORECASE)
_GENERATED_MARKER_HEADER_LINES = 20
_HEADER_COMMENT_PREFIXES = ("#", "//", "/*", "*", "--", ";")
_LOCKFILE_MANIFESTS: dict[str, tuple[str, ...]] = {
    "Cargo.lock": ("Cargo.toml",),
    "package-lock.json": ("package.json",),
    "pnpm-lock.yaml": ("package.json",),
    "yarn.lock": ("package.json",),
    "uv.lock": ("pyproject.toml",),
    "poetry.lock": ("pyproject.toml",),
    "go.sum": ("go.mod",),
}

GENERATED_FILES_PROMPT_RULE = """
Never edit or rewrite an existing generated file. This includes SQL migrations under `migrations/`
(including sqlx, Alembic `versions/`, and Django-style migrations), generated code output (including files
marked `@generated` or `Code generated ... DO NOT EDIT.`), and lockfiles such as `Cargo.lock`,
`package-lock.json`, `pnpm-lock.yaml`, and `uv.lock` where applicable. When a finding legitimately targets
schema behavior, add a NEW migration file such as `migrations/<timestamp>_<name>.sql`; never modify an
existing migration. Do not add, remove, or change dependencies in package manifests, because those changes
require a synchronized lockfile update.
""".strip()


def is_generated_file(path: str, content: str | bytes | None = None) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    parts = normalized.split("/")
    for pattern in GENERATED_FILE_GLOBS:
        candidates = ("/".join(parts[index:]) for index in range(len(parts)))
        if any(fnmatchcase(candidate, pattern) for candidate in candidates):
            return True
    if content is None:
        return False
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    header_lines: list[str] = []
    for line in content.splitlines()[:_GENERATED_MARKER_HEADER_LINES]:
        stripped = line.strip()
        if not stripped:
            header_lines.append(line)
        elif stripped.startswith(_HEADER_COMMENT_PREFIXES):
            header_lines.append(line)
        elif _GENERATED_MARKER.fullmatch(stripped):
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines)
    return bool(_GENERATED_MARKER.search(header))


def related_manifest_paths(path: str) -> tuple[str, ...]:
    """Return dependency manifests that must stay in sync with *path*'s lockfile."""
    normalized = path.replace("\\", "/").lstrip("/")
    directory, _, filename = normalized.rpartition("/")
    prefix = f"{directory}/" if directory else ""
    return tuple(f"{prefix}{manifest}" for manifest in _LOCKFILE_MANIFESTS.get(filename, ()))


def _snapshot_untracked_generated_files(repo: Path, paths: set[str]) -> dict[str, bytes]:
    """Capture byte baselines for generated files already untracked at fix start."""
    snapshot: dict[str, bytes] = {}
    for path in sorted(paths):
        content = (repo / path).read_bytes()
        if is_generated_file(path, content):
            snapshot[path] = content
    return snapshot


def _changed_untracked_generated_files(repo: Path, snapshot: dict[str, bytes]) -> list[str]:
    """Return snapshotted untracked generated paths whose bytes changed or disappeared."""
    changed: list[str] = []
    for path, baseline in snapshot.items():
        try:
            current = (repo / path).read_bytes()
        except FileNotFoundError:
            changed.append(path)
            continue
        if current != baseline:
            changed.append(path)
    return changed


def _restore_untracked_generated_file(repo: Path, path: str, baseline: bytes) -> None:
    """Write an untracked generated file's exact pre-fix bytes back to disk."""
    file_path = repo / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(baseline)
