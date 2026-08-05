import re
from fnmatch import fnmatchcase

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
    return bool(_GENERATED_MARKER.search(content[:8192]))


def related_manifest_paths(path: str) -> tuple[str, ...]:
    """Return dependency manifests that must stay in sync with *path*'s lockfile."""
    normalized = path.replace("\\", "/").lstrip("/")
    directory, _, filename = normalized.rpartition("/")
    prefix = f"{directory}/" if directory else ""
    return tuple(f"{prefix}{manifest}" for manifest in _LOCKFILE_MANIFESTS.get(filename, ()))
