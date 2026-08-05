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

GENERATED_FILES_PROMPT_RULE = """
Never edit or rewrite an existing generated file. This includes SQL migrations under `migrations/`
(including sqlx, Alembic `versions/`, and Django-style migrations), generated code output (including files
marked `@generated` or `Code generated ... DO NOT EDIT.`), and lockfiles such as `Cargo.lock`,
`package-lock.json`, `pnpm-lock.yaml`, and `uv.lock` where applicable. When a finding legitimately targets
schema behavior, add a NEW migration file such as `migrations/<timestamp>_<name>.sql`; never modify an
existing migration.
""".strip()


def is_generated_file(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    parts = normalized.split("/")
    for pattern in GENERATED_FILE_GLOBS:
        candidates = ("/".join(parts[index:]) for index in range(len(parts)))
        if any(fnmatchcase(candidate, pattern) for candidate in candidates):
            return True
    return False
