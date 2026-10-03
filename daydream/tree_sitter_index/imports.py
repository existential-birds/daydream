"""Diff paths and direct import impact, including bounded reverse lookups."""

from __future__ import annotations

import ast
from pathlib import Path

from daydream import git_ops
from daydream._tree_sitter_safety import assert_tree_sitter_safe
from daydream.exploration import FileInfo
from daydream.git_ops import GitError
from daydream.hunk_index import iter_diff_blocks

from .runtime import (
    LANGUAGES,
    _def_query_for_language,
    _query_for_language,
    extract_definitions,
    extract_imports,
    get_parser,
)


def _module_candidates(base: Path, dotted: str) -> list[Path]:
    """Resolve a dotted module name under `base` to .py and package candidates."""
    target = base
    for part in dotted.split("."):
        target = target / part
    return [target.with_suffix(".py"), target / "__init__.py"]


def _resolve_python_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    candidates: list[Path] = []
    if import_str.startswith("from "):
        # Relative component of a `from` statement; parse it to read the
        # ImportFrom level (ascent) and retained aliases (bare-relative names).
        # Best-effort: malformed/unsupported captures degrade to no candidates.
        try:
            body = ast.parse(import_str).body
        except SyntaxError:
            return []
        if len(body) != 1 or not isinstance(body[0], ast.ImportFrom):
            return []
        node = body[0]
        if node.level < 1:
            return []
        base = importer.parent
        for _ in range(node.level - 1):
            base = base.parent
        if node.module is not None:
            candidates.extend(_module_candidates(base, node.module))
        else:
            candidates.append(base / "__init__.py")
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidates.extend(_module_candidates(base, alias.name))
    else:
        parts = import_str.split(".")
        candidates.extend(_module_candidates(repo_root, import_str))
        # Also try resolving the parent (e.g. `from foo.bar import baz`).
        if len(parts) >= 2:
            candidates.extend(_module_candidates(repo_root, ".".join(parts[:-1])))
    return [c for c in candidates if c.exists() and c.is_file()]


def _resolve_ts_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    if not import_str.startswith("."):
        return []
    base = importer.parent / import_str
    suffixes = [".ts", ".tsx", ".d.ts", ".js", ".jsx"]
    candidates: list[Path] = [base.with_suffix(s) for s in suffixes]
    candidates.extend([base / f"index{s}" for s in suffixes])
    return [c for c in candidates if c.exists() and c.is_file()]


def _build_go_package_index(repo_root: Path) -> dict[str, tuple[Path, ...]]:
    """Index Go files in one traversal; the first directory with each name wins.

    An OSError anywhere in the traversal discards the index."""
    index: dict[str, list[Path]] = {}
    owners: dict[str, Path] = {}
    try:
        for candidate in repo_root.rglob("*.go"):
            if not candidate.is_file():
                continue
            parent = candidate.parent
            name = parent.name
            if not name:
                continue
            owner = owners.get(name)
            if owner is None:
                owners[name] = parent
                index[name] = [candidate]
            elif owner == parent:
                index[name].append(candidate)
    except OSError:
        return {}
    return {name: tuple(files) for name, files in index.items()}


def _resolve_go_import(import_str: str, package_index: dict[str, tuple[Path, ...]]) -> list[Path]:
    # Best-effort: look up the import's terminal component in the indexed tree.
    if not import_str:
        return []
    suffix = import_str.strip("/").split("/")[-1]
    return list(package_index.get(suffix, ()))


def _resolve_rust_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    if import_str.startswith("std::") or "::" not in import_str:
        return []
    if import_str.startswith("crate::"):
        rest = import_str[len("crate::") :]
    else:
        rest = import_str
    parts = rest.split("::")
    # Drop the trailing item name (often a type/fn) and try the module path.
    module_parts = parts[:-1] if len(parts) > 1 else parts
    if not module_parts:
        return []
    src = repo_root / "src"
    target = src
    for part in module_parts:
        target = target / part
    candidates = [target.with_suffix(".rs"), target / "mod.rs"]
    return [c for c in candidates if c.exists() and c.is_file()]


def _resolve_import(
    language_id: str,
    import_str: str,
    repo_root: Path,
    importer: Path,
    go_package_index: dict[str, tuple[Path, ...]] | None = None,
) -> list[Path]:
    if language_id == "python":
        return _resolve_python_import(import_str, repo_root, importer)
    if language_id in ("typescript", "tsx", "javascript"):
        return _resolve_ts_import(import_str, repo_root, importer)
    if language_id == "go":
        return _resolve_go_import(import_str, go_package_index or {})
    if language_id == "rust":
        return _resolve_rust_import(import_str, repo_root, importer)
    return []




# Bare generic names create noisy reverse edges; symbol-defining files are
# exempt. Forward import edges remain available for all names.
_GENERIC_STEMS = frozenset(
    {
        "__init__", "mod", "index", "main", "app", "api", "base", "common",
        "utils", "util", "helpers", "constants", "config", "settings",
        "models", "model", "types", "type", "schema", "schemas", "client",
        "server", "service", "services", "handler", "handlers", "views",
        "urls", "test", "tests", "conftest", "setup",
    }
)

# Reverse-edge grep is a best-effort seed, not an exhaustive index. A
# legitimately widely-imported module can match hundreds of files; cap the
# seed so no single module can blow the downstream prompt's context window.
def _eligible_for_reverse_grep(path: str, defining_paths: set[str]) -> bool:
    """Skip invalid stems and generic names unless the file defines a symbol."""
    stem = Path(path).stem
    if not stem or "\x00" in stem or "\r" in stem or "\n" in stem:
        return False
    if stem in _GENERIC_STEMS and path not in defining_paths:
        return False
    return True


_MAX_IMPORTERS = 40

# Restrict the reverse-edge grep to source files. A doc, plan, or config file
# cannot import a code module, so matches in them are always false positives.
_CODE_PATHSPECS: tuple[str, ...] = tuple(f"*{suffix}" for suffix in LANGUAGES)


def _build_importer_lookup(
    repo_root: Path,
    modified_paths: list[str],
    defining_paths: set[str] | None = None,
) -> dict[str, list[str]]:
    """Batch source-only reverse lookups, with a separate 40-result cap per path.

    Defining paths may use generic stems. Each path excludes itself; paths sharing
    one stem retain separate caps. Git failures return empty lists."""
    defining_paths = defining_paths or set()
    lookup: dict[str, list[str]] = {path: [] for path in modified_paths}
    unique_stems: list[str] = []
    seen_stems: set[str] = set()
    for path in modified_paths:
        if not _eligible_for_reverse_grep(path, defining_paths):
            continue
        stem = Path(path).stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        unique_stems.append(stem)
    if not unique_stems:
        return lookup

    try:
        pairs = git_ops.grep_fixed_matches(
            repo_root, unique_stems, word=True, pathspecs=_CODE_PATHSPECS
        )
    except GitError:
        return lookup

    by_stem: dict[str, list[str]] = {}
    for path, matched in pairs:
        if matched not in seen_stems:
            continue
        bucket = by_stem.setdefault(matched, [])
        if path not in bucket:
            bucket.append(path)
    for path in modified_paths:
        if not _eligible_for_reverse_grep(path, defining_paths):
            continue
        stem = Path(path).stem
        lookup[path] = [p for p in by_stem.get(stem, ()) if p != path][:_MAX_IMPORTERS]
    return lookup


def detect_affected_files(
    diff_text: str,
    repo_root: Path,
) -> list[FileInfo]:
    """Return changed paths and their direct imports/importers, deduped by role.

    Deleted files remain modified entries. Known-bad parser versions raise before
    any native construction."""
    # Keep unsafe-version errors outside the fail-open native parsing boundary.
    assert_tree_sitter_safe()

    results: list[FileInfo] = []
    seen: set[tuple[str, str]] = set()

    def _add(path: str, role: str) -> None:
        key = (path, role)
        if key in seen:
            return
        seen.add(key)
        results.append(FileInfo(path=path, role=role))

    entries = [
        (path, any(line.startswith("deleted file mode") for line in block.split("\n@@", 1)[0].splitlines()))
        for path, block in iter_diff_blocks(diff_text)
    ]
    reverse_paths = [
        path for path, deleted in entries if not deleted and Path(path).suffix in LANGUAGES
    ]
    # Collect symbol exemptions during the forward pass to avoid rereading files.
    defining_paths: set[str] = set()
    go_package_index: dict[str, tuple[Path, ...]] | None = None

    for path, deleted in entries:
        _add(path, "modified")

        if deleted:
            continue

        suffix = Path(path).suffix
        lang_entry = LANGUAGES.get(suffix)
        if lang_entry is None:
            continue
        language_id, _factory = lang_entry

        abs_path = repo_root / path
        try:
            source = abs_path.read_bytes()
        except (FileNotFoundError, OSError):
            continue

        parser = get_parser(language_id)
        if parser is None:
            continue

        def_query = _def_query_for_language(language_id)
        if def_query is not None and extract_definitions(parser, source, def_query):
            defining_paths.add(path)

        query_string = _query_for_language(language_id)
        if query_string is None:
            continue

        imports = extract_imports(parser, source, query_string)
        if language_id == "go" and imports and go_package_index is None:
            go_package_index = _build_go_package_index(repo_root)
        for imp in imports:
            resolved_paths = _resolve_import(language_id, imp, repo_root, abs_path, go_package_index)
            for resolved in resolved_paths:
                try:
                    rel = resolved.resolve().relative_to(repo_root.resolve())
                except (ValueError, OSError):
                    continue
                _add(str(rel), "imports")

    # Reverse edges need the complete defining set, so the batched grep runs
    # after the forward pass (whose reads already covered every file once).
    importers_by_path = _build_importer_lookup(repo_root, reverse_paths, defining_paths)
    for path, importers in importers_by_path.items():
        for importer in importers:
            _add(importer, "imported_by")

    return results
