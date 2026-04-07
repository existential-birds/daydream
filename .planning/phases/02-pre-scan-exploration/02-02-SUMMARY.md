---
phase: 02-pre-scan-exploration
plan: 02
subsystem: tree-sitter-index
tags: [tree-sitter, static-analysis, exploration]
requires: [02-01]
provides: [detect_affected_files, LANGUAGES, get_parser, extract_imports]
affects: [daydream/tree_sitter_index.py, tests/test_tree_sitter_index.py]
tech_added: [tree-sitter Query API (QueryCursor)]
patterns: [lazy language registry, parser cache by language id, graceful degradation on parse errors]
key_files_created:
  - daydream/tree_sitter_index.py
key_files_modified:
  - tests/test_tree_sitter_index.py
decisions:
  - "Reverse-edge importer search uses git grep on file stem (best-effort, skipped for __init__/mod/index)"
  - "Go import resolution is best-effort rglob for a directory whose suffix matches the import path"
  - "depth != 1 raises NotImplementedError so callers cannot silently get partial multi-hop results"
  - "extract_imports() catches all exceptions and returns [] (D-06 graceful degradation)"
metrics:
  duration: ~12min
  tasks: 2
  files: 2
  tests_added: 7
  tests_total: 168
completed: 2026-04-07
---

# Phase 02 Plan 02: Tree-sitter Impact Surface Summary

Built `detect_affected_files()` -- a pure synchronous function that turns a git diff into the impact surface (changed files + 1-hop imports/importers) for Python, TypeScript/TSX/JavaScript, Go, and Rust using tree-sitter Query.

## What Was Built

- `daydream/tree_sitter_index.py` (416 lines)
  - Lazy factory functions for 5 grammars (`_python_lang`, `_typescript_lang`, `_tsx_lang`, `_go_lang`, `_rust_lang`)
  - `LANGUAGES` registry mapping 7 file extensions to `(language_id, factory)` tuples
  - Module-level `_PARSER_CACHE` and `get_parser()` (identity-cached per language id)
  - Four query constants: `PYTHON_IMPORT_QUERY`, `TYPESCRIPT_IMPORT_QUERY`, `GO_IMPORT_QUERY`, `RUST_IMPORT_QUERY`
  - `extract_imports()` helper using `tree_sitter.Query` + `QueryCursor` (modern py-tree-sitter API)
  - `_parse_diff_name_status()` -- tolerant parser that scans `diff --git` headers and `new file`/`deleted file`/`rename to` markers
  - Per-language import resolvers (`_resolve_python_import`, `_resolve_ts_import`, `_resolve_go_import`, `_resolve_rust_import`)
  - `_find_importers()` -- best-effort `git grep` reverse-edge lookup (with `# noqa: S603` justification)
  - Public `detect_affected_files(diff_text, repo_root, depth=1) -> list[FileInfo]`
  - `__main__` smoke entrypoint for manual verification

- `tests/test_tree_sitter_index.py` (7 tests, all passing)
  - Removed `importorskip` and all `xfail` markers
  - Added `_materialize()` helper to write fixture source files into `tmp_path` so resolution can find them on disk
  - Multi-file diff coverage for python/typescript/go/rust
  - Default-depth-is-1, unsupported-language fallback, deleted-file safety

## Verification

- `uv run pytest tests/test_tree_sitter_index.py -v` -> 7 passed
- `make check` -> 168 passed, ruff clean, mypy clean
- Manual smoke: `uv run python -m daydream.tree_sitter_index` extracts imports correctly from all four languages

## Requirements Satisfied

- **EXPL-01** (impact surface for changed files) -- modified files always returned
- **EXPL-02** (1-hop imports/importers) -- direct imports + git-grep reverse edges
- **D-06** (graceful degradation) -- unsupported languages get `role="modified"` with no exception; deleted files don't raise FileNotFoundError; parse errors return empty import lists

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mypy `Argument 1 to "Query" has incompatible type "Language | None"`**
- **Found during:** Task 1 verification
- **Issue:** `parser.language` is typed as `Language | None`, mypy rejected passing it directly to `Query()`
- **Fix:** Added explicit `None` check before constructing `Query`
- **Files modified:** `daydream/tree_sitter_index.py`
- **Commit:** dd5d25c

### Notes on API choice

- The plan referenced `Query(parser.language, ...)` (older py-tree-sitter API). Confirmed via probe that py-tree-sitter ships `tree_sitter.Query` + `tree_sitter.QueryCursor` and `cursor.captures(node)` returns `dict[str, list[Node]]`. Implementation uses this modern shape.
- Tree-sitter `Node.sexp()` referenced in plan task 1 read_first does not exist on this version; used `str(node)` in probe scripts.

## Commits

- `dd5d25c` feat(02-02): add tree-sitter LANGUAGES registry and import extractor
- `81532d3` test(02-02): unmask tree_sitter_index tests with real fixtures

## Self-Check: PASSED

- FOUND: daydream/tree_sitter_index.py
- FOUND: tests/test_tree_sitter_index.py (modified)
- FOUND: dd5d25c
- FOUND: 81532d3
- All 7 plan tests pass; full suite (168) passes; mypy + ruff clean
