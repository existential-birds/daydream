# Service-enumeration heuristics spike

## Scope and fixtures

The probe used only Python 3 stdlib modules (`pathlib`, `glob`, `json`,
`tomllib`, and `re`). It ran against:

- `daydream`: `/Users/ka/github/existential-birds/daydream-improve-phase1`,
  a single-package Python repository with only a root `pyproject.toml`.
- `improve`: `/Users/ka/github/improve`, a single-package skill repository
  with none of the candidate root manifests.
- `apps-fastapi`: a constructed root project with
  `apps/{billing,catalog}/pyproject.toml`, UV and Poetry-shaped workspace
  tables, and Compose build contexts for both apps.
- `mixed`: a constructed root JavaScript app with `packages/{js-lib,rust-kit,
  go-kit}`, package and pnpm workspace globs, Cargo and Go workspace
  declarations, and a matching child manifest in each package.

Shared command:

```console
python3 /tmp/daydream-service-heuristics/probe.py <repo-root>
```

Each snippet below resolves matched directories, makes them relative to the
repository root, sorts them, and prints the resulting JSON array.

## Per-signal results

| Signal | Probe snippet | `daydream` | `improve` | `apps-fastapi` | `mixed` |
|---|---|---|---|---|---|
| `package.json` `workspaces` | `json.loads(package_json)["workspaces"]`, supporting both the array and `{packages: [...]}` forms, then `glob.glob` | `[]` | `[]` | `[]` | `["packages/go-kit", "packages/js-lib", "packages/rust-kit"]` |
| `pnpm-workspace.yaml` `packages` | narrow line scan for the top-level `packages:` string list, then `glob.glob` | `[]` | `[]` | `[]` | `["packages/go-kit", "packages/js-lib", "packages/rust-kit"]` |
| Cargo workspace `members` | `tomllib.loads(Cargo.toml)["workspace"]["members"]`, then `glob.glob` | `[]` | `[]` | `[]` | `["packages/rust-kit"]` |
| `go.work` `use` | line scan for single-line `use PATH` and `use (...)` entries, then `glob.glob` | `[]` | `[]` | `[]` | `["packages/go-kit"]` |
| UV / Poetry workspace tables | `tomllib.loads(pyproject)["tool"][tool]["workspace"]["members"]`, where `tool` is `uv` or `poetry`, then `glob.glob` | `uv=[]; poetry=[]` | `uv=[]; poetry=[]` | `uv=["apps/billing", "apps/catalog"]; poetry=["apps/billing", "apps/catalog"]` | `uv=[]; poetry=[]` |
| Compose service build contexts | narrow line scan for scalar `build: PATH` and nested `build: {context: PATH}` blocks in the four conventional Compose filenames, then `glob.glob` | `[]` | `[]` | `["apps/billing", "apps/catalog"]` | `[]` |
| Manifest under a conventional root | for each direct child of `apps/`, `services/`, `packages/`, `crates/`, or `cmd/`, test for `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, or `mix.exs` | `[]` | `[]` | `["apps/billing", "apps/catalog"]` | `["packages/go-kit", "packages/js-lib", "packages/rust-kit"]` |

Both required single-package repositories returned an empty array for every
signal, so the probe produced no false positive there.

## Phase-1 decision

Phase 1 should implement this bounded inventory:

1. `package.json` `workspaces` (array and `packages` object forms).
2. `pnpm-workspace.yaml` top-level `packages` globs, accepting only its narrow
   string-list grammar rather than attempting general YAML parsing.
3. Cargo `[workspace].members`.
4. `go.work` `use` entries, both single-line and block forms.
5. Direct children with their own manifest under `apps/`, `services/`,
   `packages/`, `crates/`, or `cmd/`.

All five use stdlib-only reads and produced zero false positives in the two
single-package probes. Workspace results should be de-duplicated with the
conventional-root scan, and only existing directories should be returned.

Rejected for Phase 1:

- **Combined UV / Poetry workspace-table inference:** UV has a defined
  workspace table, but Poetry does not provide the same interoperable
  workspace contract; treating the two as one generic `pyproject.toml`
  convention would encode a synthetic table shape. A dedicated UV signal can
  be reconsidered independently when a real UV monorepo is available.
- **Compose build contexts:** Compose YAML is structurally rich (including
  mappings, extensions, and aliases), so a stdlib line parser is not reliable;
  build contexts can also describe images or infrastructure rather than
  repository services.

The surviving set is non-empty and reliable within the stated probe boundary.
It is the heuristic inventory for Task 3; config-declared roots remain the
guaranteed path.
