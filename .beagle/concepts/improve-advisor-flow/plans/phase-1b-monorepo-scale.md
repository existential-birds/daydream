# Improve Advisor Flow — Phase 1b (Monorepo Scale) Implementation Plan

> **Source spec:** `.beagle/concepts/improve-advisor-flow/spec.md` — this plan completes
> the *Phase 1* monorepo requirements (spec.md:29-30, validation case spec.md:148) that
> Phase 1 shipped incompletely. It is **not** the spec's "Phase 2 — Tracker publishing".
> **Source behavior:** `/Users/ka/github/improve/skills/improve/SKILL.md:62` — *"On a
> large monorepo even `deep` scopes subagents to packages, not the root."*
> **For downstream agents:** Execute task-by-task. Each task uses `- [x]` checkboxes.
> Do not skip the test-first steps. The verification gate for **every** task is
> `make check` (Step 4 of each task names the narrower commands to run first).

**Goal:** `daydream improve <path>` audit coverage scales with repository size: fan-out
is `O(partition-groups × categories)` with every agent's surface bounded by a file-count
cap, file lists never inlined into prompts, per-partition-group command recon, fanned-out
vetting, hierarchical aggregation, and a real (data-driven, not prose) "what was not
audited" accounting.

**Why (measured, 2026-07-22, against a real 1892-file Go+React+Python monorepo with 19
discoverable services):**

- `_audit_assignments` (`daydream/improve/orchestrator.py:670-702`) axes fan-out on
  `category × language-stack`. Result: 14 audit agents, **9 of which received all 1892
  tracked files**, because `detect_stacks` appends a `structure` meta-stack carrying the
  union of every file (`daydream/deep/detection.py:282-290`) and no
  `audit:<category>:structure` slot exists, so every category's `remaining_files`
  swallowed the whole repo. A 10× larger repo produces the same 14 agents.
- Each agent's file scope was inlined as a comma-joined path list
  (`orchestrator.py:843-847`): 93,798 chars (~23k tokens) for the largest assignment.
- 19 services were enumerated (`daydream/improve/services.py:31-52`) but used only for
  prompt annotation and `--scope` filtering — never as a fan-out axis.
- The repo's largest tree (`frontend/`, 755 files + `admin-dashboard/`, 104 files) was
  discovered as **zero** services: `_CONVENTIONAL_ROOTS` (`services.py:16`) covers only
  `apps|services|packages|crates|cmd`, root `package.json` has no `workspaces`, and the
  `go.work` signal does not reach non-Go trees.
- `_step_vet` (`orchestrator.py:1080-1096`) is a serial per-category loop that inlines
  **every** candidate finding as JSON into one prompt — the next blowup once audit
  volume scales.
- Effective audit concurrency is `min(tier.max_concurrency=10, backend hint=4)` = 4
  (`daydream/backends/__init__.py:336-341`, `backends/claude.py:323`, `codex.py:78`).

**Architecture:** A new pure-logic partition layer (`daydream/improve/partition.py`)
covers every tracked file with bounded partitions (service-derived where a service
exists, directory-derived elsewhere, residue for root-level loose files), then packs
partitions into stack-homogeneous **partition groups** bounded by the same file cap.
The audit step fans out `group × category`; prompts carry group member roots and counts,
never file lists — agents enumerate their own subtree. Recon adds a per-group command
discovery fan-out merged into the validated command list. Vet fans out over bounded
finding batches. Findings are stamped with their partition, aggregated per group, then
merged cross-partition. A coverage ledger artifact drives the report's "What was not
audited" section and the tier ceilings' skip accounting.

**Tech stack:** unchanged — Python 3.12, anyio, `Backend`/`AgentEvent` seam
(claude/codex/pi) via `run_agent()`, ATIF v1.7 recorder, jsonschema, pytest.

---

## Assumptions (decisions pinned; confirm or correct while executing)

1. **Partition file bound = 400** (`PARTITION_MAX_FILES` in
   `daydream/improve/partition.py`). Operator-overridable via
   `[tool.daydream.improve] partition-max-files`. Rationale: bounds each agent's
   search surface to what one context window handles comfortably; on the measured
   repo it yields ~7-8 groups.
2. **Tier group ceilings** (new `EffortTier.max_partition_groups`): `quick=None`,
   `standard=8`, `deep=None` (unbounded — the skill's `deep` contract is "Whole repo,
   every package", SKILL.md effort table). Operator-overridable via
   `[tool.daydream.improve] max-partition-groups` (applies to `standard` and `deep`).
   When the ceiling binds, the **largest groups by file count are kept** (crude
   hotspot weighting — matches `standard`'s "hotspot-weighted, key packages") and
   every skipped partition is recorded in the coverage ledger and report.
   **`quick` never partitions:** it builds one synthetic whole-repo group (the source
   skill defines `quick` as "recon hotspots only" *across the whole repo*, ≤1
   subagent; the context risk that motivated narrowing is gone once file lists are no
   longer inlined), so `quick` = one agent per quick category over the repo root,
   as today.
3. **Services are never silently split, but oversized services are split with their
   identity kept**: a service whose subtree exceeds the bound is split by
   subdirectory into partitions that all carry `service=<name>`, so findings still
   stamp the service (spec.md:29 "findings and plans name the service(s)").
4. **Groups are stack-homogeneous** (dominant stack per partition, majority by file
   count, ties alphabetical; the `structure` meta-stack is excluded from the
   path→stack map). Skill selection is per group:
   `registry.skill_if_registered(f"audit:{category}:{group.stack}")`, falling back to
   `registry.skill_if_registered(f"audit:{category}")`. This removes the whole-repo
   catch-all entirely — every category iterates every group.
5. **Extension API bump: version 3, hard-breaking** (Kevin's decision). The `audit`
   prompt slot's kwargs change (`services` → `group`), and a new `recon-commands`
   prompt slot is added. Per the documented policy (`docs/extensions.md:107-112`),
   `EXTENSION_API_VERSION` → 3 **and** `MIN_SUPPORTED_EXTENSION_API_VERSION` → 3 in
   the same release (`daydream/extensions/api.py:23-25`).
6. **Per-agent budgets stay deliberately unbounded** for the improve flow — the
   existing decision at `daydream/config.py:236-243` (a truncated audit silently
   reads as complete) is unchanged. Cost is bounded at the **fan-out layer**: agents
   per run = `groups × categories` (audit) + `groups` (command recon, gated) + vet
   batches, with `groups` capped by tier/config. This is the answer to "concurrency
   and cost ceilings at this scale".
7. **Branch focus bypasses partitioning**: one synthetic group over the branch's
   changed files (small by construction), preserving Phase 1 branch-focus behavior
   and its tests. `improve plan <description>` / `review-plan` runs are unaffected
   (they never run the audit step, `orchestrator.py:2392-2396`).
8. **All-assignments-failed is a run failure.** Today a run where every audit agent
   errors falls through to "No vetted defect findings -- done." with exit 0
   (`orchestrator.py:1330`) — a total failure indistinguishable from a clean repo.
   `_step_audit` returns `Stop(1)` when `assignments` is non-empty and every one
   failed.
9. **Vet batch bound = 20 findings per prompt** (`VET_BATCH_MAX_FINDINGS` in
   `daydream/config.py`), batches fanned out in parallel under the same
   `effective_fanout_concurrency(tier.max_concurrency, backend)` limiter as audit.
10. **The repo-survey 500-file sample stays** (`repo_scan`,
    `daydream/exploration_runner.py:311-316`). Post-change it only needs to extract
    repo-global conventions/guidelines; per-tree characterization now happens inside
    the per-group audit and command-recon agents, which enumerate their own subtrees
    directly. Recon decomposition is delivered as the per-group command fan-out
    (Task 5), not as N repo surveys.

## Patterns

**P1 — Improve real-path test harness (reuse).** All flow-behavior tests enter through
`runner.run` with `RunConfig(flow_name="improve", ...)` against a real temp git repo,
patching only `daydream.runner.create_backend` to return `_ImproveStubBackend`
(`tests/test_improve_flow.py:335`, installer `_install_improve_stub` at `:981`). Assert
observable outcomes: exit code, files under `.daydream/improve/` and `daydream_plans/`,
prompt *properties* recorded at the execute seam (e.g. "no audit prompt contains a
joined file list"), trajectory content. Never "function was called".

**P2 — Registered-surface doc drift.** Any task that changes a flow step, skill slot,
prompt name, or prompt kwargs MUST update the matching inventory table in
`docs/extensions.md` in the same task —
`tests/test_extension_contract_doc.py` pins the doc to the registry.

**P3 — Config-bounded scale testing.** Partition splitting and group ceilings are
exercised through the *production config path*: tests write
`[tool.daydream.improve] partition-max-files = N` / `max-partition-groups = M` into the
fixture repo's `pyproject.toml` and re-commit, instead of monkeypatching constants.
Pure-logic scale shapes (thousands of paths, oversized unsplittable directories) are
unit-tested directly against `partition.py` functions.

## File Structure

**Create**

| Path | Responsibility |
|------|----------------|
| `daydream/improve/partition.py` | `Partition`, `PartitionGroup`, `build_partitions`, `group_partitions`, `stack_by_path`, `PARTITION_MAX_FILES` |
| `tests/test_improve_partition.py` | Unit: partition cover, splitting, sibling merge, grouping, ceilings, scale shapes |

**Modify**

| Path | Change |
|------|--------|
| `daydream/config.py` | `EffortTier.max_partition_groups`, `VET_BATCH_MAX_FINDINGS` |
| `daydream/config_file.py` | `[tool.daydream.improve] partition-max-files` / `max-partition-groups` |
| `daydream/improve/orchestrator.py` | recon computes partitions/groups; `_AuditAssignment` re-axis; audit prompt scope; per-group recon-commands fan-out; vet batching; partition stamping; per-group caps; coverage ledger; `Stop(1)` on total audit failure |
| `daydream/improve/prompts.py` | `build_audit_prompt` v3 signature (`group` kwarg), `build_recon_commands_prompt`, `RECON_COMMANDS_ONLY_SCHEMA` |
| `daydream/improve/prioritize.py` | `_finding_services` includes `partition` stamp |
| `daydream/improve/artifacts.py` | `audit_findings_path(directory, category, group_name)`, `coverage_path` |
| `daydream/extensions/api.py` | `EXTENSION_API_VERSION = 3`, `MIN_SUPPORTED_EXTENSION_API_VERSION = 3` |
| `daydream/extensions/builtins.py` | register `recon-commands` prompt |
| `docs/extensions.md` | prompt-kwargs table, changelog v3, floor note |
| `tests/conftest.py` | `improve_scaled_monorepo_target` fixture |
| `tests/test_improve_flow.py` | stub group-awareness; updated expectations; new real-path tests |
| `tests/test_config_file.py` | new improve keys |
| `tests/test_config.py` | tier field pins |
| `CLAUDE.md` | `[tool.daydream.improve]` config keys doc (final task) |

Dependency order: Task 1 → Task 2 → Task 3 → Task 4 → Tasks 5, 6, 7 (independent of
each other, all on 4) → Task 8 (needs 4; reads 5-7 outputs) → Task 9.

---

### Task 1: Partition model and partitioner (pure logic)

**Files:**
- Create: `daydream/improve/partition.py`, `tests/test_improve_partition.py`

- [x] **Step 1: Write the failing tests**

`tests/test_improve_partition.py` (module-level imports:
`from daydream.improve.partition import Partition, PartitionGroup, PARTITION_MAX_FILES, build_partitions, group_partitions, stack_by_path`;
`from daydream.improve.services import Service`; `from pathlib import Path`):

```python
def test_service_files_cover_into_service_partitions() -> None:
    files = ["apps/billing/api.py", "apps/billing/pyproject.toml", "apps/catalog/api.py"]
    services = [Service(name="billing", root=Path("apps/billing"), source="heuristic:conventional-root"),
                Service(name="catalog", root=Path("apps/catalog"), source="heuristic:conventional-root")]
    partitions = build_partitions(files, services)
    assert [(p.name, p.root, p.service, p.files) for p in partitions] == [
        ("billing", "apps/billing", "billing", ("apps/billing/api.py", "apps/billing/pyproject.toml")),
        ("catalog", "apps/catalog", "catalog", ("apps/catalog/api.py",)),
    ]

def test_uncovered_tree_becomes_directory_partitions_and_root_files_residue() -> None:
    files = ["frontend/src/a.tsx", "frontend/src/b.tsx", "README.md"]
    partitions = build_partitions(files, [])
    assert [(p.name, p.root, p.source) for p in partitions] == [
        ("frontend", "frontend", "directory"),
        ("residue", ".", "residue"),
    ]
    assert partitions[1].files == ("README.md",)

def test_oversized_directory_splits_by_subdirectory() -> None:
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c") for i in range(4)]
    partitions = build_partitions(files, [], max_files=6)
    # web has 12 files > 6 -> split at web/*; a+b = 8 > 6 so siblings cannot merge;
    # every child <= 6 stands alone.
    assert [(p.name, p.root, len(p.files)) for p in partitions] == [
        ("web/a", "web/a", 4), ("web/b", "web/b", 4), ("web/c", "web/c", 4),
    ]

def test_small_sibling_directories_merge_up_to_the_bound() -> None:
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c") for i in range(2)]
    partitions = build_partitions(files, [], max_files=6)
    assert [(p.name, p.root, len(p.files)) for p in partitions] == [("web", "web", 6)]

def test_oversized_service_splits_but_keeps_service_stamp() -> None:
    files = [f"apps/big/{sub}/f{i}.py" for sub in ("x", "y") for i in range(4)]
    services = [Service(name="big", root=Path("apps/big"), source="config")]
    partitions = build_partitions(files, services, max_files=6)
    assert all(p.service == "big" for p in partitions)
    assert [p.name for p in partitions] == ["big/x", "big/y"]

def test_unsplittable_flat_directory_stays_oversized() -> None:
    files = [f"flat/f{i}.py" for i in range(10)]
    partitions = build_partitions(files, [], max_files=6)
    assert [(p.root, len(p.files)) for p in partitions] == [("flat", 10)]

def test_partition_cover_is_total_and_disjoint_at_scale() -> None:
    files = sorted(f"pkg{i:03d}/mod{j}/f{k}.go" for i in range(50) for j in range(4) for k in range(5))
    partitions = build_partitions(files, [], max_files=40)
    covered = [f for p in partitions for f in p.files]
    assert sorted(covered) == files and len(covered) == len(files)
    assert all(len(p.files) <= 40 for p in partitions)

def test_stack_by_path_excludes_the_structure_meta_stack() -> None:
    from daydream.deep.detection import detect_stacks
    stacks = detect_stacks(["a.py", "b.go"])
    mapping = stack_by_path(stacks)
    assert mapping == {"a.py": "python", "b.go": "go"}

def test_grouping_is_stack_homogeneous_and_bounded() -> None:
    parts = build_partitions(
        ["apps/s1/a.py", "apps/s2/b.py", "web/c.tsx", "README.md"],
        [Service(name="s1", root=Path("apps/s1"), source="config"),
         Service(name="s2", root=Path("apps/s2"), source="config")],
    )
    stack_of = {"apps/s1/a.py": "python", "apps/s2/b.py": "python", "web/c.tsx": "react"}
    groups, skipped = group_partitions(parts, stack_of, max_files=10, max_groups=None)
    assert skipped == []
    assert [(g.name, g.stack, tuple(p.name for p in g.partitions)) for g in groups] == [
        ("group-01", "generic", ("residue",)),
        ("group-02", "python", ("s1", "s2")),
        ("group-03", "react", ("web",)),
    ]

def test_group_ceiling_keeps_largest_groups_and_reports_the_rest() -> None:
    parts = build_partitions(
        [f"apps/s{i}/f{j}.py" for i in range(3) for j in range(i + 1)], [], max_files=2)
    stack_of = {f: "python" for p in parts for f in p.files}
    groups, skipped = group_partitions(parts, stack_of, max_files=2, max_groups=1)
    assert len(groups) == 1
    assert groups[0].file_count == max(2, groups[0].file_count)  # largest kept
    assert {p.name for p in skipped} and all(isinstance(p, Partition) for p in skipped)

def test_empty_and_single_file_repos() -> None:
    assert build_partitions([], []) == []
    partitions = build_partitions(["main.py"], [])
    assert [(p.root, p.source) for p in partitions] == [(".", "residue")]
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_partition.py -x`
Expected: FAIL — `ModuleNotFoundError: daydream.improve.partition`.

- [x] **Step 3: Implement against the tests**

**Behavior contract** for `daydream/improve/partition.py`:

```python
PARTITION_MAX_FILES: int = 400

@dataclass(frozen=True)
class Partition:
    name: str                    # unique; service name, or root path for splits
    root: str                    # repo-relative POSIX dir; "." only for residue
    source: str                  # "service:<Service.source>" | "directory" | "residue"
    service: str | None          # owning Service.name, kept through oversized splits
    files: tuple[str, ...]       # sorted; host-side only, never serialized to prompts

@dataclass(frozen=True)
class PartitionGroup:
    name: str                    # "group-01", "group-02", ... (filesystem-safe)
    stack: str | None            # dominant stack key, or None for generic/mixed
    partitions: tuple[Partition, ...]

    @property
    def file_count(self) -> int: ...        # sum of member len(files)
    @property
    def roots(self) -> tuple[str, ...]: ... # member roots, sorted

def build_partitions(files: Sequence[str], services: Sequence[Service],
                     *, max_files: int = PARTITION_MAX_FILES) -> list[Partition]: ...
def stack_by_path(stacks: Sequence[StackAssignment]) -> dict[str, str]: ...
def group_partitions(partitions: Sequence[Partition], stack_of: Mapping[str, str],
                     *, max_files: int = PARTITION_MAX_FILES,
                     max_groups: int | None = None,
                     ) -> tuple[list[PartitionGroup], list[Partition]]: ...
```

> **Executed 2026-07-22 — one step deliberately not implemented.** Step 3's
> sibling merge below is incoherent as specified: a parent only splits when it
> exceeds `max_files`, so its children can never all merge back, and any partial
> merge yields a partition whose root overlaps the siblings that stayed behind
> (at the repo root, several partitions all named `.`). The stated goal — a split
> subtree stays with one agent — is delivered instead when groups are packed
> (`partition.py::_pack`, commit `fcd95ad`), measured at 9 → 5 cross-group hops
> on the 1892-file probe repo and pinned by `test_sibling_partitions_stay_in_one_group`.

`build_partitions` algorithm (deterministic, total, disjoint):
1. Sort service roots deepest-first (`-len(root.parts)`, then path); assign each file
   to the first service root that equals or prefixes it (POSIX string prefix with
   trailing `/`, mirroring `_services_for_files` at `orchestrator.py:705-718`).
2. Any service subtree with more than `max_files` files is split by subdirectory
   (recursive, same splitter as step 3) into partitions named
   `<service.name>/<relative-subpath>`, each with `service=<service.name>` and
   `source=f"service:{service.source}"`.
3. Uncovered files: group by top-level directory. For each directory over
   `max_files`, recurse into child directories; files directly inside a split
   directory form their own partition for that directory (root = the directory,
   containing only its direct files). A flat directory that cannot split stays
   oversized (never silently dropped). After splitting, greedily merge *adjacent
   sibling* directory partitions (same parent, sorted by root) while the combined
   size stays ≤ `max_files`; a merged partition's root is the common parent and its
   name is the parent path. Service partitions never merge with directory partitions.
4. Files with no `/` (repo root level) form one `residue` partition
   (`name="residue"`, `root="."`).
5. Return partitions sorted by `root`.

`stack_by_path`: iterate `StackAssignment`s, skip `stack_name == STRUCTURE_STACK_NAME`
(import from `daydream.config`), map each file to its stack (first assignment wins —
assignments are disjoint apart from structure).

`group_partitions`:
1. Dominant stack per partition: `Counter(stack_of.get(f, "generic") for f in files)`
   most common; ties broken alphabetically. Map the `"generic"` dominant to
   `stack=None`.
2. Per stack bucket, pack partitions first-fit-decreasing by `len(files)` (ties by
   name) into bins of capacity `max_files`; a single partition larger than the
   capacity gets its own bin.
3. Order bins by `(stack-key or "~", first member partition name)` and name them
   `group-01`, `group-02`, ... **after** ordering, so names are deterministic.
4. If `max_groups` is not `None` and there are more bins: keep the `max_groups`
   largest by `file_count` (ties by name), re-name the kept bins `group-01..`, and
   return every partition of a dropped bin in the second tuple element (the
   not-audited list), sorted by root.

No I/O in this module. No `Any` in signatures. Match the repo's docstring style
(one-line summary; Args/Returns only where non-obvious).

- [x] **Step 4: Run the new tests AND the gate**

Run: `uv run pytest tests/test_improve_partition.py -x` → PASS. Then `make check` →
green.

- [x] **Step 5: Sweep**

Confirm `partition.py` imports nothing from `orchestrator.py` (no cycles; it may
import `daydream.improve.services.Service`, `daydream.deep.detection.StackAssignment`,
`daydream.config.STRUCTURE_STACK_NAME`).

- [x] **Step 6: Commit**

```bash
git add daydream/improve/partition.py tests/test_improve_partition.py
git commit -m "feat(improve): bounded partition cover and stack-homogeneous grouping"
```

---

### Task 2: Tier and config-file surface for partition bounds

**Files:**
- Modify: `daydream/config.py`, `daydream/config_file.py`
- Test: `tests/test_config.py`, `tests/test_config_file.py`

- [x] **Step 1: Write the failing tests**

In `tests/test_config.py`:

```python
def test_effort_tiers_carry_partition_group_ceilings() -> None:
    assert EFFORT_TIERS["quick"].max_partition_groups is None   # quick never partitions
    assert EFFORT_TIERS["standard"].max_partition_groups == 8
    assert EFFORT_TIERS["deep"].max_partition_groups is None
```

In `tests/test_config_file.py` (follow the file's existing parse-table test pattern
for `[tool.daydream.improve]`, currently covering `service_roots` /
`service_groups` — `daydream/config_file.py:281-282`):

```python
def test_improve_partition_bounds_parse_from_tool_daydream_improve(tmp_path: Path) -> None:
    # pyproject with [tool.daydream.improve] partition-max-files = 7, max-partition-groups = 2
    ...
    assert config.improve_partition_max_files == 7
    assert config.improve_max_partition_groups == 2

def test_improve_partition_bounds_default_to_none(...) -> None: ...
def test_improve_partition_bounds_reject_non_positive(...) -> None:
    # 0 or negative values coerce to None (ignored), matching the file's lenient
    # coercion style (_coerce_* helpers).
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py tests/test_config_file.py -x -k "partition"`
Expected: FAIL — `AttributeError` / `TypeError` (unknown field).

- [x] **Step 3: Implement against the tests**

- `daydream/config.py`: add `max_partition_groups: int | None` to `EffortTier`
  (`config.py:294-302`) and set it in `EFFORT_TIERS` (`:305-327`): quick `None`
  (unused — quick bypasses partitioning, Assumption 2), standard `8`, deep `None`. Add module constant `VET_BATCH_MAX_FINDINGS: int = 20`
  next to `PLAN_WRITE_MAX_CONCURRENCY` (used in Task 6).
- `daydream/config_file.py`: add `improve_partition_max_files: int | None = None` and
  `improve_max_partition_groups: int | None = None` to `DaydreamFileConfig`
  (fields near `improve_service_roots`, `config_file.py:83-84`); parse
  `partition-max-files` and `max-partition-groups` from the same `improve` table the
  existing keys use (`config_file.py:281-282`), coercing non-int / non-positive
  values to `None`. Both the `[tool.daydream.improve]` (pyproject) and top-level
  `[improve]` (`.daydream.toml`) spellings must work — reuse however
  `service_roots` already achieves that; do not build a second lookup path.

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_config.py tests/test_config_file.py -x` → PASS.
`make check` → green.

- [x] **Step 5: Sweep**

`EffortTier` is constructed in tests and in `orchestrator.py:481-483`
(`replace(requested, categories=None, max_concurrency=1)`) — `dataclasses.replace`
carries new fields automatically; grep `EffortTier(` across `daydream tests bench`
and fix any positional constructions.

- [x] **Step 6: Commit**

```bash
git add daydream/config.py daydream/config_file.py tests/test_config.py tests/test_config_file.py
git commit -m "feat(improve): partition-group ceilings in tiers and operator config"
```

---

### Task 3: Extension API v3 — audit prompt re-signature, recon-commands slot

This is the fork-visible breaking change, landed *before* the orchestrator re-axis so
Task 4's prompts compile against the new contract.

**Files:**
- Modify: `daydream/improve/prompts.py`, `daydream/extensions/api.py`,
  `daydream/extensions/builtins.py`, `docs/extensions.md`
- Test: `tests/test_extension_contract_doc.py` (existing pin),
  `tests/test_extensions_loader.py`

- [x] **Step 1: Write the failing tests**

In `tests/test_extensions_loader.py`, follow the existing version-window tests:

```python
def test_v2_extension_no_longer_loads() -> None:
    # DAYDREAM_EXT_API = 2 module -> ExtensionVersionError naming supported range 3..3
def test_v3_extension_loads() -> None: ...
```

Add a prompt-contract unit test (in `tests/test_improve_flow.py` or the prompts unit
test file if one exists — grep `build_audit_prompt` in tests first):

```python
def test_audit_prompt_carries_group_roots_and_no_file_list() -> None:
    prompt = build_audit_prompt(
        category="correctness", skill_invocation=None,
        group={"name": "group-01", "stack": "python", "file_count": 24,
               "partitions": [{"name": "billing", "root": "apps/billing",
                                "file_count": 12, "service": "billing"}]},
        scope_note="", recon_summary="{}", cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "apps/billing" in prompt and "group-01" in prompt
    assert "Relevant tracked files:" not in prompt

def test_recon_commands_prompt_names_group_roots() -> None: ...
```

- [x] **Step 2: Run to verify failure**

`uv run pytest tests/test_extensions_loader.py -x -k "v2 or v3"` → FAIL (v2 loads).
The prompt test fails with `TypeError: unexpected keyword argument 'group'`.

- [x] **Step 3: Implement against the tests**

- `daydream/improve/prompts.py`:
  - `build_audit_prompt` signature becomes keyword-only
    `(*, category, skill_invocation, group, scope_note, recon_summary, cwd, tier)`.
    `group` is a plain dict:
    `{"name": str, "stack": str | None, "file_count": int, "partitions": [{"name", "root", "file_count", "service"}]}`.
    Drop the `services` kwarg and `_service_block`'s "Service slices" section;
    replace with a "Partition group" block rendering one line per member partition
    (`- <name> — `<root>/` (<file_count> files[, service <service>])`) and the
    instruction: *"Enumerate this group's files yourself (e.g.
    `git ls-files -- '<root>/**'` or your Glob tool scoped to the roots above); the
    host never inlines file lists."* Keep the existing search-bounds-vs-read-bounds
    sentence (`prompts.py:1360-1361`), tier instruction, playbook section, finding
    format, Hard Rules 4 and 6, and schema block unchanged.
  - Add `RECON_COMMANDS_ONLY_SCHEMA` — `{"type": "object", "additionalProperties":
    False, "required": ["commands"], "properties": {"commands": {"type": "array",
    "items": RECON_COMMAND_SCHEMA}}}` (import `RECON_COMMAND_SCHEMA` from
    `daydream.improve.command_contract`).
  - Add `build_recon_commands_prompt(*, group, recon_summary, cwd)` — a read-only
    command-discovery prompt: render the same partition-group block; instruct the
    agent to find build/test/lint commands for these subtrees only, with
    `working_directory` set per command (the contract's `WORKING_DIRECTORY_SCHEMA`,
    `command_contract.py:36-41`) and `in-scope-paths` applicability naming the
    partition roots; require the same evidence provenance wording as the
    `IMPROVE_RECON` prompt's command bullet (`orchestrator.py:427-438`) — copy that
    bullet verbatim into this builder rather than re-deriving it. End with
    `_schema_block(RECON_COMMANDS_ONLY_SCHEMA)`.
  - Export both from `__all__`.
- `daydream/extensions/api.py`: `EXTENSION_API_VERSION = 3`;
  `MIN_SUPPORTED_EXTENSION_API_VERSION = 3`.
- `daydream/extensions/builtins.py`: in `_register_improve_builtins`
  (`builtins.py:34-46`), add
  `registry.override_prompt("recon-commands", prompts.build_recon_commands_prompt)`.
- `docs/extensions.md`:
  - Prompt-kwargs table (`docs/extensions.md:346`): `audit` row becomes
    `category`, `skill_invocation`, `group`, `scope_note`, `recon_summary`, `cwd`,
    `tier`; add a `recon-commands` row (`group`, `recon_summary`, `cwd`).
  - Changelog: add **Version 3** entry — audit prompt re-signature (`services` →
    `group`; file lists no longer inlined), new `recon-commands` prompt slot, floor
    raised to 3 (hard-breaking per the documented policy at `docs/extensions.md:107-112`).
  - Update the `DAYDREAM_EXT_API` literal mentions from 2 to 3 wherever the doc
    states the current value (grep `DAYDREAM_EXT_API = 2` and `1..2`).

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_extensions_loader.py tests/test_extension_contract_doc.py -x`
→ PASS. `make check` will FAIL at this point only if `orchestrator.py` still calls the
old audit-prompt signature — it does (`orchestrator.py:872-880`). To keep this task
independently green, update that one call site minimally: build a **single whole-repo
group dict** `{"name": "group-01", "stack": assignment.stack, "file_count":
len(assignment.files), "partitions": [{"name": assignment.stack or "repository",
"root": ".", "file_count": len(assignment.files), "service": None}]}` and stop
appending the file list to `scope_note` (delete the `"Relevant tracked files: " +
", ".join(...)` construction at `orchestrator.py:842-847`, keep the stack/remaining
sentence). Task 4 replaces this shim with the real group axis. Update any
`test_improve_flow.py` assertions that match the removed "Relevant tracked files"
text (grep it). Then `make check` → green.

- [x] **Step 5: Sweep**

Grep `build_audit_prompt`, `prompt("audit")`, `services=` across `daydream tests
bench` for stragglers. Confirm `CLAUDE.md`'s extensions line still says the API
version generically (it names `EXTENSION_API_VERSION (currently 2)` — update to 3).

- [x] **Step 6: Commit**

```bash
git add daydream/improve/prompts.py daydream/extensions/api.py daydream/extensions/builtins.py \
        docs/extensions.md daydream/improve/orchestrator.py tests/ CLAUDE.md
git commit -m "feat(improve)!: extension API v3 — partition-group audit prompt, recon-commands slot"
```

---

### Task 4: Re-axis the audit fan-out on partition groups

**Files:**
- Modify: `daydream/improve/orchestrator.py`, `daydream/improve/artifacts.py`
- Test: `tests/conftest.py` (`improve_scaled_monorepo_target`),
  `tests/test_improve_flow.py`

- [x] **Step 1: Write the failing tests**

`tests/conftest.py` — new fixture, modeled on `improve_monorepo_target`
(`conftest.py:149-172`, reuse `_init_repo`/`_git`/`_commit`):

```python
@pytest.fixture
def improve_scaled_monorepo_target(tmp_path: Path) -> Path:
    """Committed monorepo large enough that partition fan-out must split."""
    project = tmp_path / "improve_scaled"
    for index in range(12):                       # 12 conventional-root services
        root = project / "apps" / f"svc{index:02d}"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f'[project]\nname = "svc{index:02d}"\n')
        (root / "api.py").write_text(f'def service_name():\n    return "svc{index:02d}"\n')
    for sub in ("alpha", "beta", "gamma"):        # uncovered react tree, no service signal
        pkg = project / "frontend" / "src" / sub
        pkg.mkdir(parents=True)
        for index in range(4):
            (pkg / f"view{index}.tsx").write_text("export const V = () => <div/>;\n")
    (project / "README.md").write_text("# scaled\n")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "improve-scaled"\n\n[tool.daydream]\ntest-command = "uv run pytest"\n'
    )
    _init_repo(project); _git(project, "add", "."); _commit(project, "initial")
    return project
```

`tests/test_improve_flow.py` — new real-path tests (all through `runner.run` with the
stub installed per P1):

```python
async def test_audit_fans_out_per_partition_group_on_scaled_monorepo(
        improve_scaled_monorepo_target: Path, monkeypatch) -> None:
    # defaults (bound 400): partitions = 12 services + frontend + residue;
    # stack-homogeneous grouping -> exactly 3 groups:
    #   group with 12 python service partitions, group with frontend (react),
    #   group with residue (generic).  Assert:
    audit_calls = [c for c in stub.calls if c["marker"] == "audit"]
    assert len(audit_calls) == 3 * len(AUDIT_CATEGORIES)          # 27
    assert all("Relevant tracked files" not in c["prompt"] for c in audit_calls)
    assert all(".py" not in ...)  # no individual file paths: assert no prompt
    # contains "apps/svc00/api.py" (a known tracked file) — roots only.
    # Artifacts: one findings file per category x group:
    names = sorted(p.name for p in (target / ".daydream/improve").glob("audit-*-findings.json"))
    assert len(names) == 27 and "audit-correctness-group-01-findings.json" in names

async def test_partition_bound_splits_oversized_trees_via_config(
        improve_scaled_monorepo_target: Path, monkeypatch) -> None:
    # Append "[tool.daydream.improve]\npartition-max-files = 5\n" to pyproject,
    # re-commit, run. frontend/src (12 files) must split into 3 partitions and
    # the python services must pack into ceil(24/5)-sized groups; assert the
    # audit call count grew accordingly and every group block in every audit
    # prompt lists roots whose summed file_count <= 5 (parse the prompt block).

async def test_group_ceiling_skips_smallest_groups_and_reports_them(
        improve_scaled_monorepo_target: Path, monkeypatch) -> None:
    # Append "[tool.daydream.improve]\nmax-partition-groups = 1\n", re-commit, run.
    # Only the largest group (the 24-file python group) is audited:
    assert len({c for c in stub.calls if c["marker"] == "audit"}) == len(AUDIT_CATEGORIES)
    report = (target / ".daydream/improve/report.md").read_text()
    assert "frontend" in report_not_audited_section(report)       # skipped partitions named

async def test_quick_tier_audits_whole_repo_in_one_group(
        improve_scaled_monorepo_target: Path, monkeypatch) -> None:
    # --effort quick -> exactly 3 audit calls (quick categories), each prompt's
    # partition-group block lists the single root "." and no partition split;
    # coverage.json not_audited == [].

async def test_all_audit_assignments_failing_exits_nonzero(
        improve_monorepo_target: Path, monkeypatch) -> None:
    # stub.fail_categories = all nine -> every assignment fails -> exit code 1,
    # report not written as "clean".

async def test_small_repo_collapses_to_bounded_groups(tmp_git_repo, ...) -> None:
    # single-package python repo -> 1 residue partition -> 1 group ->
    # exactly len(AUDIT_CATEGORIES) audit agents, run completes exit 0.
```

Also update the existing scaled expectations: grep `test_improve_flow.py` for
assertions on audit call counts, `audit-` artifact filenames
(`audit_findings_path` shape changed), and prompt fragments
(`"Relevant tracked files"`, `"Cover the remaining repository surface"`); recompute
each against the new axis (on `improve_monorepo_target`: partitions = billing,
catalog, web, residue; groups = python(billing+catalog), react(web),
generic(residue) → **3 groups × 9 categories = 27 audit calls**, artifacts
`audit-<category>-group-0N-findings.json`).

Stub changes (`_ImproveStubBackend.execute`, `tests/test_improve_flow.py:383-465`):
extract the group from the prompt
(`re.search(r"Partition group `(group-\d+)`", prompt)`) and the first member root
(first `` `<root>/` `` line of the block); emit findings whose `path`/`evidence`
cite `<first-root>/api.py:1` so partition/service stamping resolves. Keep every
existing knob (`fail_categories`, `n_findings`, credential injection) working.

- [x] **Step 2: Run to verify failure**

`uv run pytest tests/test_improve_flow.py -x -k "partition or scaled or ceiling or collapses or failing"`
Expected: FAIL — fixture missing, then call-count mismatches.

- [x] **Step 3: Implement against the tests**

**`daydream/improve/artifacts.py`:**
- `audit_findings_path(improve_dir_path: Path, category: str, group_name: str) -> Path`
  → `f"audit-{category}-{group_name}-findings.json"` (drop the `stack | None`
  parameter; group names are always present and filesystem-safe).
- Add `coverage_path(improve_dir_path: Path) -> Path` → `"coverage.json"`
  (written in this task, extended in Task 8).

**`daydream/improve/orchestrator.py`:**

- `_step_recon` (`:450-667`), after `detect_stacks` (`:656-662`):
  ```python
  file_config = ctx.config.file_config or DaydreamFileConfig()
  max_files = file_config.improve_partition_max_files or PARTITION_MAX_FILES
  tier: EffortTier = ctx.data["effort_tier"]
  max_groups = file_config.improve_max_partition_groups or tier.max_partition_groups
  ```
  - `--effort quick` (audit run, not branch focus): skip partitioning — one
    synthetic `Partition(name="repository", root=".", source="quick",
    service=None, files=tuple(all_files))` in one
    `PartitionGroup(name="group-01", stack=<dominant repo-wide stack via
    stack_by_path>, partitions=(...,))`; `skipped = []`.
  - Audit runs (`_is_audit_run` true, not branch focus, not quick): `all_files` = the same list
    passed to `detect_stacks` (scoped by `_stacks_for_services` when
    `improve_scope` is set — build partitions from the scoped services' file union
    in that case); `partitions = build_partitions(all_files, services,
    max_files=max_files)`; `groups, skipped = group_partitions(partitions,
    stack_by_path(stacks), max_files=max_files, max_groups=max_groups)`.
  - Branch focus: one synthetic
    `Partition(name="branch", root=".", source="branch", service=None,
    files=tuple(branch_files))` in one `PartitionGroup(name="group-01",
    stack=<dominant via stack_by_path>, partitions=(...,))`; `skipped = []`.
  - Store `ctx.data["partitions"]`, `ctx.data["partition_groups"]`,
    `ctx.data["partitions_not_audited"]` (the skipped list). Write
    `coverage_path(directory)` with
    `{"artifact_provenance": ..., "partitions": [...], "groups": [{"name", "stack",
    "file_count", "partitions": [names]}], "not_audited": [{"partition", "root",
    "reason": "group-ceiling"}]}` via `_with_artifact_provenance(...,
    phase=DaydreamPhase.RECON)`. Description mode (`improve_plan_description`)
    skips all of this (empty lists), matching its existing services skip (`:487-494`).

- `_AuditAssignment` (`:388-401`) becomes:
  ```python
  @dataclass(frozen=True)
  class _AuditAssignment:
      category: str
      group: PartitionGroup
      skill: str | None

      @property
      def key(self) -> str:
          return f"{self.category}:{self.group.name}"
  ```
- `_audit_assignments(ctx, categories, groups)` (replaces `:670-702`): for each
  category × group, `skill = ctx.registry.skill_if_registered(
  f"audit:{category}:{group.stack}") if group.stack else None`, falling back to
  `ctx.registry.skill_if_registered(f"audit:{category}")`. No catch-all bucket, no
  `remaining_files`.
- `_step_audit` (`:816-994`):
  - Build the prompt with `group=` dict (name/stack/file_count/partitions with
    name/root/file_count/service from the group's `Partition`s). `scope_note`
    carries only the focus/scope/branch extras that exist today (`:849-871`) — the
    file-list construction (`:842-847`) is fully deleted (Task 3 left a shim).
  - `scoped_services` for the prompt drops out (the group block carries service
    names).
  - Fork descriptors: `f"audit-{current.category}-{current.group.name}"`.
  - Findings persistence: `audit_findings_path(directory, assignment.category,
    assignment.group.name)`.
  - Stamping: pass `ctx.data["partitions"]` into `_stamp_finding` and set
    `stamped["partition"]` = name of the first partition whose root equals or
    prefixes the first evidence path (`None` when nothing matches). Keep the
    existing `services` stamp exactly as is (`:768-777`).
  - After the task group: if `assignments and len(failures) == len(assignments)`:
    `print_error(console, "Improve audit failed", "every audit assignment failed")`
    and `return Stop(1)` (change return annotation to `Stop | None`; the engine
    accepts it — `flows/engine.py:98-103`).
  - Per-group cap: apply `tier.max_findings` per group (order each group's findings
    by leverage, truncate, count drops) **then** the existing global cap
    (`:974-978`) so `quick` (one group) is unchanged; sum `dropped_by_cap`.
- `_render_report`/`_step_report`: unchanged in this task beyond compiling
  (Task 8 rebuilds the not-audited section); `failed` keys now read
  `<category>:<group-name>` — acceptable interim rendering.

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_improve_flow.py tests/test_improve_partition.py -x` → PASS.
Then the full flow-adjacent set:
`uv run pytest tests/test_improve_flow.py tests/test_improve_budgets.py tests/test_improve_services.py -x`
→ PASS. Then `make check` → green.

- [x] **Step 5: Sweep**

Grep for the deleted symbols: `remaining_files`, `"Relevant tracked files"`,
`audit_findings_path(` (all call sites updated to group names, including any test
helpers), `_services_for_files(` (still used by branch-focus service narrowing at
`:507` — keep that one). Re-read `_step_audit` top to bottom for orphaned locals.

- [x] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py daydream/improve/artifacts.py tests/
git commit -m "feat(improve): partition-group audit fan-out with bounded per-agent scope"
```

---

### Task 5: Per-group recon command discovery

**Files:**
- Modify: `daydream/improve/orchestrator.py`
- Test: `tests/test_improve_flow.py`

- [x] **Step 1: Write the failing tests**

```python
async def test_recon_commands_fan_out_per_group_and_merge(
        improve_scaled_monorepo_target: Path, monkeypatch) -> None:
    # Stub answers the recon-commands marker ("discover build, test, and lint
    # commands" sentinel from build_recon_commands_prompt) with one valid
    # command record per group (working_directory = the group's first root).
    # Assert: one recon-commands call per partition group (3 on this fixture);
    # recon.json "commands" contains the merged records with ids prefixed
    # "group-01-", "group-02-", ...; invalid records from one group are rejected
    # with diagnostics while other groups' records survive.

async def test_single_group_repo_skips_per_group_recon(tmp_git_repo, ...) -> None:
    # 1 group -> no recon-commands calls; recon behaves exactly as before.

async def test_quick_and_branch_and_description_skip_per_group_recon(...) -> None:
    # --effort quick, --focus branch, and `improve plan "..."` each produce zero
    # recon-commands calls.
```

Stub: add a `recon-commands` marker branch keyed on a stable sentence from
`build_recon_commands_prompt`, returning `{"commands": [...]}` shaped like
`_stub_recon_commands()` entries with `working_directory` set from the prompt's
first root.

- [x] **Step 2: Run to verify failure**

`uv run pytest tests/test_improve_flow.py -x -k "recon_commands or skips_per_group"`
→ FAIL (zero recon-commands calls everywhere).

- [x] **Step 3: Implement against the tests**

In `_step_recon`, after groups are computed and after the existing repo-level recon
agent returns, and only when `len(groups) > 1` and the run is a full audit
(not quick, not branch focus, not description mode):

- Fan out one agent per group under
  `anyio.CapacityLimiter(effective_fanout_concurrency(tier.max_concurrency, backend))`
  with `maybe_fork(recorder, f"recon-commands-{group.name}")`, prompt
  `ctx.registry.prompt("recon-commands")(group=<group dict>,
  recon_summary=json.dumps(recon_data, sort_keys=True), cwd=target)`,
  `output_schema=RECON_COMMANDS_ONLY_SCHEMA`, `read_only=True`,
  `persist_session=False`, phase `DaydreamPhase.RECON`. A failed group agent is
  recorded (redacted, like audit failures) and skipped — never fatal.
- Merge: for each group's structured output, `_redact_model_value` it, prefix every
  candidate command's `id` with `f"{group.name}-"` (the prefixed id still matches
  the contract's id pattern), then run the *merged* candidate list through
  `validate_recon_commands({..., "commands": merged}, repo=target)` exactly once so
  cross-group id collisions and semantic checks fail closed. Append surviving
  records to `recon_data["commands"]` and rejected codes to
  `recon_data["command_rejections"]`; extend the command-validation diagnostics
  artifact with the same envelope builder already used
  (`_command_validation_diagnostics`, `:257-341`).
- `recorder.create_dispatch_step(phase=DaydreamPhase.RECON)` after the fan-out.

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_improve_flow.py -x` → PASS. `make check` → green.

- [x] **Step 5: Sweep**

Confirm plan-writer inputs still validate: `_verification_commands` (`:1384-1392`)
and the plan writer's recon-command menu (`prompts.py:1411-1435`) render the
prefixed ids untouched; `assemble_plan` command-ref matching is by id string and
needs no change (verify by reading `daydream/improve/assemble.py`'s ref-resolution
before closing the task; if it re-derives ids, fix there, not downstream).

- [x] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py tests/test_improve_flow.py
git commit -m "feat(improve): per-partition-group recon command discovery"
```

---

### Task 6: Vet fan-out with bounded batches

**Files:**
- Modify: `daydream/improve/orchestrator.py`
- Test: `tests/test_improve_flow.py`

- [x] **Step 1: Write the failing tests**

```python
async def test_vet_batches_are_bounded_and_parallel(
        improve_monorepo_target: Path, monkeypatch) -> None:
    # Stub returns 45 correctness findings (extend the stub finding factory to
    # emit n per category via a new knob).  With VET_BATCH_MAX_FINDINGS = 20:
    vet_calls = [c for c in stub.calls if c["marker"] == "vet"]
    assert len(vet_calls) == 3                      # 45 -> 20 + 20 + 5
    for call in vet_calls:
        payload = json.loads(call["prompt"].split("```json\n")[1].split("```")[0])
        assert len(payload) <= VET_BATCH_MAX_FINDINGS
    # Verdicts across batches all apply: vetted-findings.json contains every
    # kept finding; a rejected title in batch 3 lands in rejected.json.

async def test_vet_batch_failure_fails_closed_per_batch(...) -> None:
    # One batch's agent raises -> only that batch's findings drop; other
    # batches' keeps survive (assert on vetted-findings.json contents).
```

- [x] **Step 2: Run to verify failure**

`uv run pytest tests/test_improve_flow.py -x -k "vet_batch"` → FAIL (one call gets
all 45).

- [x] **Step 3: Implement against the tests**

Rework `_step_vet` (`:1056-1142`): keep the per-category grouping and
previously-rejected filtering; chunk each category's candidates into slices of
`VET_BATCH_MAX_FINDINGS` (import from `daydream.config`); run all batches in one
`anyio.create_task_group()` under
`CapacityLimiter(effective_fanout_concurrency(tier.max_concurrency, backend))`
(read `tier` from `ctx.data["effort_tier"]`), each inside
`maybe_fork(recorder, f"vet-{category}-{batch_index:02d}")` mirroring the audit
task shape (`:888-931`) with `phase=DaydreamPhase.VET`. `vet_id` stays the 1-based
index *within the batch* (prompt text at `prompts.py:1402` already says "1-based
array index" — still true per batch). Each batch applies `_apply_vet_verdicts` on
its own slice (fail-closed per batch: an exception yields `output = {}` for that
batch only). Collect `kept`/`rejected` thread-safely by writing into
pre-sized per-batch result slots (the audit step's `results` dict pattern), then
flatten in deterministic batch order. Call
`recorder.create_dispatch_step(phase=DaydreamPhase.VET)` after the group; drop the
now-redundant `phase_scope(DaydreamPhase.VET)` wrapper (fork descriptors carry the
phase, as in audit). Branch-focus diff suffix (`:1089-1096`) is appended per batch
prompt unchanged.

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_improve_flow.py -x` → PASS. `make check` → green.

- [x] **Step 5: Sweep**

Re-read `_step_vet` end-to-end; confirm rejected-finding recording
(`record_rejections`) still receives every batch's rejections exactly once.

- [x] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py daydream/config.py tests/test_improve_flow.py
git commit -m "feat(improve): parallel bounded-batch vetting"
```

---

### Task 7: Partition-aware aggregation and top offenders

**Files:**
- Modify: `daydream/improve/prioritize.py`
- Test: `tests/test_improve_prioritize.py`, `tests/test_improve_flow.py`

- [x] **Step 1: Write the failing tests**

`tests/test_improve_prioritize.py`:

```python
def test_cross_partition_findings_merge_like_cross_service() -> None:
    # Two findings, same category, similar titles, empty "services", distinct
    # "partition" values ("frontend/src/alpha" vs "frontend/src/beta") -> one
    # aggregated finding whose body lists both locations.

def test_same_partition_findings_never_merge() -> None: ...
def test_service_and_partition_stamps_compose() -> None:
    # services=["billing"] + partition="billing" on one, services=["catalog"] +
    # partition="catalog" on the other -> merge (disjoint), unchanged from today.
```

`tests/test_improve_flow.py`: extend an existing full-pipeline test to assert the
report's "Top offenders" section names a directory partition when the stub's
findings cite the uncovered frontend tree (partition stamp reaches
`_top_offender_lines` through the services/partition union).

- [x] **Step 2: Run to verify failure**

`uv run pytest tests/test_improve_prioritize.py -x -k partition` → FAIL (no merge:
`_finding_services` ignores `partition`).

- [x] **Step 3: Implement against the tests**

`daydream/improve/prioritize.py`:
- `_finding_services` (`prioritize.py:101-105`): return
  `{*services, partition}` where `partition = finding.get("partition")` when it is
  a non-empty `str`. This makes `aggregate_cross_service` (`:55-98`) merge
  same-pattern findings across directory partitions (disjoint sets) while the
  overlap guard keeps same-partition findings unmerged — no other logic change.
- `_merge_cross_service_group` / `_cross_service_locations` (`:108-154`): also
  carry partitions — `merged["partitions"] = _ordered_unique(...)` from each
  finding's `partition`, and the location lines fall back to the partition name
  when a finding has no services.
- `orchestrator._top_offender_lines` (`orchestrator.py:2288-2306`): include
  `finding.get("partition")` in the per-finding key set alongside services
  (dedup so a service-named partition is not double counted).

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_improve_prioritize.py tests/test_improve_flow.py -x` →
PASS. `make check` → green.

- [x] **Step 5: Sweep**

Confirm the vetted-findings artifact round-trips the new `partition` /
`partitions` keys (they are plain dict passthrough — verify one full-pipeline test
asserts the key survives into `vetted-findings.json`).

- [x] **Step 6: Commit**

```bash
git add daydream/improve/prioritize.py daydream/improve/orchestrator.py tests/
git commit -m "feat(improve): aggregate findings across partitions, partition top offenders"
```

---

### Task 8: Coverage ledger drives "What was not audited"

**Files:**
- Modify: `daydream/improve/orchestrator.py`
- Test: `tests/test_improve_flow.py`

- [x] **Step 1: Write the failing tests**

```python
async def test_report_names_unaudited_partitions_and_failed_groups(
        improve_scaled_monorepo_target: Path, monkeypatch) -> None:
    # max-partition-groups = 1 via config (P3) AND stub.fail_categories = {"docs"}.
    report = (target / ".daydream/improve/report.md").read_text()
    section = report.split("## What was not audited")[1].split("## ")[0]
    # every skipped partition named with its root and reason:
    assert "frontend" in section and "group-ceiling" in section
    # failed assignment maps to its group's partition roots, not just a key:
    assert "docs" in section or "docs" in report.split("### Failed audit assignments")[1]
    coverage = json.loads((target / ".daydream/improve/coverage.json").read_text())
    assert {e["reason"] for e in coverage["not_audited"]} == {"group-ceiling"}
    assert coverage["groups"] and coverage["partitions"]

async def test_clean_full_coverage_reports_nothing_skipped(
        improve_monorepo_target: Path, monkeypatch) -> None:
    # No ceiling, no failures -> coverage.not_audited == [] and the report's
    # not-audited section contains the tier statement plus "All N partitions
    # were audited" (exact wording pinned here).
```

- [x] **Step 2: Run to verify failure**

`uv run pytest tests/test_improve_flow.py -x -k "not_audited or nothing_skipped"` →
FAIL (static prose only).

- [x] **Step 3: Implement against the tests**

- `_step_audit`: after the fan-out, rewrite `coverage.json` (first written in
  recon, Task 4) adding `"failed_assignments": {key: reason}` and per-group
  `"status"`: `"audited"` / `"failed"` (all-failed already `Stop(1)`s).
- `_render_report` (`:2151-2255`): add parameters
  `partitions_not_audited: list[Partition]` and `groups: list[PartitionGroup]`
  (threaded from `ctx.data` in `_step_report`, `:2354-2374`). The
  "What was not audited" section becomes: the existing tier sentence
  (`tier_bound`, `:2188-2201`), the scope statement (unchanged), then a
  data-driven block — one line per not-audited partition
  (`- **<name>** — `<root>/` (<file_count> files; group-ceiling)`), or the exact
  sentence `All <len(partitions)> partitions were audited.` when the list is
  empty. "Failed audit assignments" lines render
  `- **<category> / <group-name>** (<comma-joined roots>) — <reason>` by
  resolving the group name from `groups`.

- [x] **Step 4: Run the new tests AND the gate**

`uv run pytest tests/test_improve_flow.py -x` → PASS. `make check` → green.

- [x] **Step 5: Sweep**

Re-read `_render_report` for stale wording that now contradicts the ledger (the
`deep` tier sentence "Coverage included every detected package" is only true when
`not_audited` is empty — make the tier sentence for `deep`/`standard` defer to the
ledger block instead of asserting coverage).

- [x] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py tests/test_improve_flow.py
git commit -m "feat(improve): data-driven not-audited accounting in the improve report"
```

---

### Task 9: Parity, docs, and fresh-eyes sweep

**Files:**
- Modify: `CLAUDE.md`, `README.md` (only if it documents improve config),
  `docs/extensions.md` (consistency re-read)
- Test: full suite

- [x] **Step 1: Small-repo parity pins**

Add (if Task 4 did not already) the real-path regression
`test_small_repo_collapses_to_bounded_groups` plus one pin that branch focus still
runs serially with unchanged assignment shape
(`--focus branch` on `improve_branch_target` → exactly `len(AUDIT_CATEGORIES)`
audit calls, one group, provenance tagging intact — recompute against the existing
branch tests and update their expectations rather than duplicating them).

- [x] **Step 2: Docs**

- `CLAUDE.md` "Improve runtime controls" block: document
  `partition-max-files` and `max-partition-groups` under the existing
  `[tool.daydream.improve]` example; update the extensions line to
  `EXTENSION_API_VERSION (currently 3)`.
- Re-read `docs/extensions.md` end-to-end for any remaining `v2`/`1..2` window
  text (the loader error string at `extensions/loader.py:75` renders the range —
  confirm tests cover `3..3`).

- [x] **Step 3: Fresh-eyes pass**

Run `daydream improve` against this repository itself
(`uv run daydream improve --non-interactive .` in a scratch clone or with
`--effort quick`) and read `.daydream/improve/report.md` cold: partition names
must be meaningful, the not-audited section must be truthful, and no prompt-sized
artifact should contain a file list. Record anything confusing and fix it in this
task, not as a follow-up.

- [x] **Step 4: Gate**

`make check` → green. Then re-run the measured probe from this plan's "Why"
section against `/Users/ka/github/shelfspace-app/shelfspace-mono` (read-only,
pure-logic — no agents): partitions must cover all 1892 files, `frontend/` must
split or group under the bound, and default-standard fan-out must be
`groups × 9` with every group ≤ 400 files. Paste the numbers into the PR
description.

- [x] **Step 5: Commit**

```bash
git add CLAUDE.md README.md docs/extensions.md tests/
git commit -m "docs(improve): partition config surface, API v3 notes, parity pins"
```

---

## Backward compatibility

- **Backends (claude/codex/pi):** untouched — every new fan-out goes through
  `run_agent()` (`daydream/agent.py:382-`) exactly like the existing audit fan-out;
  no backend-protocol change.
- **ATIF trajectories:** additive — new fork descriptors
  (`audit-<category>-<group>`, `vet-<category>-<batch>`,
  `recon-commands-<group>`) and unchanged `DaydreamPhase` values; the recorder
  contract is untouched.
- **Extension registry:** hard-breaking v3 (decided): `audit` prompt kwargs change
  and the floor rises to 3, so v1/v2 `daydream_ext` packages stop loading with a
  loud `ExtensionVersionError` until they declare `DAYDREAM_EXT_API = 3` (and, if
  they override `audit`, adopt the `group` kwarg). Documented in the
  `docs/extensions.md` changelog (Task 3). `vet` and `plan-writer` prompt
  contracts are unchanged.
- **Artifacts:** `audit-*-findings.json` filenames change from
  `audit-<category>[-<stack>]-` to `audit-<category>-<group>-`; `coverage.json` is
  new; `recon.json` gains group-prefixed command ids. These are per-run
  `.daydream/improve/` artifacts with no cross-run readers in-tree (verified:
  `load_rejections`/`planned_fingerprints` read only `daydream_plans/`,
  `plans.py:66-73,1539-1543`).
- **`quick` tier** keeps its whole-repo scope (one synthetic root group,
  Assumption 2) — no coverage change; it simply runs through the same group
  machinery as the other tiers.

## Cost and concurrency ceilings (stated per the handoff's ask)

Per-agent wall/tool budgets remain intentionally unset for improve
(`daydream/config.py:236-243`). Scale is bounded at the fan-out layer:

| Tier | Groups | Audit agents | Recon-commands agents | Concurrency |
|------|--------|--------------|----------------------|-------------|
| quick | 1 (whole repo, never partitioned) | 3 (categories) | 0 | 1 |
| standard | ≤ 8 (config-overridable) | ≤ 72 | ≤ 8 | `min(10, backend hint 4)` |
| deep | unbounded (spec: every package) | groups × 9 | groups | `min(10, backend hint 4)` |

Vet adds `ceil(findings / 20)` agents. A 2000-package repo on `standard` runs at
most 72 + 8 + vet batches; on `deep` the operator caps it with
`max-partition-groups` or accepts linear wall-clock at concurrency 4 — and either
way every skipped partition is named in the report. No silent truncation anywhere:
ceiling skips, per-group caps, and failed groups all land in `coverage.json` and
the report.
