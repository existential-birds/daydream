# Continuous Loop Mode

## Problem

Daydream currently runs a single pass: review, parse, fix, test. If the reviewer finds 10 issues but the fixer only resolves 8 cleanly, those remaining issues require a manual re-run. Additionally, fixes themselves can introduce new review findings that go undetected.

## Solution

Add a `--loop` flag that repeats the full review-parse-fix-test cycle until the reviewer finds zero issues or a maximum iteration cap is reached.

## CLI Changes

New flags:

- `--loop` — enable continuous looping (default: off)
- `--max-iterations N` — cap on iterations (default: 5, only meaningful with `--loop`)

Validation rules:

- `--loop` + `--review-only` is an error (can't fix in a loop without fixing)
- `--loop` + `--start-at` anything other than `"review"` is an error (must run full cycle)
- `--max-iterations` without `--loop` prints a warning but is accepted

## RunConfig Changes

```python
@dataclass
class RunConfig:
    # ... existing fields ...
    loop: bool = False
    max_iterations: int = 5
```

## Runner Logic

When `loop=True`, the existing phases are wrapped in a loop:

```
iteration = 0
total_feedback = 0
total_fixes = 0

while iteration < max_iterations:
    iteration += 1

    # Phase 1: Review
    phase_review(backend, target, skill)

    # Phase 2: Parse
    feedback_items = phase_parse_feedback(backend, target)

    # Zero issues — clean review, done
    if len(feedback_items) == 0:
        break

    total_feedback += len(feedback_items)

    # Phase 3: Fix each item
    for item in feedback_items:
        phase_fix(backend, target, item, ...)
        total_fixes += 1

    # Phase 4: Test
    tests_passed, retries = phase_test_and_heal(backend, target)

    if not tests_passed:
        break  # stop looping on test failure

# After loop exits
if iteration == max_iterations and len(feedback_items) > 0:
    print remaining issues, exit(1)
```

When `loop=False`, behavior is identical to today (single pass, no wrapping).

Key behaviors:

- If tests fail during any iteration, the loop stops immediately
- The test-and-heal phase retains its interactive retry/fix behavior within each iteration
- Each iteration gets a fresh review of the (now-modified) code
- The `.review-output.md` file is overwritten each iteration

## UI Changes

Per-iteration divider (after the first iteration):

```
━━━ Iteration 2 of 5 ━━━
```

Existing phase headers (BREATHE, REFLECT, HEAL, AWAKEN) display as usual within each iteration.

## Summary Changes

`SummaryData` gets new fields:

- `iterations_used: int`
- `loop_mode: bool`

When in loop mode, the summary includes total iterations, total issues found across all iterations, and total fixes applied.

## Exit Behavior

| Scenario | Exit code | Message |
|----------|-----------|---------|
| Zero issues found (clean review) | 0 | "Clean review on iteration N" |
| Max iterations reached, issues remain | 1 | "Reached max iterations, N issues remain" |
| Tests failed during an iteration | 1 | "Tests failed on iteration N" |
| Loop off, normal flow | unchanged | unchanged |

## Files Changed

1. **`cli.py`** — add `--loop` and `--max-iterations` arguments, validation
2. **`runner.py`** — wrap phases in loop, accumulate stats, handle exit conditions
3. **`ui.py`** — iteration divider, updated summary for loop mode
4. **`runner.py` (RunConfig)** — add `loop` and `max_iterations` fields

## Tests

1. `test_loop_exits_on_zero_issues` — issues on iteration 1, zero on iteration 2, exits 0
2. `test_loop_respects_max_iterations` — always returns issues, verify cap honored, exits 1
3. `test_loop_stops_on_test_failure` — tests fail mid-loop, verify immediate stop
4. `test_loop_accumulates_stats` — feedback_count and fixes_applied accumulate across iterations
5. `test_loop_false_single_pass` — loop=False behavior unchanged
6. `test_cli_loop_flags` — --loop and --max-iterations parsed correctly
7. `test_cli_loop_review_only_conflict` — --loop + --review-only raises error
