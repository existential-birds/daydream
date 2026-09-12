# Generic-fallback stack review — calc.py, tests/test_calc.py

## Dependency Impact

Per `.daydream/exploration/dependencies.md`, the only dependency edge in scope is
`tests/test_calc.py` → imports/tests → `calc.py`. `calc.py` has no other importers
in the repo (`.daydream/exploration/affected_files.md` lists `tests/__init__.py` as
an unrelated, empty package marker with no dependency on `calc.py`'s content). This
means the blast radius of the new `mean()` function is fully contained: it is
exercised only by `tests/test_calc.py::TestCalc.test_mean`, and no other module
consumes `calc.py`. `.daydream/exploration/conventions.md` has no data collected,
so no convention-name citations are available for this stack; findings below are
graded HIGH only where a specific exploration entry supports them, MEDIUM
otherwise.

## Findings

### 1. `test_mean` cannot distinguish correct float division from a floor-division regression (test-coverage)

**Gate-0 echo** — `tests/test_calc.py:13-14` (read fresh this turn):
```python
    def test_mean(self) -> None:
        self.assertEqual(mean([1, 2, 3]), 2.0)
```
and the implementation under test, `calc.py:14-16`:
```python
def mean(values: list[int]) -> float:
    """Return the arithmetic mean of *values*."""
    return sum(values) / len(values)
```

**Gate 1 (anchor)** — Both symbols are fully self-contained (module-level function
+ single test method); no additional enclosing context needed. File paths and
line ranges are as cited above, from the current versions of `calc.py` and
`tests/test_calc.py`.

**Gate 2 (evidence)** — Ran the following to confirm:
```
>>> sum([1,2,3]) / len([1,2,3])   # correct impl
2.0
>>> sum([1,2,3]) // len([1,2,3])  # hypothetical floor-division bug
2
>>> (sum([1,2,3]) / len([1,2,3])) == (sum([1,2,3]) // len([1,2,3]))
True
```
Because `[1, 2, 3]` averages evenly to `2`, Python's numeric equality
(`int == float`) makes `assertEqual(mean([1, 2, 3]), 2.0)` pass identically
whether `mean` uses `/` (true division, correct) or `//` (floor division, a
plausible off-by-behavior regression). The test input was not chosen to expose
this difference — an input like `[1, 2]` (mean `1.5`) would fail under `//` and
pass only under `/`.

**Gate 3 (severity)** — This is a real gap in the new test added by this diff
(not pre-existing code), so it is not purely informational, but the impact is
limited to weakened regression protection for a 3-line pure function with no
other callers (per the dependency edge above) — Low/Minor severity.

**Summary:** `test_mean`'s only assertion uses an evenly-divisible input, so it
would not catch a regression that swapped `/` for `//` (or any other bug that
happens to floor to the same integer). Recommend adding or replacing with a
case that produces a non-integer mean, e.g. `mean([1, 2]) == 1.5`, to actually
exercise true (float) division.

**Confidence:** HIGH — directly verified by executing both the correct and a
plausible-regression implementation against the exact test input and observing
identical outcomes (Gate 2 evidence above); grounded in the single dependency
edge `tests/test_calc.py` → tests → `calc.py` from `dependencies.md`.

---

### 2. `mean([])` raises `ZeroDivisionError` — unguarded, and untested (informational)

**Gate-0 echo** — `calc.py:14-16` (read fresh this turn):
```python
def mean(values: list[int]) -> float:
    """Return the arithmetic mean of *values*."""
    return sum(values) / len(values)
```

**Gate 1 (anchor)** — Same module, `calc.py:1-17` reviewed in full: `add` (lines
4-6) and `divide` (lines 9-11) are the only other functions, and `divide`
likewise performs `a / b` with no guard against `b == 0`.

**Gate 2 (evidence)** — `.daydream/deep/intent.md` explicitly calls this out:
*"mean doesn't guard against an empty list — mean([]) would raise
ZeroDivisionError — and there's no test covering that edge case, mirroring the
fact that divide also doesn't guard against division by zero... this looks like
an intentional, consistent minimalism rather than an oversight."* This matches
the code as written: `divide` (calc.py:9-11) has the identical unguarded-division
shape as the new `mean` (calc.py:14-16), so the new function is consistent with
the existing style already present in the file rather than introducing a new
pattern.

**Gate 3 (severity)** — Per the review instructions, a request for behavior
(input validation) that did not exist in the code this diff extends is
Informational only, especially since the author's stated intent (per
`intent.md`) treats this as deliberate minimalism consistent with `divide`.

**Summary:** No behavioral change is recommended here — `mean` follows the same
unguarded-division convention already established by `divide` in the same file.
Flagging only for visibility: if `divide`'s lack of a zero-check is ever
revisited, `mean`'s empty-list case should be revisited alongside it, and vice
versa, since they're now the same pattern in two places.

**Confidence:** MEDIUM — consistent with the exploration context (`intent.md`'s
explicit note and the observed symmetry with `divide` in `calc.py:9-11`), but
not tied to a named convention entry (`conventions.md` has no data collected for
this stack).

---

## Non-findings (checked, no issue)

- Import wiring: `tests/test_calc.py:3` correctly adds `mean` to the `from calc
  import add, divide, mean` line; no stale imports or missed references found
  (single grep-equivalent read of the whole diff hunk, both files fully read).
- Type hint `list[int]` on `mean`'s parameter is narrower than what the
  implementation actually supports (any numeric sequence), but this is a
  pre-existing style choice consistent with `add`'s `int`-typed parameters
  (calc.py:4) — not a defect introduced by this diff.
- Docstring style (`"""Return the arithmetic mean of *values*."""`) matches the
  existing docstring conventions for `add` and `divide` in the same file
  (calc.py:5, calc.py:10) — consistent, no issue.
