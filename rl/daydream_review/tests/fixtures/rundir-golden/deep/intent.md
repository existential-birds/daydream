This diff is a small, self-contained addition to a tiny arithmetic helper module — nothing more.

**What it does:**
- Adds a new function `mean(values: list[int]) -> float` to `calc.py`, alongside the existing `add` and `divide` functions. It computes the arithmetic mean as `sum(values) / len(values)`.
- Updates `tests/test_calc.py` to import `mean` and adds a single test case (`test_mean`) asserting `mean([1, 2, 3]) == 2.0`.

**Nothing else changes:** `add` and `divide` are untouched, and there's no other code in the repo that calls into `calc.py` beyond the test file (per the exploration's dependency scan).

**Notable gap:** `mean` doesn't guard against an empty list — `mean([])` would raise `ZeroDivisionError` — and there's no test covering that edge case, mirroring the fact that `divide` also doesn't guard against division by zero. Given the existing style in the file (no defensive checks on `divide` either), this looks like an intentional, consistent minimalism rather than an oversight, but it's worth flagging since it's the one behavioral edge the new function introduces.