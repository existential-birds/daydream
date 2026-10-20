# Coverage ratchet

Run `make test` (also part of `make check`) for full branch coverage and
`coverage.xml`. Focused pytest runs deliberately have no coverage floor.
The authoritative floor and exclusions are in [pyproject.toml](../pyproject.toml).
Vendored ATIF is excluded because it cannot receive local hand fixes.

To raise the floor, measure the complete suite on current main with `make test`,
round the observed percentage down to a whole percent, and raise `fail_under`
only when that value increases. Record SHA, date, environment, and raw percentage
in the commit. Never lower the floor to make a change pass.

## CI aggregation

[CI](../.github/workflows/ci.yml) and [shard tooling](../scripts/ci_shard.py)
own scheduling. Full collection owns selection; timing profiles are hints.
Aggregation requires every expected shard, matching run/attempt and checkout
identities, complete execution, successful exits, and valid branch coverage.
Missing, duplicate, corrupt, or cancelled evidence fails the gate. Diagnostic
coverage from failed executions cannot authorize a merge or reusable proof.
Use **Re-run all jobs**: receipts from different attempts cannot be combined.
Only a successful complete merge-group gate can publish reusable tree proof.
