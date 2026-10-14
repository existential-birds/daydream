# Evaluation Rules

Track claim validity separately from implementation status. Preserve reviewer ordering, IDs, URLs, and snapshot identity throughout verification, fixes, and replies.

| Assessment | Evidence required | Action |
|------------|-------------------|--------|
| VALID | The reported defect holds at the current target; its conditions and impact are established | Fix within authorized scope, or present a concrete proposal when implementation is not authorized |
| INVALID | Evidence disproves the claim for its stated conditions or establishes a false premise | Reject with code/command evidence |
| UNCERTAIN | Evidence is incomplete, contradictory, stale, or cannot establish reachability | Investigate further where possible and state missing evidence |
| NEEDS CLARIFICATION | Ambiguity affects intended behavior or the fix | Ask the specific missing question; continue independent work |

For valid findings, distinguish `implemented and verified`, `changed but validation incomplete`, `blocked`, `outside authorized scope`, and `user-excluded`. An implementation blocker does not invalidate the claim. A proposed patch is not an implemented fix. A skipped test does not establish correct behavior.

An explicit user override changes the approved work set, not the assessment. Retain the assessment and evidence, record the requested change, and track its implementation separately under [Fix-launch decision](SKILL.md#fix-launch-decision).

## Rejection evidence

- A supposedly unused symbol has a live production consumer. Cite it and check whether the claim concerns the whole symbol or an unused branch inside it.
- A reported missing guard exists earlier in the reachable path. Cite the guard and the conditions it covers.
- Reproduction using the reported conditions demonstrates required behavior. A passing test with different conditions does not disprove the bug.
- A suggestion conflicts with a documented Daydream invariant. Cite the invariant and implementation; investigate whether the underlying defect needs another solution.
- The comment targets an older snapshot and the issue is already corrected. Record source and current snapshots; do not invent a new fix.

An empty search, one non-reproducing run, a convention, or an unsupported opinion alone cannot establish invalidity. Lack of evidence also cannot establish validity. Record uncertainty when verification cannot settle the claim.

## Scope and execution order

Keep source numbering stable even when dependencies or severity change execution order. Explain such changes briefly. Whether a defect predates the PR is factual context, not an excuse to ignore authorized work or permission to expand it. Respect the user's scope and exclusions. When all supplied findings are authorized, pursue every confirmed item and honestly record blocked or unresolved work.

Avoid performative agreement. State findings, evidence, decisions, and results plainly.
