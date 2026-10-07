# Fresh production-owner redundancy sweep

Pinned baseline: `f7494ba50b512b4449dd13563031057590c9d030`.
PR #1497 is open at that exact verified head; this branch `test-audit/owner-redundancy` is isolated and stacks on `test-audit/flow-redundancy`. Current merged main is `6c48d368d529a89748c93e7bb61ae9ac6566936e`. Prior improvements and all settings are preserved. Revalidated 2026-10-07.

The audit examined every declaration/parameter row across the requested owner files and the associated Hub run consumer. [Inventory](inventory.csv) marks all 1,259 baseline cases /815 declarations in 66 collected files. Files without collected declarations remain supporting sources; all complete test bodies, parameter tables, fixture chains, owners/callers, siblings, prior ledgers and relevant history were read. Retained entries are conservative where exact replacement proof is absent. Real-path tests remain primary; helpers supplement them.

| Lane | Baseline cases | Functions | Disposition |
| --- | ---: | ---: | --- |
| Training/adjudication/corpus/storage and Hub run consumer | 423 | 291 | 9 functions/cases retired; retain storage/CLI/fault/lineage/race proofs |
| Backend/agent/contract | 478 | 281 | 16 functions /24 cases retired; retain native transports, cancellation, security and lifecycle |
| Diagram/parser | 263 | 175 | 6 functions /17 cases retired; retain schemas, grounding, reexports, executable grammar compatibility and real diagrams |
| Config/extensions | 95 | 68 | All retained; prior partial sweep already removed matching duplicates |

Exact per-node inputs, regressions, keepers, assertion transfers, callers, history, risks and validation: [training map](training-map.md), [backend map](backend-map.md), [diagram/parser map](diagram-map.md). Those plans were reported before source edits and independently reviewed. Independent final source reviews against the pinned full baseline: [training](final-training-review.md), [backend](final-backend-review.md), [diagram/parser](final-diagram-review.md). Source SHA256 pins reflect final whitespace/comment cleanup. No gaps remain.

## Counts and independent scenarios

Full collection is **8,182 → 8,132 cases** and **5,346 → 5,315 distinct functions**: 50 cases and 31 functions removed. No new node or parametrization loop hides scenarios. Of the 50 retired rows, 28 transfer assertions and 22 have existing proof or only implementation inventory/identity checks. There is no wrapper aggregation like the preceding sweep's 721 source-policy rows.

**Independent behavioral scenarios retired: zero.** Fresh/stale cache states, default/read-only native streams, Pi read-only execution, all transport fixtures, report/diagram ordering, immutability, exact wrappers and both metric/recording layers remain executed. The queue keeper now tests accepted/rejected/ambiguous/unanswered outputs through the real projector/queue, replacing internal singleton identity pins and preserving all original predicate inputs. The parser subset inventories never detected member removal; actual literal classification tests remain, with the differing TSX terminal input limitation recorded explicitly. An exact global count of independent semantic scenarios is not inferred from case/function counts: existing parameter rows and loops can carry multiple or duplicate contracts.

Removed in-file test support: the unused branch-probe inventory and `_node_types` helper plus now-unused test imports. No production, public Backend/create_backend/extension API, shared harness, fixture asset, lockfile, workflow, coverage or scheduler changes. Public disposition aliases remain exported; CandidateRoot/DiagramThresholds reexport test remains. There is no performance repair bundled with pruning.

## Baseline, environment and comparable runtime

Python 3.12.13, uv0.11.29, macOS26.6.1 arm64, same machine and locked 179-package resolution/all extras. `uv.lock` SHA256 `e035f70c23015967fb6d69d5f09dac7d77081e8cf943e60e0c41bd58680da5e1`. Root and RL lockchecks passed before their uv commands. All Git hooks remain enabled. Full local `-n auto` selects16 workers; focused comparisons explicitly use4 on both sides. CI remains Blacksmith4vCPU with auto4. Ten-case dispatch chunks, source/exclusions/branch coverage and86% floor are unchanged.

Pinned full baseline `make test`: **8,168 passed /14 skipped /233 warnings**, coverage **90.54%**, pytest352.83s /wall353.98s. It ran in a separate untouched baseline checkout. All in-scope cases therefore have a baseline result; no retained baseline failure was deleted.

Identical79 owner/shared-consumer selections, Python/deps/machine, n4, no coverage on either: baseline **1,452 passed /4 skipped /10 warnings**, pytest82.33s /wall82.96s; candidate **1,402 passed /4 skipped /10 warnings**, pytest95.83s /wall96.29s. **Observed regression +13.33s (+16.07%)** in this single pair. Common-node JUnit increases are led by unchanged native Claude finalization (+1.935s), exported finalization telemetry (+1.738s), then real archive/runner cases. This does not establish a product performance cause or a sustained effect; no speedup is claimed from fewer nodes. Full-run comparison follows below.

Revalidated parent hosted run [37592103107](https://github.com/existential-birds/daydream/actions/runs/37592103107), exact baseline head: check **412s (6m52s)**, test step **373s**, RL succeeds separately. Four-minute check objective remains unmet by **172s** on that measured CI head. A stacked PR targeting the parent's branch does not match this repository's CI pull_request main/master filter; hosted check proof for the candidate requires parent merge/retargeting.

Concrete next action: after #1497 merges, rebase/retarget this reviewed batch onto merged main and run the unchanged4vCPU CI gate. Profile its JUnit setup/call/teardown costs and repeated real runner/archive/native-process journeys; evaluate one separately reviewed owner/harness performance repair against identical4-worker coverage runs. Fifty fewer cases cannot establish or predict a four-minute gate.

## Mutation and preservation proof

[Thirty-one actual-owner controls](mutation-results.json) record exact replacement bytes, nodes, failure reason and SHA256 restoration. Each runs in a separate control worktree, fails behaviorally (not collection/import), restores exact owner bytes, verifies the hash, then reruns its keeper successfully. All production files in that checkout match the pinned baseline after controls. Controls cover projected-input routing, contested outcome labels, adjudication order, fresh cache reuse, ratio, rejected/ambiguous queue classification, persisted labels, Codex legacy/extraction/provenance/display/xcrun/metrics and totals, Pi native readonly stream, Osprey native/session provenance, Claude foreground env/malformed guard/wording, diagram wrapper/order/immutability/report and Python branch/terminal classification.

Control harness corrections are explicit: one same-length source mutation initially left stale timestamp/size-based CPython bytecode after exact source restoration; deleting only the isolated owner's compiled cache before mutation/restoration and rerunning proved the contract. A malformed-Bash draft mutation tested None rather than the decoder's actual empty string and changed no behavior; corrected to empty-string allowance, it fails the registered guard keeper and passes after restoration. A guard with no behavioral effect is not counted as caught proof. An Osprey replacement was narrowed to the actual TurnEnd context before execution. No control or compiled cache change enters this PR.

Validation completed: root/RL lockchecks, collection delta exactly50/31, focused79file owners/consumers, additional **269 passed** lifecycle/transport/read-only/display/artifact/trajectory/shared-harness tests, Ruff and diffcheck. The initial additional selection referenced nonexistent `tests/test_harness.py` and ran no cases; corrected to `tests/harness/` before proof. The full gate passed; ordinary signed commit and push hooks remain in progress and will be recorded below.

## Final full gate and size accounting

`make check` passed: root lockcheck/all extras, Ruff, root/RL Vulture, mypy723files, **8,118 passed /14 skipped /232 warnings**, **90.54% integrated branch coverage**, workflow actionlint (Docker available; passed), coverage.xml and naming check. Pytest419.44s; complete gate wall436.95s. Coverage source/exclusions/86% floor and all workers are unchanged. Root pytest includes root training tests; standalone RL tests were not run because no RL files changed (its lockcheck and dead-code scan did pass).

The identical full pytest invocation regressed **352.83s →419.44s (+66.61s /18.88%)** with the same16workers/Python/dependencies/coverage/machine. Baseline wall353.98s belongs to make test, while candidate wall436.95s includes the other make check gates, so those two whole-command wall times are not presented as an identical-command comparison. No runtime benefit is established. The hook will run its required gate again; that measurement will be reported separately in the PR.

Tracked Python size by consistent categories: production105,189→105,189lines; test modules (excluding harness/fixtures)117,472→117,258; shared harness/fixtures6,955→6,955. Test diff **+84/-298 (net-214)** across17files; production0, shared support0, configuration/workflows/lockfiles0. In-file unused parser helper/inventory deletion is counted in tests, not counted twice as shared support. Evidence documentation is separate.

Focused skips on both sides: three unchanged non-Darwin Codex environment/xcrun rows on this Darwin machine, plus the opt-in live Codex smoke. Full14skip count is unchanged; existing filesystem/platform, opt-in live GitHub/Codex and unavailable prime-rl workspace prerequisites remain explicitly unverified locally. No new skip marker, selection exclusion or relaxed assertion/gate was added.

The independent control review additionally required agent-message-only Codex extraction mutation so the top-level keeper fails its intended emitted-text/structured-output oracle rather than an earlier reasoning assertion. The refined3node selection fails as intended and passes restored. A separate first-turn MetricsEvent omission control now proves exact native event cardinality while CostEvent remains. The original zero-prompt-token control is named precisely. All31finalcontrols pass exact restoration; [independent mutation review](final-mutation-review.md) approves all62logs and35 behavioral keeper failures/restoredpasses; the draft no-effect/too-broad controls are not counted as adequate proof.
