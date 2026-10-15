# Native Pi wall-time guidance evaluation

Issue #1534 adds truthful refreshed timing context. These measurements do not establish a latency benefit or improved deadline-pressure convergence.

The comparison uses pinned main `dbf4881156564c664920cc54ea9f483b688cfd15` and one frozen candidate production source. Each condition has an immutable Git snapshot, four changed Python files, a fixed source/diff hash, and two interleaved repetitions per producer. Pi 1.0.4 / Node 26.5.0 used the existing `nous` provider, `deepseek/deepseek-v4.1-flash`, and explicit `medium` effort. Each complete `ReviewInvestigation` had one 480-second/48-start cumulative allocation, host retries disabled, and identical stage/reservation/profile policy. Four fixed first-pass assignments were reviewed; the normal triage/admission path remained available. Outer runner routing, synthesis, fan-out, publication, and historical Shelfspace replay are outside this measurement. The separate production-runner regression covers cutoff/publication preservation.

The candidate-free control adds correct batch-cache and pagination APIs. Independently seeded cross-file defects omit seconds-to-milliseconds conversion in a cache caller and one-based-to-zero-based conversion in a feed caller. An external oracle, absent from model inputs, passes all 17 clean checks; each seed independently fails exactly its expected check family, and both seeds together fail five checks. The reviewer receives ordinary API contracts and source/assigned diffs, not the oracle or seed manifest.

| Control | Producer | Repeat | Elapsed (s) | Starts | Reported completion tokens | Findings | Reviewed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| clean | baseline | 1 | 52.300 | 22 | 5,496 | 0 FP | 4/4 |
| clean | candidate | 1 | 78.029 | 26 | 6,300 | 0 FP | 4/4 |
| clean | baseline | 2 | 47.915 | 23 | 5,041 | 0 FP | 4/4 |
| clean | candidate | 2 | 51.443 | 20 | 4,316 | 0 FP | 4/4 |
| defects | baseline | 1 | 47.887 | 20 | 4,868 | 2/2 TP, 0 FP | 4/4 |
| defects | candidate | 1 | 62.596 | 22 | 6,637 | 2/2 TP, 0 FP | 4/4 |
| defects | baseline | 2 | 50.827 | 21 | 5,989 | 2/2 TP, 0 FP | 4/4 |
| defects | candidate | 2 | 57.526 | 17 | 5,813 | 2/2 TP, 0 FP | 4/4 |

All eight scopes completed, admitted all 32 assigned targets, and made four successful native submissions each. Both seeds were recovered in both repetitions by both producers, with no false-positive roots, duplicates, unresolved candidates, failed/truncated tool calls, or recorded concrete findings lost at admission. Remaining capacity was unused honestly; no incomplete run is treated as a gain.

Independent assessment found citation imprecision: candidate defect repetition 1 anchors at cache.py:21/feed.py:11 instead of the faulty calls at 22/12; baseline defect repetition 2 anchors the feed finding at 10 instead of 12. All remain in the changed hunks with correct quoted code and cross-file diagnoses. Some supporting line-number prose is also inaccurate. Exact citation precision did not uniformly improve.

**The candidate was slower in every matched pair.** Mean scope elapsed was 62.399s versus 49.732s baseline (25.5% higher); ranges were 51.443–78.029s versus 47.887–52.300s. Mean clean-control elapsed was 64.736s versus 50.108s, and defect-control elapsed 60.061s versus 49.357s. Aggregate reported completion tokens were 23,066 versus 21,394; prompt tokens 595,980 versus 560,304. Aggregate starts were 85 versus 86. These measurements show no useful-work/latency benefit for the candidate on these controls. They do not justify a performance-improvement claim.

Every final trial used identical target bytes and its pinned target Git revision. Both runtime variants had one consistent production fingerprint; the harness was unchanged. Baseline production content SHA256: `e32fffee49d0a617482ca6d3a26b9bd379ea7e4a2512f0b935473af2b26a5233`. Candidate: `1d10e889643d5b16c267655ad0d0e382ca71de9cd44cf1ce480320713c535d01`. Harness: `7d4500f5fff75081574c48dde674bd41fccc7069f65e3a6f97b0e51100386546`.

Clean target: `9baa8698f12ef8b410b422c269e26e7156445db3`, diff SHA256 `f0d626800a08ccbefe7689beaff21352e11a4f0d29a6cbf938fafdc4e0350e1b`. Defect target: `85fd85115e0f6d5ad025df35eafc7d8ff47c1d9d`, diff SHA256 `b41207cbf171935ece7bdfece722b5238ced0112c94fcd946e816f4b6f734555`.

Raw reports retain an overbroad integrity warning: the harness interpreted the expected generated untracked `.daydream/` directory as fixture mutation. Separate integrity sidecars and two independent assessments confirm the original HEAD, every source hash, and an empty tracked diff in all eight trial clones; only expected review artifacts were added. The warning is qualified, not silently removed.

Two earlier candidate development pilots were excluded after the capability opt-out integration repair changed candidate source. They are preserved separately, and were not excluded by outcome or speed. The original baseline clean repetition 1 remained eligible because its source/harness/target were unchanged. Final order was baseline-clean1, candidate-clean1, candidate-defects1, baseline-defects1, baseline-defects2, candidate-defects2, candidate-clean2, baseline-clean2. Provider load, ordering, and the development interval are additional variability limitations.

Synthetic fixtures, the external oracle, frozen full-stage harness, raw native events, per-stage results, adjudications, integrity sidecars, and excluded pilots are retained in the task's durable local evidence directory under `evaluation/`. No historical private Shelfspace capture was exported and no dataset uploader was activated.

These small synthetic controls leave ample wall capacity, and two repetitions cannot establish a causal/general speed or cost effect. Native reported monetary cost was zero; actual billed cost is unknown. An elapsed-time reduction without full coverage and retained concrete findings would not count as improvement.

Deterministic native loopback tests separately prove one bounded note, independent clock origins, preparation/startup charging, progression without tool starts, monotonic never-negative countdowns, post-handoff wall-clock changes, optional/malformed/unbounded metadata, native compaction, failed/batched starts, and retry continuity. The production runner regression preserves the original source finding, provenance, snapshot-bound public output, and honest incomplete coverage at shared deadline cutoff. Three new owner-boundary checks fail against pinned pre-fix code for the intended missing wall guidance/metadata.
