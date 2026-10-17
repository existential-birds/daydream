# Assessments and synthesis

Generate the executable report shape with `reports.py template` rather than
recreating it. Preserve its `plan_id` and cohort, batch, and run identities; fill
the analysis fields and remove the `template` marker. Run `synthesize` against
that same frozen plan. Its digest binds the question, limits, and assignments;
regenerate templates after changing the plan. Validation checks those bindings,
frozen sources, resolving pointers, and excerpts; it cannot establish causal truth.

Generate each assessment's fields, then insert the filled object in its run's
`assessments` array:

```bash
python3 "$SKILL/scripts/reports.py" assessment-template \
  --pattern unchanged-search --category investigation_loop
```

The initial unknown verdict needs an explanation before validation succeeds.

## Choose a category

| Category | Investigate |
| --- | --- |
| `environment_scope` | Wrong roots, tool syntax, repeated discovery, misunderstood assignments |
| `investigation_loop` | Repeated actions without new evidence or a changed hypothesis |
| `handoff_loss` | Evidence rediscovered, dropped, or contradicted between stages |
| `verification_gap` | Accepted fixes versus executed checks, retained changes, and verdicts |
| `budget_allocation` | Investigation consuming time needed for required finalization |
| `finding_quality` | Unsupported or missed findings supported by independent adjudication |
| `backend_lifecycle` | Host-observed transport, cancellation, drain, or cleanup failures |
| `capture_failure` | Missing, failed, or partial acquisition of evidence |

For effectiveness, inspect task/assignment compliance, retained findings and fixes,
truthful coverage, and verification evidence. Finding counts and run success are
not correctness labels. Where adjudication or needed actions are missing, mark
finding quality unknown rather than infer false positives or missed findings.

## Fill each run

Run `status` is `analyzed`, `skipped`, or `unavailable`; explain skipped or
unavailable evidence. Each assessment carries a stable `pattern` name, category,
`verdict`, `occurrences`, `observation`, confidence, evidence, counterevidence,
and intervention. Use one assessment per pattern per run. Keep source paths,
line numbers, hashes, and comparison context in the manifest; synthesis retains
that frozen provenance alongside the analysis.

- `present`: observed pattern, positive occurrence count, and decisive evidence.
- `absent`: an assessed counterexample, zero occurrences, and supporting evidence.
- `unknown`: insufficient evidence, null occurrences, and an explanation.

Evidence entries use an exact JSON `pointer` and short `excerpt` at that pointer.
Use a narrow string or leaf object, not the whole record. Include the sequence,
later recovery, alternative explanations, and confidence in the observation.
Propose the smallest prompt, tool, handoff, or orchestration intervention.
The plan bounds assessments per run and excerpt characters; text fields are
bounded by the helper. Consult `--help` and generated output for current fields.

## Read the synthesis

The helper counts affected distinct runs / assessable runs (`present + absent`),
separately from occurrence totals. Unknown, unassessed, skipped, unavailable,
and missing-run states remain explicit. Results retain comparison-context strata;
raw totals across different configurations are descriptive, not causal estimates.

The main agent merges equivalent pattern names only after checking the same
behavior is described. Review disagreements and counterexamples before ranking
interventions. Missing batches or unfinished runs prevent complete analysis;
structurally valid output may still have insufficient evidence for a conclusion.
Report measured calls, output, tokens, and durations only where recorded.
Root billing can already include forks; overlapping durations are not additive.
Use [format.md](format.md) for metric ownership and authoritative reducers.
