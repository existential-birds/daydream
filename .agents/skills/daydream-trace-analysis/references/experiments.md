# Propose a matched experiment

Turn a supported pattern into one reviewable harness change. Keep implementation
and live execution within the user's existing authorization; these helpers only
prepare and validate JSON.

```bash
python3 "$SKILL/scripts/reports.py" experiment-template \
  "$ANALYSIS/synthesis.json" > "$ANALYSIS/experiment.json"
# Fill the packet and remove its template marker.
python3 "$SKILL/scripts/reports.py" validate-experiment "$ANALYSIS/experiment.json"
```

Specify the supported pattern, one concrete change, and its predicted measurable
effect. Match baseline/candidate conditions, including reasoning effort, and
record the producer revisions. Only producer revision may differ; extra condition
fields must match too. Use the generated template for required fields. For a
budget, profile, or reasoning-effort intervention, explain that deliberate
difference in a separate proposal; it cannot validate as a matched packet.

Use repeated trials (at least two per condition) and explicit discovery and
held-out workload IDs. Baseline and candidate evaluate the same union of those
disjoint sets. Preserve held-out evidence for evaluation, not prompt tuning.
Choose enough repetitions to assess variability; the minimum is not statistical
significance. State the metric, expected direction, and rejection threshold.

Require quality guardrails alongside efficiency metrics: retained adjudicated
findings, clean-case false positives where labeled, retained authorized fixes,
truthful incomplete coverage, and executed verification appropriate to the flow.
Use available independent labels; missing adjudication leaves quality unknown.
Reject changes that lose required findings or coverage even when faster.

Reuse existing owners rather than inventing metric aggregation:

- `scripts/measure_review_runtime.py`: isolated matched fixtures; currently Pi
  per-stack reviewer measurements, not a universal full-flow benchmark.
- `daydream/eval/latency_report.py`: corpus/profile timing and quality reports.
- `daydream/trajectory/timing.py` and `daydream/eval/analyzer.py`: timing and
  billing ownership; the analyzer consumes archived workspaces/snapshots rather
  than embedded JSONL directly.

Present the packet with recurrence, counterexamples, uncertainty, likely quality
risk, and the rejection criterion. A valid packet is an experiment proposal;
it is not evidence that the change improved the reviewer.
