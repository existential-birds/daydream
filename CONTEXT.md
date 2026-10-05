# Daydream

Daydream reviews code changes and collects evidence for evaluating and improving code review, including training review models.

## Language

**Run**:
One execution of a Daydream workflow, with its own session identity and execution outcome.

**Trajectory**:
The recorded sequence of agent interactions within a run. A run may contain a root trajectory and related child trajectories.

**Finding**:
A structured claim about a problem in the reviewed code, with location and supporting explanation.

**Finding identity**:
A reference to a particular finding within a particular run, retaining its relationship to the review claims from which it was derived.
_Avoid_: Finding fingerprint, display number

**Finding fingerprint**:
A signature used to match similar reported defects. Matching fingerprints do not make two findings the same occurrence in a run.
_Avoid_: Finding identity

**Run record**:
An immutable account of a run's captured task, trajectories, findings, verification evidence, provenance, and evidence completeness.
_Avoid_: Training example, trace

**Observation**:
A later acquired fact, judgment, or correction associated with a run or finding, retaining its source and history. It may describe an outcome or enrich previously unavailable task information.
_Avoid_: Run record, gold label

**Adjudication**:
Resolution of judgments about a finding using their evidence, author roles, and precedence. A model suggestion alone is not a decisive human judgment.

**Intrinsic reward**:
A score derived from recorded review and verifier evidence under a specified reward policy.
_Avoid_: Posterior cost

**Posterior cost**:
A separate penalty derived from later maintainer outcomes for a review.
_Avoid_: Intrinsic reward

**Frozen corpus**:
A reproducible selection and projection of run evidence and eligible observations into training examples under pinned policies and split rules.
_Avoid_: Run archive, raw run records

**Dataset snapshot**:
A fixed selection of captured runs and observations that remains unchanged when new evidence arrives. Different snapshots may include different evidence about the same run.
_Avoid_: Latest dataset, frozen run snapshot

**Review improvement report**:
An analysis of collected reviews and their supporting evidence that helps the operator identify changes to Daydream's review behavior.
_Avoid_: Training dataset, source-code quality report
