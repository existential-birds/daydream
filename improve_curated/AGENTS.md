# Human-curated Improve findings

This directory holds durable cleanup/refactor opportunities discovered during human
review and assisted sessions. Each Markdown record preserves the opportunity before
its fix, the human judgment, and eventual resolution evidence. Use
`.agents/skills/publish-improve-finding/SKILL.md` to capture a finding and
`.agents/skills/create-improve-pr/SKILL.md` to connect an implementation PR.
Do not create example findings that could be mistaken for real reference data.

Implementation tracking: [Prime-based Improve discovery benchmark and training epic
#1455](https://github.com/existential-birds/daydream/issues/1455). Its sub-issues
cover snapshot materialization, harness integration, scoring, evaluation, and training.

## Benchmark boundary

These records are **hidden grader reference material**, not input for an agent asked
to discover improvements. Keeping them in this repository does not make an ordinary
checkout safe for evaluation. Future snapshot builders must explicitly exclude this
entire directory and any copies, fix diffs, PR descriptions, prior audit reports,
session transcripts, or other answer-bearing artifacts. Also exclude `.git` history,
remotes, and access to future commits or answer-bearing network resources. A checkout
of an old SHA inside the live clone can still reveal later fixes.

Keep frozen target snapshots independent of the current Improve harness. Merging
cleanup PRs is compatible with repeatedly auditing a preserved pre-fix snapshot.
Never move a frozen baseline to make a finding fit. A finding is eligible only for
snapshots where its opportunity has been verified to exist. Preserve exact dirty
source content separately when necessary; HEAD alone cannot identify it.

The initial benchmark measures blind rediscovery of human-confirmed opportunities.
Match root causes and supporting evidence, not exact phrasing or exact patch shape.
Supplying a finding or patch creates a separate repair/review task. The human reference
set is partial: unmatched findings remain unjudged until reviewed. Related symptoms
must not inflate recall. LOC reduction is evidence, not the reward by itself.

Version reference-set changes and scores. Cases used to tune Improve are development
data; keep related findings, overlapping snapshots, and before/after variants together
when making eventual development/holdout splits. Training and benchmark integration
should favor Prime Intellect's supported stack. This directory defines records, not
a bespoke runner, reward service, dataset backend, or replacement training platform.

## Files and shared schema (version 1)

One file per coherent root cause:
`<id>-<short-kebab-case-slug>.md`, with ID `improve-YYYYMMDD-<8-hex-random>`.
IDs and filenames remain stable. Search existing records before creating another.
Update a duplicate; link related but independent opportunities. Preserve rejected
candidates and explain the rejection. `AGENTS.md` is guidance, not a finding.

The frontmatter fields below are the shared contract for both skills. Use YAML null
for unknown facts, empty lists for absent entries, quoted timestamps/SHA strings,
and no placeholder values in saved findings. Keep every top-level field present;
the capture template lives alongside `publish-improve-finding/SKILL.md`.

| Field | Meaning and allowed shape |
| --- | --- |
| `schema_version` | Integer `1`. |
| `id` | Unique stable ID as above. |
| `title` | Concrete opportunity, not a generic instruction to refactor. |
| `repository` | Canonical repository URL without credentials, or `null` if unknown. |
| `discovered_at` | Actual discovery timestamp, ISO 8601 with timezone, or `null` if unknown. Do not substitute capture time for an unknown discovery time. |
| `origin.kind` | `human`, `agent`, or `mixed`; discovery provenance, not document authorship. |
| `origin.session_ref` | Existing durable session/trace reference or `null`. No invented IDs. |
| `origin.human_confirmation` | `null`, or `{by, at, evidence}` recording explicit human acceptance. `by` is the stated identity or `user`; `at` may be `null`; `evidence` is a quote or faithful summary with context/reference. |
| `status` | `proposed`, `confirmed`, or `rejected`. `confirmed` requires human confirmation evidence; agent confidence or a successful fix is insufficient. |
| `observed.commit` | Full pre-fix Git commit SHA, or `null` when not established. Never record the post-fix HEAD as the original source. |
| `observed.working_tree` | `clean`, `dirty`, or `unknown`, describing the observed source state. |
| `observed.snapshot_ref` | Durable reference to exact source archive/manifest, or `null`. Include retrieval location/checksum in the body. |
| `observed.dirty_paths` | List of known modified, staged, untracked, or otherwise relevant dirty paths. Explain their relevance/completeness in the body. Empty does not prove cleanliness when state is unknown. |
| `benchmark.commit` | Full designated baseline SHA, or `null`; it need not equal `observed.commit`. |
| `benchmark.snapshot_ref` | Immutable benchmark source archive/manifest reference, or `null`. |
| `benchmark.verified` | Boolean; `true` only after checking the opportunity in the identified baseline. Requires a commit and/or immutable snapshot reference, and verification evidence in the body. A dirty baseline needs exact source identity beyond its base commit. |
| `affected_paths` | Repository-relative path strings in the observed source. |
| `affected_symbols` | Symbol strings, preferably qualified with path/module. |
| `related_findings` | Stable IDs of related records; describe overlap in the body. |
| `resolution.status` | `open`, `in_progress`, `fixed`, or `wont_fix`, independent of finding validity. A draft PR is `in_progress`, not `fixed`. |
| `resolution.prs` | Canonical GitHub PR URL strings; append idempotently. |
| `resolution.fix_commits` | Full implementation/merge commit SHAs established by evidence; distinguish their roles in the body. |
| `resolution.validation` | List of `{command, result, checked_commit}` objects. Use exact commands and observed results; `checked_commit` is a full SHA or `null` for dirty source, with the exact tested source reference/gap explained in the body. |

Body sections retain evidence/root cause, maintenance value, proposed simplification
and preserved contracts, observable discovery trail and human judgment, discovery
matching criteria (including alternatives and rejection conditions), source/baseline
provenance, resolution/validation, estimated versus observed savings, and amendments.
Unknown detail may remain explicitly unknown: missing enrichment should not prevent
capture. Missing source identity or confirmation does prevent treating a candidate
as verified human gold for a frozen benchmark.

## Updating records

Preserve original source identity and discovery evidence when fixing a finding.
Append dated corrections and judgment changes rather than silently rewriting their
history. A change in human judgment may update `status`; retain the prior judgment
and explanation in the body. Creating a PR adds its URL to `resolution.prs` and
links back to this record in the PR description. Do not mark a finding fixed merely
because a PR exists or tests pass; record the actual implemented/landed state.

Inspect references against their recorded source state rather than assuming deleted
symbols must still exist at HEAD. Keep secrets, credentials, and unrelated private
session material out of records. Do not change application code while capturing a
finding. Ordinary hook-enabled commit/push commands remain mandatory whenever those
actions are separately authorized; never bypass a failing hook.
