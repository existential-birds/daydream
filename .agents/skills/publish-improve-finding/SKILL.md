---
name: publish-improve-finding
description: Capture or update a cleanup/refactor finding from the active session in improve_curated/, preserving human provenance and pre-fix evidence for blind Improve discovery benchmarks and later training.
---

# Publish an Improve finding

Turn an opportunity already discussed or investigated in the active session into
one durable Markdown record. This is capture, not a new repository audit. Here,
**publish means write the local finding document**. It does not authorize a commit,
push, GitHub issue, PR, merge, source edit, or training upload. Follow any separately
authorized actions without requesting the same permission again.

Read `improve_curated/AGENTS.md` and use [finding-template.md](finding-template.md).
The directory's schema is shared with `create-improve-issue` and `create-improve-pr`;
preserve its field names and any existing issue/PR links. After capture, the user can
invoke `create-improve-issue` to queue implementation in a future session. Do not
create that issue automatically as part of capture.

## Capture procedure

1. Read applicable repository instructions and inspect the active session for the
   actual finding, user judgments, code evidence, searches, and any existing fix.
   Use `git status --short`, `git rev-parse HEAD`, and the repository's configured
   remote to establish facts. Do not copy credentials from remote URLs. Preserve
   unrelated working-tree changes.
2. Search existing `improve_curated/*.md` records by affected symbols and root cause.
   Update an existing record for the same underlying opportunity, preserving its ID,
   path, original provenance, and historical evidence. Link distinct related findings
   with `related_findings`. Several symptoms of one simplification are one finding,
   not several recall credits.
3. Identify the source state **where the opportunity exists before its fix**. Record
   its full commit SHA in `observed.commit` only when established. Current HEAD is
   insufficient if relevant changes are uncommitted or the fix has already landed.
   Inspect historical evidence when needed; do not reset the user's checkout.
   Record dirty paths, whether the evidence depends on them, and an existing durable
   source archive or manifest reference if available. A patch alone omitting untracked
   or binary files is not an exact snapshot. Leave unavailable facts `null` and explain
   the gap; capture can succeed while benchmark eligibility remains unresolved.
4. Set provenance from the conversation. `origin.kind` describes who discovered the
   opportunity, not who wrote this document. Use `human` for human discovery, `agent`
   for agent discovery, and `mixed` for substantive joint discovery. Record a session
   reference only when available; include enough evidence in the document to survive
   a missing transcript. Never invent a session ID, quote, search, or reasoning trace.
5. Set `status: confirmed` only when a human has explicitly accepted the finding's
   validity and maintenance value. Record that person's stated identity (or `user`
   when no name is known), timestamp if known, and the actual confirmation evidence.
   Merely asking to save a candidate, a passing test, or an agent's confidence is not
   human confirmation. Use `proposed` for unresolved findings. Capture explicit
   rejections and their reasons as `rejected`; do not delete useful negative examples.
6. Choose `id: improve-YYYYMMDD-<8-hex-random>` using the actual capture date and a
   generated random suffix; check for collisions. Write
   `improve_curated/<id>-<short-kebab-case-slug>.md`. Keep the ID and filename stable
   across title changes and fixes. Keep one record per independent root cause.
7. Fill the template with facts from the session and narrowly targeted read-only
   checks. The minimum useful record contains provenance, observed source identity or
   its explicit gap, concrete evidence, maintenance value, proposed simplification,
   contracts to preserve, and matching criteria. Mark remaining sections `Unknown`
   or `Not yet measured` rather than delaying capture or inventing content.
8. If a benchmark baseline is already designated, verify that this opportunity
   exists in that exact state before setting `benchmark.verified: true`. Copy its
   immutable identity; do not silently designate current HEAD as the baseline. A
   finding introduced later belongs to another baseline. Never change a frozen
   baseline to accommodate a fix or a new finding.
9. Validate frontmatter shape and allowed values against `improve_curated/AGENTS.md`,
   confirm that referenced paths/symbols exist in the **observed** source state, and
   inspect the complete document for unsupported claims and exposed secrets. No
   source tests are needed for this documentation-only capture. Report the local
   file link, confirmation status, and any source/baseline provenance gap.

## Evidence quality

- Explain the relationship that exposes unnecessary complexity: duplicated ownership,
  repeated state/conversions, equivalent lifecycle implementations, unused extension
  machinery, or an existing implementation that can replace another. File length or
  a cosmetic preference alone does not establish a valuable simplification.
- Record observable search steps and useful negative results, not invented private
  reasoning. Distinguish what was observed from hypotheses and confidence estimates.
- Define a match by the underlying opportunity and supporting evidence, allowing
  different wording or a sound alternative fix. Define what would make a proposed
  match too broad, superficial, or behavior-breaking. The final patch is evidence,
  not the only acceptable answer.
- Keep estimated and observed savings separate. Record eliminated concepts and
  responsibilities as well as any line counts; LOC is supporting evidence, not a
  standalone reward. A coherent simplification need not satisfy an arbitrary size gate.
- Preserve original evidence after fixes. Append dated amendments for corrections or
  changed judgments; do not rewrite history to imply the final implementation was
  known at discovery time. `resolution` tracks implementation separately from validity.
- This record is hidden reference material for a **discovery** task. Supplying it to
  the evaluated agent turns discovery into a prompted repair task. The reference set
  is incomplete: unmatched agent findings require judgment, not automatic rejection.

If separately authorized to commit or push, use ordinary hook-enabled commands.
Never use `--no-verify`, hook-disabling environment variables, or any other bypass.
Fix hook failures on the current branch or stop and report the blocker.
