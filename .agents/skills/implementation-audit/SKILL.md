---
name: implementation-audit
description: Audit implementation design in a PR, diff, or subsystem for scattered ownership, shallow interfaces, duplicated policy, and unnecessary complexity. Use for deeper design review or identifying the weakest part of an implementation.
---

# Implementation Audit

Find the design decisions that make correct behavior difficult to understand,
use, or change. A working implementation can still have a weak design; a design
finding does not need a contrived bug to justify it.

This audit owns production design and responsibility placement. `test-audit`
owns the value and coverage of tests; `refactor-audit`, when available, owns its
refactoring workflow. Hand findings to those skills where appropriate rather
than duplicating their work. Audit requests are read-only unless the user also
authorizes implementation.

## Establish the scope

Read repository guidance and the relevant subsystem contracts. Pin the revision,
comparison base for a diff, and any existing local edits. Use the issue or spec
to distinguish required complexity from accidental complexity. For a subsystem
audit, identify its entry points and consumers instead of inventing a diff.

Trace a representative operation through its owners: input, validation, state,
side effects, failure handling, and cleanup. Include relevant optional modes and
failure paths. Read full enclosing functions and callers; a large diff or long
function alone is not a finding. Split independent investigations by owner when
delegation helps, then reconcile findings across their shared interfaces.

## Examine ownership and interfaces

Use these questions to locate problems, not to produce a smell inventory:

- **Scattered ownership:** Which invariant requires several callers to remember
  the same checks, ordering, cleanup, or retry rules? Who owns enforcing it?
- **Shallow interfaces:** How much implementation knowledge must a caller carry?
  Look for paired arguments that must agree, flags that select internal recovery
  modes, and callbacks that make callers enforce the callee's contract.
- **Duplicated policy:** Is the same rule, schema, or state transition maintained
  in multiple places? Distinguish duplicate authority from independent checks
  serving different contracts.
- **Fragmented state:** Do several objects store overlapping facts that need
  synchronization? Can the implementation derive them from an existing owner?
- **Weak locality:** Trace an ordinary change to an existing rule. Which owners,
  signatures, and callers must change together, and why?
- **Unnecessary machinery:** Which wrappers, compatibility paths, abstractions,
  configuration, or fallback layers exist without a current consumer or distinct
  contract? Search all consumers before recommending removal.

Prefer a cohesive owner with a small interface. Centralizing a policy means the
owning operations enforce it; merely moving repeated checks into a helper still
leaves callers responsible for remembering the protocol. Necessary checks may
execute repeatedly at filesystem, process, or transaction boundaries while their
policy and lifecycle remain owned in one place. Preserve those checks and keep
unrelated responsibilities with their existing owners.

## Prove the design cost

For each candidate, collect:

1. **Observed structure:** Cite the responsible code and representative callers.
   State the rule or responsibility currently spread between them.
2. **Concrete cost:** Show today's repeated caller obligations, inconsistent
   handling, or coordinated edits needed for an ordinary maintenance change.
   Explain the consequence in plain language. A plausible future feature alone
   is weak evidence.
3. **Counterevidence:** Check whether separate trust boundaries, lifecycles,
   extension consumers, or failure semantics justify the apparent duplication.
   Identify the strongest reason to keep the current arrangement.
4. **Simpler ownership:** Name the existing owner to deepen or the cohesive
   responsibility that deserves one. Describe what callers would stop knowing
   and which old plumbing could disappear. A new layer must reduce obligations,
   not just relocate lines or create a general framework.
5. **Preservation and proof:** Identify behavior that must survive and the
   existing acceptance journeys that demonstrate it. Use the repository's
   test-audit guidance if test changes are proposed.

Keep confirmed bugs, design findings, and unresolved suspicions distinct. For a
bug, establish the trigger and observable consequence with suitable evidence;
read the lifecycle owner before alleging missing validation or cleanup. When a
reproduction forces unusual timing or external interference, state that
assumption and assess its plausibility. Do not promote it above demonstrated
design costs just because it can be made to fail.

## Report and handoff

Lead with the most consequential supported design problem and explain it simply:
what is spread around, why that burdens callers or maintainers, and where it
should live. Rank by current impact, breadth of affected callers, and confidence;
report fewer findings when the evidence is weak. Separate remaining findings
from questions, and state inspected scope and verification limits.

Each finding should carry its source locations, concrete cost, proposed owner,
preserved contracts, and validation approach. Treat reduced repetition or line
count as supporting evidence, not the goal. If the design is justified, say so.

When asked for an implementation prompt, carry those facts and the verified
baseline into a self-contained handoff using the repository's preparation
workflow. Require the change to simplify real callers and remove superseded
plumbing while preserving independent behavior and failure-path proof. Creating
that prompt does not authorize executing it or publishing changes.
