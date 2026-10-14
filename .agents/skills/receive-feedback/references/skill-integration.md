# Local Skill Integration

Choose available Daydream skills for the actual finding. Current repository instructions, user scope, and authorization govern their application. These specialized skills are optional: discover them in the current environment's skill catalog before reading them. If unavailable, use [VERIFICATION.md](../VERIFICATION.md), current repository standards, and direct code/command evidence. Verification and the authorized fix loop do not depend on installing sibling skill catalogs.

| Need | Local skill | Use |
|------|-------------|-----|
| Python quality review | `python-code-review` | Types, exceptions, async behavior, naming, and language-specific concerns against current repo standards |
| Pytest evidence | `pytest-code-review` | Test quality and fixtures; repo-mandated real-path coverage remains required |
| Structured verification | `review-verification-protocol` | Evidence and disposition discipline; preserve identities and uncertainty |
| Broader branch/PR standards and spec review | `code-review` | Pin comparison/spec and retain distinct axes; item verification does not require another full review. |
| Difficult bug/performance regression | `diagnosing-bugs` | Build a red-capable reproduction, test hypotheses, and verify the original conditions after fixing |
| Test-first fixes/integration tests | `tdd` | One observable behavior slice at a time. Use repository-established production entrypoints and dependencies without redundant confirmation. |
| Module/interface decision | `codebase-design` | Consult shared vocabulary for necessary architectural changes; do not initiate a broader redesign automatically |

Read supporting files only for the current branch of work.

## Daydream invariants

Read current `AGENTS.md` architecture details before changing these areas:

- `agent.run_agent()` is the sole agent execution entrypoint. Phases must not call backends/SDKs directly. Runtime fanout is bounded and preserves recorder context.
- `git_ops/` is the single Git/GitHub subprocess boundary. Preserve process/auth policy and keep credentials out of command lines, remote URLs, and published artifacts.
- Resolve private artifact ownership once and carry that owner/session throughout the run. Join writers before freezing immutable evidence; archive/evaluate/publish after freezing. Preserve snapshot binding and do not broaden backend read roots.
- Host-owned identity differs from provenance. Preserve `uid`, `item_uid`, and `source_uids` semantics; display numbering is not durable identity.
- Vendored `daydream/atif/` is updated wholesale, not patched locally. ATIF construction belongs in trajectory code, not phases or UI.
- Root `pyproject.toml`/`uv.lock` and the standalone RL project have separate gates. Use current repository commands and never bypass hooks.

If reviewer advice conflicts with a documented invariant, verify the underlying problem and choose a solution preserving that invariant. Cite both evidence and constraint; disagreement about a suggested implementation does not automatically invalidate a defect. If resolution requires a product decision or changes beyond authorized scope, identify the dependency and continue independent work.
