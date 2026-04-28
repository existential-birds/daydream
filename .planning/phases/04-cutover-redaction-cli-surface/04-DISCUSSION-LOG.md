# Phase 4: Cutover + Redaction + CLI Surface - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-28
**Phase:** 04-cutover-redaction-cli-surface
**Areas discussed:** Redaction token style, --debug removal posture, Operational log line fate

---

## Redaction Token Style

### Q1: Type-specific vs generic redaction tokens

| Option | Description | Selected |
|--------|-------------|----------|
| Type-specific tokens | Each pattern category gets its own replacement: [REDACTED_API_KEY], [REDACTED_PATH], etc. | ✓ |
| Generic [REDACTED] | Single token everywhere. Simpler but less informative. | |
| You decide | Claude picks during implementation. | |

**User's choice:** Type-specific tokens
**Notes:** User selected based on preview showing consumer debugging experience.

### Q2: Path redaction scope

| Option | Description | Selected |
|--------|-------------|----------|
| Username segment only | Replace only the username portion, preserve project-relative path | ✓ |
| Entire home prefix | Replace everything up to and including home directory | |
| You decide | Claude picks based on regex feasibility | |

**User's choice:** Username segment only
**Notes:** Preserves project-relative paths for trajectory replay and debugging.

### Q3: .env-style key=value handling

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve key, redact value | OPENAI_API_KEY=[REDACTED_ENV_VAR]. Only matches secret-indicating key names. | ✓ |
| Redact entire line | [REDACTED_ENV_LINE]. Safer but loses context. | |
| You decide | Claude picks the approach. | |

**User's choice:** Preserve key, redact value
**Notes:** Non-secret env vars (DEBUG=true, APP_NAME=myproject) pass through unredacted.

### Q4: Flat regex vs JSON-aware deep walk

| Option | Description | Selected |
|--------|-------------|----------|
| Flat regex on serialized text | Single _redact_text() method on raw strings. | ✓ |
| JSON-aware deep walk | Parse ToolCall.arguments as JSON, walk values, re-serialize. | |
| You decide | Claude picks based on ATIF model field types. | |

**User's choice:** Flat regex on serialized text

---

## --debug Removal Posture

### Q1: --debug removal behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Hard reject | argparse error: 'unrecognized arguments: --debug'. Clean break. | ✓ |
| One-release deprecation warning | Accept --debug, print warning, continue without debug logging. | |
| You decide | Simplest to implement. | |

**User's choice:** Hard reject

### Q2: Explicit --trajectory write failure behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Exit with error | User explicitly asked for that path; fail loudly. | ✓ |
| Warn and continue | Print warning but proceed with run. | |
| You decide | Based on Phase 2 D-11. | |

**User's choice:** Exit with error
**Notes:** User initially selected "Warn and continue" then corrected to "Exit with error" (hard fail). Matches Phase 2 D-11 intent.

### Q3: SIGINT partial flush path

| Option | Description | Selected |
|--------|-------------|----------|
| .partial suffix | Write to <path>.partial. Clean filesystem separation. | ✓ |
| Same path, partial flag inside | Write to normal path with extra.partial=true. | |
| You decide | Cleanest for ATIF consumers. | |

**User's choice:** .partial suffix

---

## Operational Log Line Fate

### Q1: Fate of ~30 _log_debug call sites

| Option | Description | Selected |
|--------|-------------|----------|
| Promote useful ones to UI | Errors→print_error, warnings→print_warning, redundant→silently removed | ✓ |
| Silent removal for all | Delete every _log_debug call site. | |
| You decide | Claude audits each site case-by-case. | |

**User's choice:** Promote useful ones to UI
**Notes:** Specific categorization agreed: EXECUTE_ERROR/INIT_ERROR/PHASE2_ERROR → print_error; REVERT/PARSE_FALLBACK/TTT_* → print_warning; all event-mirroring and best-effort sites → silent removal.

### Q2: Quiet-mode contract for promoted messages

| Option | Description | Selected |
|--------|-------------|----------|
| Errors always shown, warnings respect quiet | print_error() always displays; print_warning() suppressed in --quiet. | ✓ |
| All suppressed in quiet | Both errors and warnings suppressed. Maximum silence. | |
| You decide | Follow existing codebase patterns. | |

**User's choice:** Errors always shown, warnings respect quiet

---

## Claude's Discretion

- AST sweep implementation for CUT-08
- Exact regex pattern details for each redaction category
- Ordering and atomicity of legacy removal commits
- Test mock/assertion updates for debug_log removal
- SIGINT handler integration mechanics
- Redaction failure mode internals (REDA-05)

## Deferred Ideas

- `--no-redact` escape hatch for debugging redaction issues
- Custom redaction pattern configuration
- Trajectory streaming writes (PERF-01)
