"""Static wire-contract review policy for deep-review prompts."""

__all__ = [
    "ANTI_SLOP_RUBRIC_INSTRUCTION",
    "CONFIG_FLOW_TRACE_INSTRUCTION",
    "CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION",
    "TEST_QUALITY_RUBRIC_INSTRUCTION",
    "TRUST_MODEL_INSTRUCTION",
    "VERIFICATION_PROTOCOL_INSTRUCTION",
    "WIRE_CONTRACT_GENERIC_INSTRUCTION",
    "WIRE_CONTRACT_RUST_INSTRUCTION",
    "generic_fallback_review_policy",
    "per_stack_review_policy",
    "structural_review_policy",
]

# Repo-wide cross-file symbol existence check (issue #310). Embedded inline as
# instruction text because reviewers run with cwd set to the reviewed repo, so
# a bare skill-file read resolves against that repo and silently drops the gate.
CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION = (
    "Cross-file symbol existence check (apply before flagging anything about a "
    "symbol defined OUTSIDE the diff):\n"
    "  1. Every referenced symbol not defined in this diff -- a function, a "
    "subcommand invoked by a CLI wrapper, a trait method implemented by "
    "generated code, a config field, a CLI flag -- must be verified to exist "
    "in the checked-out repo before you report a finding about it.\n"
    "  2. Evidence (Gate-2): `rg` for the definition in the repo and cite the "
    "file:line where it is declared. Never assert a symbol's behavior from the "
    "call site alone.\n"
    "  3. If no definition can be found, say so explicitly and downgrade the "
    "finding's confidence -- an unresolved reference is reportable only when "
    "the missing definition is real, never when you simply failed to locate it."
)

CONFIG_FLOW_TRACE_INSTRUCTION = (
    "Config/env flow trace (apply to every config field or env var plumbed "
    "through layers):\n"
    "  1. Trace the full path of each plumbed field: config struct -> driver "
    "config -> request construction.\n"
    "  2. Emit a one-line trace statement per field naming where it is parsed, "
    "where it is forwarded, and where (if anywhere) it reaches the request.\n"
    "  3. Flag silent drops -- a field parsed but never forwarded to the next "
    "layer.\n"
    "  4. Flag double-resolves -- the same value read twice at different points "
    "with the source able to change between reads (TOCTOU)."
)

TRUST_MODEL_INSTRUCTION = (
    "Trust-model check (apply to every security-relevant marker: cache-control "
    "injection, trust boundaries, escaping, credential forwarding):\n"
    "  For each marker, state the trust model in one sentence: who is the "
    "untrusted party here, and does this path honor the boundary?\n"
    "  Flag any path that instructs an untrusted party to retain or forward "
    "sensitive content -- e.g. an edge proxy echoing an untrusted response's "
    "cache-control directive, or credentials passed through an intermediate hop."
)

VERIFICATION_PROTOCOL_INSTRUCTION = (
    "Before writing findings, apply the review-verification-protocol gates "
    "(stated inline here — no skill file read is required):\n"
    "  Gate-0 anti-confabulation (before ANY finding): echo the exact artifact "
    "you are judging — file:line plus the cited code, read freshly in THIS turn, "
    "not recalled. The source is the only truth; never infer a finding from the "
    "branch name, cwd, or memory. A finding without a same-turn echo of its "
    "target is INVALID.\n"
    "  Gate 1 (anchor): read the full enclosing symbol/module, not just the diff "
    "hunk; state the file path and line range you are judging.\n"
    "  Gate 2 (evidence): produce an artifact for the finding's type — pasted "
    'tool output, a file:line citation, or an explicit "none" / "N matches" '
    'after a repo search. Never claim you "looked" without an artifact.\n'
    "  Gate 3 (severity): calibrate severity to impact; a request for net-new "
    "code that did not exist in scope is Informational only.\n"
    "Do NOT report a finding that fails any gate."
)

TEST_QUALITY_RUBRIC_INSTRUCTION = (
    "Apply the test-quality rubric to every test hunk in the diff "
    "(stated inline here — no skill file read is required):\n"
    "  1. Would this test fail if the behavior under test were wrong? Scan for "
    "vacuous assertions — e.g. `read_to_string(...).unwrap_or_default()` "
    "returning empty on failure, expected values built with the same helper "
    "under test, a wait/retry helper returning the last nonmatching frame.\n"
    "  2. Does it assert observable consequences (output, filesystem, exit code, "
    "store state) rather than internal fields/pointers/dispatch plumbing "
    "(`context as *const _ as usize`, dispatch internals, event payloads with no "
    "observable check)?\n"
    "  3. Is it deterministic (no sleeps, no `yield_now()` reaping assumptions, "
    "no environment leaks — require restore guards for any env mutation)?\n"
    "  4. Does it exercise the new behavior through the canonical public path (no "
    "raw `system_prompt` copies, no bypassing the public API the behavior lives "
    "behind)?\n"
    "  5. Does it compile on all platforms (`#[cfg]` gates)?\n"
    "Layering awareness: legitimate pure-function seams are fine — a unit test of "
    "a pure `build_driver_request` or driver-boundary propagation helper is NOT an "
    "internal-field assertion. Flag a seam ONLY when it bypasses the observable "
    "behavior the test claims to cover."
)

ANTI_SLOP_RUBRIC_INSTRUCTION = (
    "Apply the anti-slop rubric to every code hunk in the diff "
    "(stated inline here -- no skill file read is required). It targets the "
    "SlopCodeBench degradation patterns -- structural erosion, verbosity, "
    "duplication:\n"
    "  1. Flag complexity concentration: when a hunk adds logic to a function "
    "that is already large/high-complexity (cyclomatic complexity > ~10, or > ~80 "
    "lines), require extraction into focused callables -- especially when the "
    "same pattern (flag pair, branch ladder, error guard) is repeated verbatim.\n"
    "  2. Verbosity: flag redundant code -- identity comprehensions instead of "
    "filter/map, empty-list guards inside loops, single-use intermediate "
    "variables, casts to dodge type checking, trivial wrapper functions, "
    "nested ladders.\n"
    "  3. Duplication: flag the same hunk structure repeated (e.g. N flags x 2 "
    "branches) that should be a loop/helper/template.\n"
    "  4. Severity: maintainability findings are medium/low -- never high -- "
    "under this rubric, full stop. The structural lens may flag real erosion, "
    "but anti-slop findings never escalate to high.\n"
    "  5. Scope: when erosion is pre-existing-and-growing, flag the growth, not "
    "the whole function -- report only the newly introduced growth, scoped to "
    "this diff's contribution."
)

# Per-stack Rust wire-contract checklist (issue #311). This is kept outside the
# prompt builder so the policy can be reviewed and tested independently.
WIRE_CONTRACT_RUST_INSTRUCTION = (
    "Wire-contract check (apply to every new-or-changed #[derive(...)] type "
    "that crosses a wire boundary -- config structs, API payloads, persisted "
    "shapes):\n"
    "  1. Nested serde defaults: every nested input struct field that can be "
    "absent must be covered by field-level #[serde(default)], an appropriate "
    "struct-level #[serde(default)] with Default values, or an intentional "
    "optional/custom deserialization contract; otherwise a partial object fails "
    "deserialization. #[serde(skip_serializing_if = \"...\")] only affects "
    "output behavior and does not accept an absent input field. Use "
    "#[serde(flatten)] where extra fields are intentionally absorbed.\n"
    "  2. Enum routing matches the constructed shape: when a Debug/Display/"
    "as_str impl is used to route a value, the match arms must match what is "
    "actually constructed -- a JSON-object variant never matches a string arm "
    "(the StepValue::Json.as_str() approval gate that never fires). Compare the "
    "constructed variant's shape against every match arm before trusting the "
    "route."
)


# Generic-fallback wire-contract checklist (issue #311). It distinguishes
# parsing a complete URL from safely constructing one from untrusted components.
WIRE_CONTRACT_GENERIC_INSTRUCTION = (
    "Wire-contract check (apply to every value assembled into a URL, shell "
    "command, heredoc, or example payload):\n"
    "  1. URL/arg/heredoc construction: do not build these by string "
    "interpolation of untrusted values -- reserved characters (a password "
    "containing '@' or '#', query separators, shell metacharacters) silently "
    "corrupt or misparse the result. url::Url, URL(string:), and new URL(...) "
    "are whole-URL parsers and must not receive interpolated components; use a "
    "component-aware URL builder or explicitly percent-encode each component, "
    "an argument-vector/quoting helper (shlex, subprocess arg lists), or a "
    "structured builder.\n"
    "  2. Cross-format consistency: doc examples and sample payloads must "
    "match the actual schema -- an added or renamed required field, or a "
    "parameter the runtime rejects, invalidates the example. Verify the "
    "example against the current struct/interface/signature before treating "
    "it as authoritative."
)


def per_stack_review_policy(stack_name: str) -> tuple[str, ...]:
    """Return inline review policy for one language-specific stack."""
    policy = (
        TEST_QUALITY_RUBRIC_INSTRUCTION,
        ANTI_SLOP_RUBRIC_INSTRUCTION,
        CONFIG_FLOW_TRACE_INSTRUCTION,
        TRUST_MODEL_INSTRUCTION,
    )
    return policy + (WIRE_CONTRACT_RUST_INSTRUCTION,) if stack_name == "rust" else policy


def structural_review_policy() -> tuple[str, ...]:
    """Return inline review policy for the repo-wide structural reviewer."""
    return (
        VERIFICATION_PROTOCOL_INSTRUCTION,
        ANTI_SLOP_RUBRIC_INSTRUCTION,
        CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION,
        TRUST_MODEL_INSTRUCTION,
    )


def generic_fallback_review_policy() -> tuple[str, ...]:
    """Return inline review policy for the language-agnostic fallback reviewer."""
    return (
        VERIFICATION_PROTOCOL_INSTRUCTION,
        CONFIG_FLOW_TRACE_INSTRUCTION,
        TRUST_MODEL_INSTRUCTION,
        WIRE_CONTRACT_GENERIC_INSTRUCTION,
    )
