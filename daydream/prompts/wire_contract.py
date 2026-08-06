"""Static wire-contract review policy for deep-review prompts."""

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
