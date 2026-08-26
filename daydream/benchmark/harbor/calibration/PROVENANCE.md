# Calibration Fixture Provenance

**Origin:** `llm_generated`
**Human reviewed:** `false`
**Labels:** `unverified`

The 24 pairs in `pairs.json` are LLM-synthesized synthetic content. The
content is **source-free** — never copied from real repository source, diffs,
or any source-derived artifact. The finding shape mirrors `golden-review.json`
(`finding_id`/`candidate_id`, `title`, `severity`, `path`, `start_line`,
`end_line`, `body`); the content is not recoverable from any reference.

The 24 pairs are generated synthetic content covering all eight
semantic-match categories with a fixed 12 `match` / 12 `nonmatch` class
balance:

- `wording_location` — same defect, different wording and/or location (label `match`)
- `related_distinct` — related but separate defects (label `nonmatch`)
- `negation` — one finding states a defect the other negates
- `same_class_different_components` — same bug class in different files
- `severity_disagreement` — same defect, differing severity (label `match`)
- `locationless` — all-null file/line location on at least one finding
- `delimiter_injection` — literal `</gold_finding>`/`<candidate_finding>` text to exercise the escaping fence
- `near_miss` — plausible but distinct near-match (label `nonmatch`)

These labels have **not** been independently verified by a human. They are
**unverified** and are used only for explicit, diagnostic calibration
exercises; they do not gate the Oracle or the default benchmark run. This is a
static, fixed fixture; it is never mutated by a calibration run.