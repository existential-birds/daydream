# Calibration Fixture Provenance

**Reviewer:** Daydream parsing + rules team
**Review date:** 2026-08-23
**Attestation:** Every finding in `pairs.json` is **source-free** — hand-authored
synthetic content, never copied from real repository source, diffs, or any
source-derived artifact. The finding shape mirrors `golden-review.json`
(`finding_id`/`candidate_id`, `title`, `severity`, `path`, `start_line`,
`end_line`, `body`); the content is not recoverable from any reference.

The 24 pairs cover all eight semantic-match categories, each independently
human-attested against the category it exercises:

- `wording_location` — same defect, different wording and/or location (label `match`)
- `related_distinct` — related but separate defects (label `nonmatch`)
- `negation` — one finding states a defect the other negates
- `same_class_different_components` — same bug class in different files
- `severity_disagreement` — same defect, differing severity (label `match`)
- `locationless` — all-null file/line location on at least one finding
- `delimiter_injection` — literal `</gold_finding>`/`<candidate_finding>` text to exercise the escaping fence
- `near_miss` — plausible but distinct near-match (label `nonmatch`)

Class balance is fixed: 12 `match` / 12 `nonmatch`. This is a static, fixed
fixture; it is never mutated by a calibration run.