# Go Review Option — Design

**Date:** 2026-02-21

## Summary

Add `--go` flag to daydream to invoke the `beagle-go:review-go` skill, mirroring the existing Python, React, and Elixir review options.

## Changes

### `daydream/config.py`

- Add `GO = "4"` to `ReviewSkillChoice` enum
- Add `ReviewSkillChoice.GO: "beagle-go:review-go"` to `REVIEW_SKILLS`
- `SKILL_MAP` is derived automatically — no changes needed

### `daydream/cli.py`

- Add `"go"` to `--skill` choices list
- Add `--go` flag to the mutually-exclusive skill group (maps `dest="skill"` to `"go"`)

## Scope

5 lines across 2 files. No changes to runner, phases, agent, or UI layers.

## Tests

- One test case for `--go` flag → `skill="go"` in `RunConfig`
- One test confirming `SKILL_MAP["go"] == "beagle-go:review-go"`
