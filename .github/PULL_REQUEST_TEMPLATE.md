## Summary

<!-- 1-3 bullet points describing what this PR does. Focus on the user-visible or developer-visible impact. -->

-

## Motivation

<!-- Why is this change needed? What problem does it solve? Link to related issues with closing keywords (Fixes #123, Closes #456). -->

## Changes

<!-- Optional: For larger PRs, break down the changes by category. Remove this section for small PRs. -->

### Added
-

### Changed
-

### Fixed
-

### Removed
-

## Test Plan

<!-- How did you verify this works? Include specific commands, manual testing steps, or checkboxes. -->

- [ ]

## Checklist

<!--
Follow CONTRIBUTING.md's pull request requirements and versioning/changelog policy:
https://github.com/existential-birds/daydream/blob/main/CONTRIBUTING.md#pull-request-requirements
https://github.com/existential-birds/daydream/blob/main/CONTRIBUTING.md#versioning-and-changelog
Keep every checklist item. Check satisfied applicable items and record evidence in Test Plan.
For an inapplicable conditional item, leave it unchecked and append "N/A: <reason>";
do not claim a check ran. Unresolved applicable items stay unchecked and block ready-for-review status.
-->

- [ ] Repository PR template followed and every required section completed
- [ ] Root and workflow gate `make check` completed successfully; any local actionlint skip due to an unavailable Docker daemon is reported in Test Plan
- [ ] Standalone RL checks pass via `make rl-check` when changing `rl/daydream_review` (deliberately outside `make check`)
- [ ] Run `$test-audit` on all test files created or updated in this PR (N/A if no test files changed)
- [ ] Documentation updated (if applicable)
- [ ] Semantic Versioning compatibility impact stated in the PR (or N/A with a reason)
- [ ] Notable changes recorded under `CHANGELOG.md`'s `[Unreleased]` in an appropriate Keep a Changelog category (or N/A with a reason)

## Additional Context

<!-- Optional: Screenshots, performance data, architecture diagrams, or other context that helps understand the change. Remove if not needed. -->
