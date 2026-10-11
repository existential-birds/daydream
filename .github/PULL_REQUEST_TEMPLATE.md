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

<!-- Verify before requesting review. -->

- [ ] Root and workflow checks pass locally via `make check` (requires Docker daemon for actionlint)
- [ ] Standalone RL checks pass via `make rl-check` when changing `rl/daydream_review` (deliberately outside `make check`)
- [ ] Run `$test-audit` on all test files created or updated in this PR (N/A if no test files changed)
- [ ] Documentation updated (if applicable)

## Additional Context

<!-- Optional: Screenshots, performance data, architecture diagrams, or other context that helps understand the change. Remove if not needed. -->
