# Relevant-root search replay (#1400)

Base: `2021615cc5a5ed5d10c78cf13ce10670c5f469f0`. macOS arm64,
Python 3.14.3, locked all-extras environment. One serial local replay, warm filesystem.
[Raw measurements](relevant-root-search.json) retain argv, scope, output and evidence.

## Archived question and comparison

Run `2fde90fd-841d-4799-9346-414d19a04be7`, Python step 17, asked whether five
command-display test names were referenced outside their three owner test files.
Its repository-root recursive grep explicitly reported a 30-second timeout.
Step 18 narrowed to `.github docs README.md Makefile scripts tests` and succeeded.
This replay uses the exact five names and file-extension filters from step 17;
the bounded treatment uses those named roots and five fixed-string CLI calls.
CLI startup is included, with no external model calls.

| Observation | Recursive grep | Bounded CLI |
|---|---:|---:|
| File scope | 8,851 extension-matching candidates | 578 enumerated and visited per question |
| Elapsed | 3.926 s | 2.152 s total across five calls |
| Stdout | 352 bytes | 1,644 bytes including JSON scope diagnostics |
| Timeouts / retries | 0 / 0 | 0 / 0 |
| Evidence hits | 3 | 3 |
| References outside owner files | 0 | 0 |

The broad candidate count comes from a separate filesystem enumeration; it is
not a syscall-instrumented read count. Bounded visited counts and bytes read are
production scanner observations. All bounded searches completed without skips
or truncation. Both approaches retained the same three source evidence hits and
answered the original question identically. The bounded JSON is larger because
it reports scope and incompleteness explicitly.

This local search-stage difference is 1.774 s; it is not a whole-review speedup
claim or a repeatable percentile estimate. The historical 30-second timeout did
not reproduce on this current checkout. Structure was slower than Python in the
archived run, so removing the Python timeout alone would not shorten that run's
critical path. Scope reduction and retained evidence are the deterministic claims.

## Test audit

- Real Git search fixtures protect tracked ignored/dotfiles, nonignored untracked
  source, literal unusual paths, nested roots and reasoned ignored expansion.
  Existing prompt tests did not execute an inventory or read the filesystem.
- Real linked-worktree `runner.run` invokes the production CLI from the provider
  boundary with no `rg` on PATH. It observes matches, skip diagnostics, expansion,
  successful review and persisted profile identity. Only the external provider
  is replaced. Same-path sibling content cannot satisfy the assertions.
- Count, byte and output caps use real files rather than patched clocks or limits.
  Escaping roots, metadata roots and a replaced tracked ancestor detect confinement
  regressions. Existing cwd prompt assertions alone missed these failures.
- Prompt composition matrices cover packaged and custom strategy text. The existing
  wholesale override keeper now checks host search guidance; it protects a distinct
  extension contract. Existing Claude guard cases admit search and inventory while
  retaining denial of mutation/chaining/redirection and unrelated Daydream verbs.
- Independent read-only audit found and prompted fixes for a metadata-symlink leak
  and Claude guard rejection of the recommended command. No tests were deleted.

`tests/test_review_profile_completeness.py` named in the issue no longer exists;
current profile, prompt, extension and runner owner suites provide its relevant
composition proof. See the PR Test Plan for commands and final gate results.
