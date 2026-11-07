.PHONY: install lint typecheck test test-scripts actionlint check static lockcheck package-check hooks deadcode coverage-report check-naming

# One worker per logical CPU: `auto` counts physical cores, which halves the
# workers on hyperthreaded cloud runners. CI shards set their own count.
PYTEST_WORKERS ?= logical
# Optional reporting on this invocation only; nested pytest subprocesses do not
# inherit these flags through PYTEST_ADDOPTS.
PYTEST_REPORT_ARGS ?=

install:
	# All extras so `make check` runs every test with the dependencies it exercises.
	uv sync --all-extras

package-check:
	bash scripts/check-packaged-runtimes.sh

lint:
	uv run ruff check daydream tests scripts/ci_*.py

typecheck:
	uv run mypy daydream tests scripts/ci_*.py

# Whole-project dead-code detection (#935), configured in [tool.vulture]. Scan
# pkg+tests together (one process) so test references keep package symbols
# alive. Exit 0 == clean.
deadcode:
	uv run vulture --config pyproject.toml daydream tests

# Coverage flags live here (not in global addopts) so targeted runs like
# `uv run pytest tests/foo.py` stay plain and never trip the floor enforced by
# [tool.coverage.report] fail_under in pyproject.toml (#336).
# CI reaches these flags through `make test`; xdist parallelism stays via
# the PYTEST_WORKERS override (logical CPUs locally). No --no-cov-on-fail so coverage.xml survives
# a failed run for the coverage-report / CI artifact upload.
test:
	uv run pytest -n $(PYTEST_WORKERS) --cov --cov-branch --cov-report=term-missing --cov-report=xml $(PYTEST_REPORT_ARGS)

# Part of the gate's parallel `static` step; CI runs it in its static job.
test-scripts:
	uv run python -m unittest discover -s .agents/skills/daydream-trace-analysis/scripts -p 'test_*.py'

# Machine-readable coverage report for humans who want the XML locally; the
# terminal term-missing report + coverage floor enforcement already happen inside
# `test` via its coverage flags (#932). docs/coverage.md documents the full flow.
coverage-report:
	@test -f coverage.xml || { echo "no coverage.xml — run make test first"; exit 1; }
	@echo "coverage.xml present (uploaded as a CI artifact on the check job)."

# actionlint over every workflow the project ships (repo-owned plus all
# template files, nested included), pinned through the uv dev group.
actionlint:
	uv run actionlint -color .github/workflows/*.yml daydream/templates/workflows/*.yml daydream/templates/workflows/single/*.yml

# Fail if uv.lock has drifted from pyproject.toml (e.g. a release bumped the
# version but forgot `uv lock`). Read-only: `--check` never heals the lock, and
# this must run BEFORE any `uv run`/`uv sync` step, which would silently re-lock.
lockcheck:
	uv lock --check

# Run all CI checks locally: lockcheck and the root uv sync --all-extras install
# step first (both before any uv run heals the lock), matching ci.yml's check job.
check: lockcheck install static test coverage-report

# Independent static checks run in parallel, so the gate pays for the slowest
# one rather than their sum. Run `make check` itself without -j so pytest starts
# only after them: a loaded host starves its real-Git timing windows.
STATIC_CHECKS = lint deadcode typecheck package-check actionlint check-naming test-scripts

static:
	$(MAKE) -j $(words $(STATIC_CHECKS)) $(STATIC_CHECKS)

# Naming grep-gate (#1093): fails when a project-owned versioned identifier
# reappears anywhere in tracked files. External tool-protocol version strings
# are allowlisted inline in the script. Runs as part of `make check` so the
# local and CI gates both enforce the neutral-naming decision.
check-naming:
	bash scripts/check-naming.sh

# Install git hooks
hooks:
	ln -sf "$$(git rev-parse --show-toplevel)/scripts/hooks/pre-push" \
	       "$$(git rev-parse --git-path hooks/pre-push)"
	ln -sf "$$(git rev-parse --show-toplevel)/scripts/hooks/pre-commit" \
	       "$$(git rev-parse --git-path hooks/pre-commit)"
	@echo "Pre-push hook installed"
	@echo "Pre-commit hook installed"
