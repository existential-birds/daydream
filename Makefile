.PHONY: install lint typecheck test actionlint rl-check check lockcheck hooks deadcode

install:
	# All extras so `make check` runs the full suite (benchmark objective tests
	# need the harbor package from the benchmark extra).
	uv sync --all-extras

lint:
	uv run ruff check daydream tests

typecheck:
	uv run mypy daydream tests

# Whole-project dead-code detection (#935). Both projects' scans live in their
# own [tool.vulture]; pass --config explicitly so each project's pyproject.toml
# is resolved. Scan pkg+tests together (one process) so test references keep
# package symbols alive. Exit 0 == clean.
deadcode:
	uv run vulture --config pyproject.toml daydream tests
	cd rl/daydream_review_v1 && uv run vulture --config pyproject.toml daydream_review_v1 tests

test:
	uv run pytest -n auto

# Docker-backed actionlint over every workflow the project ships (repo-owned
# plus all template files, nested included). Image is digest-pinned exactly as
# .github/workflows/ci.yml does; mounting $(CURDIR) at /repo so the selectors
# expand to the same set CI checks. Enforced whenever a Docker daemon is
# available; skipped with a note when it is not, so the local/pre-push gate is
# not a hard Docker-daemon requirement (CI always runs it).
actionlint:
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
	  docker run --rm \
	    -v "$(CURDIR)":/repo -w /repo \
	    rhysd/actionlint:1.7.7@sha256:887a259a5a534f3c4f36cb02dca341673c6089431057242cdc931e9f133147e9 \
	    -color .github/workflows/*.yml daydream/templates/workflows/*.yml daydream/templates/workflows/single/*.yml; \
	else \
	  echo "actionlint skipped: Docker daemon is not available"; \
fi

# Standalone RL project gates, run from its own directory via per-line cd
# (each recipe line is its own shell). Mirrors ci.yml's rl-check job, including
# its 'Configure git identity' step: the suite's negative-gate tests commit into
# throwaway fixtures with no per-repo identity, so without a global identity a
# contributor machine would fail where CI (which sets it) passes. The identity
# is injected as rl-check-scoped process environment, never via
# `git config --global`, which would silently overwrite the invoking user's
# own identity (a side effect no local gate may cause).
rl-check: export GIT_AUTHOR_NAME = daydream CI
rl-check: export GIT_AUTHOR_EMAIL = ci@daydream.invalid
rl-check: export GIT_COMMITTER_NAME = daydream CI
rl-check: export GIT_COMMITTER_EMAIL = ci@daydream.invalid

rl-check:
	cd rl/daydream_review_v1 && uv lock --check
	cd rl/daydream_review_v1 && uv sync
	cd rl/daydream_review_v1 && uv run ruff check .
	cd rl/daydream_review_v1 && uv run mypy daydream_review_v1 tests
	cd rl/daydream_review_v1 && uv run pytest

# Fail if uv.lock has drifted from pyproject.toml (e.g. a release bumped the
# version but forgot `uv lock`). Read-only: `--check` never heals the lock, and
# this must run BEFORE any `uv run`/`uv sync` step, which would silently re-lock.
lockcheck:
	uv lock --check

# Run all CI checks locally: lockcheck and the root uv sync --all-extras install
# step first (both before any uv run heals the lock), matching ci.yml's check job.
check: lockcheck deadcode install lint typecheck test actionlint rl-check

# Install git hooks
hooks:
	ln -sf "$$(git rev-parse --show-toplevel)/scripts/hooks/pre-push" \
	       "$$(git rev-parse --git-path hooks/pre-push)"
	@echo "Pre-push hook installed"
