.PHONY: install lint typecheck test check lockcheck hooks deadcode

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

# Fail if uv.lock has drifted from pyproject.toml (e.g. a release bumped the
# version but forgot `uv lock`). Read-only: `--check` never heals the lock, and
# this must run BEFORE any `uv run`/`uv sync` step, which would silently re-lock.
lockcheck:
	uv lock --check

# Run all CI checks locally (lockcheck first — before uv heals the lock)
check: lockcheck deadcode lint typecheck test

# Install git hooks
hooks:
	ln -sf "$$(git rev-parse --show-toplevel)/scripts/hooks/pre-push" \
	       "$$(git rev-parse --git-path hooks/pre-push)"
	@echo "Pre-push hook installed"
