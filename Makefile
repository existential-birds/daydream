.PHONY: install lint typecheck actionlint rl-check check lockcheck hooks

install:
	# All extras so `make check` runs the full suite (benchmark objective tests
	# need the harbor package from the benchmark extra).
	uv sync --all-extras

lint:
	uv run ruff check daydream tests

typecheck:
	uv run mypy daydream tests

test:
	uv run pytest -n auto

# Docker-backed actionlint over every workflow the project ships (repo-owned
# plus all template files, nested included). Image is digest-pinned exactly as
# .github/workflows/ci.yml does; mounting $(CURDIR) at /repo so the selectors
# expand to the same set CI checks. Requires a running Docker daemon.
actionlint:
	docker run --rm \
	  -v "$(CURDIR)":/repo -w /repo \
	  rhysd/actionlint:1.7.7@sha256:887a259a5a534f3c4f36cb02dca341673c6089431057242cdc931e9f133147e9 \
	  -color .github/workflows/*.yml daydream/templates/workflows/*.yml daydream/templates/workflows/single/*.yml

# Standalone RL project gates, run from its own directory via per-line cd
# (each recipe line is its own shell).
rl-check:
	cd rl/daydream_review_v1 && uv lock --check
	cd rl/daydream_review_v1 && uv run ruff check .
	cd rl/daydream_review_v1 && uv run mypy daydream_review_v1 tests
	cd rl/daydream_review_v1 && uv run pytest

# Fail if uv.lock has drifted from pyproject.toml (e.g. a release bumped the
# version but forgot `uv lock`). Read-only: `--check` never heals the lock, and
# this must run BEFORE any `uv run`/`uv sync` step, which would silently re-lock.
lockcheck:
	uv lock --check

# Run all CI checks locally (lockcheck first — before uv heals the lock)
check: lockcheck lint typecheck test actionlint rl-check

# Install git hooks
hooks:
	ln -sf "$$(git rev-parse --show-toplevel)/scripts/hooks/pre-push" \
	       "$$(git rev-parse --git-path hooks/pre-push)"
	@echo "Pre-push hook installed"
