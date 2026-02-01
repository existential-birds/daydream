.PHONY: install lint typecheck test check hooks

install:
	uv sync

lint:
	uv run ruff check daydream

typecheck:
	uv run mypy daydream

test:
	uv run pytest -v

# Run all CI checks locally
check: lint typecheck test

# Install git hooks
hooks:
	ln -sf ../../scripts/hooks/pre-push .git/hooks/pre-push
	@echo "Pre-push hook installed"
