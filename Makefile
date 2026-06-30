.PHONY: install lint typecheck test check hooks benchmark-report

install:
	uv sync

lint:
	uv run ruff check daydream tests

typecheck:
	uv run mypy daydream tests

test:
	uv run pytest -n auto

# Run all CI checks locally
check: lint typecheck test

# Install git hooks
hooks:
	ln -sf ../../scripts/hooks/pre-push .git/hooks/pre-push
	@echo "Pre-push hook installed"

# Generate the offline benchmark report from a benchmark run.
# BENCH = path to the code-review-benchmark offline/ dir (contains results/ + trajectories).
# Auto-discovers every results/<judge>/evaluations.json; override the daydream
# label or price card per run. Reads the corpus only; never modifies it.
# Each run writes a NEW self-contained folder under bench/benchmark-report/runs/
# (never overwrites a prior report); RUN names the folder, else a UTC timestamp +
# corpus fingerprint is used. `runs/latest` always points at the freshest report.
BENCH ?= ../code-review-benchmark/offline
DAYDREAM_TOOL ?= daydream-owl-alpha
PRICE_MODEL ?= glm-5.2
RUN ?=
benchmark-report:
	uv run python bench/benchmark-report/build.py "$(BENCH)/results" \
		--daydream-tool "$(DAYDREAM_TOOL)" --price-model "$(PRICE_MODEL)" \
		$(if $(RUN),--run-id "$(RUN)",)
	@echo "→ open bench/benchmark-report/runs/latest/index.html"
