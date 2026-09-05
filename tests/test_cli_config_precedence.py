"""Phase model/backend resolution precedence: CLI > config-file > default.

Drives the ``_resolved_model`` / ``_resolved_backend_name`` helpers split out of
``_resolve_backend`` so the decision is unit-testable without constructing a
Backend. The source tiers are (highest first): explicit per-phase field, global
``--model``/``--backend``, file-config phase override, file-config global, then
the terminal default (``PHASE_DEFAULT_MODELS`` table / ``"claude"``). There is no
environment-variable tier — ``DAYDREAM_MODEL``/``DAYDREAM_BACKEND`` are not read.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.backends import Backend
from daydream.backends.codex import CodexBackend
from daydream.config_file import DaydreamFileConfig
from daydream.runner import (
    RunConfig,
    _default_backend_name,
    _resolve_backend,
    _resolved_backend_name,
    _resolved_model,
    _resolved_reasoning_effort,
    _resolved_review_backend_name,
)


def test_model_precedence_cli_over_file_over_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fc = DaydreamFileConfig(model="file-model", backend=None, phases={"fix": {"model": "file-fix"}})
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_model(cfg, "fix") == "file-fix"        # file phase override, nothing higher
    cfg.model = "cli-global"
    assert _resolved_model(cfg, "fix") == "cli-global"      # global --model beats file phase override
    cfg.fix_model = "cli-fix"
    assert _resolved_model(cfg, "fix") == "cli-fix"         # explicit per-phase beats global --model
    cfg.model = None
    cfg.fix_model = None
    assert _resolved_model(cfg, "review") == "file-model"   # no phase override -> file global
    cfg2 = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_model(cfg2, "parse") == "claude-haiku-4-5"   # falls through to table default


def test_per_stack_review_and_arbiter_resolution(tmp_path: Path) -> None:
    """#168: the per-stack fan-out and arbiter resolve as independent phase keys.

    ``per_stack_review`` defaults to Sonnet (split off the Opus ``review`` tier)
    and ``arbiter`` to Opus, and a ``[tool.daydream.phases.per_stack_review]``
    file override resolves through ``_resolved_model`` without disturbing
    ``review``.
    """
    bare = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_model(bare, "per_stack_review") == "claude-sonnet-5"    # table default
    assert _resolved_model(bare, "arbiter") == "claude-opus-5"             # table default
    assert _resolved_model(bare, "review") == "claude-opus-5"              # unchanged

    fc = DaydreamFileConfig(
        model=None,
        backend=None,
        phases={"per_stack_review": {"model": "file-psr"}},
    )
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_model(cfg, "per_stack_review") == "file-psr"   # file phase override wins
    assert _resolved_model(cfg, "review") == "claude-opus-5"      # review untouched by the override
    assert _resolved_model(cfg, "arbiter") == "claude-opus-5"     # arbiter untouched


def test_backend_precedence_mirrors_model(tmp_path: Path) -> None:
    """Backend resolves through the same tiers as model, so the two stay symmetric."""
    fc = DaydreamFileConfig(model=None, backend="file-global", phases={"fix": {"backend": "file-fix"}})
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_backend_name(cfg, "fix") == "file-fix"        # file phase override, nothing higher
    cfg.backend = "cli-global"
    assert _resolved_backend_name(cfg, "fix") == "cli-global"      # global --backend beats file phase override
    cfg.fix_backend = "cli-fix"
    assert _resolved_backend_name(cfg, "fix") == "cli-fix"         # explicit per-phase beats global --backend
    cfg.backend = None
    cfg.fix_backend = None
    assert _resolved_backend_name(cfg, "review") == "file-global"  # no phase override -> file global
    cfg2 = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_backend_name(cfg2, "parse") == "claude"       # terminal fallback


def test_env_vars_are_not_a_precedence_tier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression guard: ``DAYDREAM_MODEL``/``DAYDREAM_BACKEND`` must be ignored.

    The env tier was removed to collapse precedence to ``CLI > config > default``.
    A config-file value must win over an ambient env var, not the reverse.
    """
    monkeypatch.setenv("DAYDREAM_MODEL", "env-model")
    monkeypatch.setenv("DAYDREAM_BACKEND", "env-backend")
    fc = DaydreamFileConfig(model="file-model", backend="file-backend", phases={})
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_model(cfg, "review") == "file-model"      # config beats env (env not read)
    assert _resolved_backend_name(cfg, "review") == "file-backend"
    # With no config either, falls straight through to the built-in defaults.
    cfg2 = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_model(cfg2, "parse") == "claude-haiku-4-5"
    assert _resolved_backend_name(cfg2, "parse") == "claude"


def test_reasoning_effort_precedence_cli_over_file(tmp_path: Path) -> None:
    """reasoning_effort mirrors model precedence: CLI global > file phase > file global > None."""
    fc = DaydreamFileConfig(
        model=None, backend=None, reasoning_effort="file-global",
        phases={"fix": {"reasoning_effort": "file-fix"}},
    )
    cfg = RunConfig(target=str(tmp_path), reasoning_effort=None, file_config=fc)
    assert _resolved_reasoning_effort(cfg, "fix") == "file-fix"       # file phase override, nothing higher
    cfg.reasoning_effort = "cli-global"
    assert _resolved_reasoning_effort(cfg, "fix") == "cli-global"     # global --reasoning-effort wins
    assert _resolved_reasoning_effort(cfg, "review") == "cli-global"  # applies to every phase
    cfg.reasoning_effort = None
    assert _resolved_reasoning_effort(cfg, "review") == "file-global"  # no phase override -> file global
    cfg2 = RunConfig(target=str(tmp_path), reasoning_effort=None, file_config=DaydreamFileConfig())
    # No CLI or file source -> the per-backend table is the terminal tier. The
    # default backend is claude, which is tiered for improve phases only.
    assert _resolved_reasoning_effort(cfg2, "plan_write") == "max"
    # A phase with no entry for this backend still resolves to None, leaving
    # the driver's ambient default in place.
    assert _resolved_reasoning_effort(cfg2, "review") is None
    assert _resolved_reasoning_effort(cfg2, "not_a_phase") is None


def _codex_backend(
    cfg: RunConfig,
    phase: str,
    cache: dict[tuple[str, str | None, str | None], Backend] | None = None,
) -> CodexBackend:
    """Resolve ``phase`` and narrow to the concrete Codex backend under test."""
    backend = _resolve_backend(cfg, phase, cache)
    assert isinstance(backend, CodexBackend), f"{phase} should resolve to a CodexBackend, got {type(backend)}"
    return backend


def test_phase_default_effort_table_is_the_lowest_precedence_tier(tmp_path: Path) -> None:
    """``PHASE_DEFAULT_EFFORT`` supplies a per-phase effort when no higher tier does.

    Asserts on the effort the resolved ``CodexBackend`` actually carries — that
    is the value forwarded to the ``codex`` CLI — not on the helper alone.
    """
    bare = RunConfig(target=str(tmp_path), backend="codex", model=None, file_config=DaydreamFileConfig())
    # (a) no override anywhere -> the phase's table default, tiered per phase.
    assert _codex_backend(bare, "arbiter").reasoning_effort == "xhigh"
    assert _codex_backend(bare, "review").reasoning_effort == "high"
    assert _codex_backend(bare, "fix").reasoning_effort == "medium"
    assert _codex_backend(bare, "parse").reasoning_effort == "low"

    # (b) a config-file phase override beats the table, and only for that phase.
    fc = DaydreamFileConfig(model=None, backend=None, phases={"parse": {"reasoning_effort": "xhigh"}})
    cfg = RunConfig(target=str(tmp_path), backend="codex", model=None, file_config=fc)
    assert _codex_backend(cfg, "parse").reasoning_effort == "xhigh"
    assert _codex_backend(cfg, "fix").reasoning_effort == "medium"  # still the table default

    # (c) --reasoning-effort beats both the file override and the table, everywhere.
    cfg.reasoning_effort = "low"
    assert _codex_backend(cfg, "parse").reasoning_effort == "low"
    assert _codex_backend(cfg, "arbiter").reasoning_effort == "low"

    # A config-file *global* also outranks the table but loses to the CLI.
    fc_global = DaydreamFileConfig(model=None, backend=None, reasoning_effort="high")
    cfg2 = RunConfig(target=str(tmp_path), backend="codex", model=None, file_config=fc_global)
    assert _codex_backend(cfg2, "parse").reasoning_effort == "high"


def test_claude_effort_is_improve_only(tmp_path: Path) -> None:
    """Claude is tiered for improve phases and left alone for deep-review ones.

    Deep phases have no Claude entry, so they resolve to None and the CLI keeps
    applying the ambient default it always had — improve tiering must not move
    deep-review behavior.
    """
    cfg = RunConfig(target=str(tmp_path), backend="claude", model=None, file_config=DaydreamFileConfig())
    assert _resolved_reasoning_effort(cfg, "arbiter") is None
    assert _resolved_reasoning_effort(cfg, "review") is None
    assert getattr(_resolve_backend(cfg, "arbiter"), "reasoning_effort") is None
    assert _resolved_reasoning_effort(cfg, "plan_write") == "max"
    assert getattr(_resolve_backend(cfg, "plan_write"), "reasoning_effort") == "max"


def test_backend_cache_splits_on_table_default_effort(tmp_path: Path) -> None:
    """Two codex phases sharing a model but not an effort must not share a backend.

    ``review`` and ``arbiter`` both default to ``gpt-5.6-sol`` but to ``high``
    and ``xhigh`` respectively, so the cache key must keep them distinct.
    """
    cfg = RunConfig(target=str(tmp_path), backend="codex", model=None, file_config=DaydreamFileConfig())
    cache: dict[tuple[str, str | None, str | None], Backend] = {}
    review = _codex_backend(cfg, "review", cache)
    arbiter = _codex_backend(cfg, "arbiter", cache)
    assert review.model == arbiter.model == "gpt-5.6-sol"
    assert review is not arbiter
    assert (review.reasoning_effort, arbiter.reasoning_effort) == ("high", "xhigh")
    assert _resolve_backend(cfg, "review", cache) is review  # still cached per (backend, model, effort)


def test_pi_native_model_is_not_replaced_by_glm_fallback(tmp_path: Path) -> None:
    """Pi's own default remains available when daydream has no model setting."""
    cfg = RunConfig(target=str(tmp_path), backend="pi", model=None)
    assert _resolved_model(cfg, "review") is None

    cfg.model = "custom-model"
    assert _resolved_model(cfg, "review") == "custom-model"


def test_default_backend_is_phase_agnostic(tmp_path: Path) -> None:
    """#647: the general default backend ignores per-phase review overrides."""
    empty = DaydreamFileConfig()
    # A review override must never leak into the general default.
    cfg = RunConfig(target=str(tmp_path), backend="claude", review_backend="codex", file_config=empty)
    assert _default_backend_name(cfg) == "claude"
    # Global CLI --backend is the source when set.
    cfg.backend = "codex"
    assert _default_backend_name(cfg) == "codex"
    # File-config global when no CLI backend.
    cfg.backend = None
    cfg.file_config = DaydreamFileConfig(backend="file-backend")
    assert _default_backend_name(cfg) == "file-backend"
    # Terminal fallback with no config.
    cfg.file_config = empty
    assert _default_backend_name(cfg) == "claude"


def test_review_backend_override_is_none_when_unset(tmp_path: Path) -> None:
    """#647: review_backend is None unless a review-specific override is set."""
    empty = DaydreamFileConfig()
    # No override anywhere -> None (not the general backend).
    bare = RunConfig(target=str(tmp_path), backend="claude", review_backend=None, file_config=empty)
    assert _resolved_review_backend_name(bare) is None
    # CLI review_backend override.
    cfg = RunConfig(target=str(tmp_path), backend="claude", review_backend="codex", file_config=empty)
    assert _resolved_review_backend_name(cfg) == "codex"
    # File-config review-phase override.
    fc = DaydreamFileConfig(phases={"review": {"backend": "file-codex"}})
    cfg2 = RunConfig(target=str(tmp_path), backend="claude", review_backend=None, file_config=fc)
    assert _resolved_review_backend_name(cfg2) == "file-codex"
    # File-config global only (no review override) -> still None.
    fc_global = DaydreamFileConfig(backend="codex")
    cfg3 = RunConfig(target=str(tmp_path), backend=None, review_backend=None, file_config=fc_global)
    assert _resolved_review_backend_name(cfg3) is None


def test_trajectory_hub_repo_flag_reaches_runconfig(tmp_path: Path) -> None:
    """The ``--trajectory-hub-repo`` shared flag must reach RunConfig via every builder.

    Traces construction paths for both flows that read shared args: deep
    (``_parse_args``) and improve (``_parse_improve_args``).
    """
    from daydream.cli import _parse_args, _parse_improve_args

    target = str(tmp_path)

    deep = _parse_args([target, "--trajectory-hub-repo", "acme/dd-trajectories"])
    assert deep.trajectory_hub_repo == "acme/dd-trajectories"

    improve = _parse_improve_args(
        ["improve", target, "--trajectory-hub-repo", "acme/dd-trajectories"]
    )
    assert improve.trajectory_hub_repo == "acme/dd-trajectories"


def test_test_command_precedence_cli_over_file_config(tmp_path: Path) -> None:
    """Issue #726: the canonical test command resolves CLI > file config.

    The CLI ``--test-command`` flag overrides the ``test_command`` config key
    (merged from ``.daydream.toml`` over ``[tool.daydream]``); when both are
    unset, resolution fails closed with an actionable error.
    """
    from types import SimpleNamespace

    from daydream.cli import _parse_args
    from daydream.config_file import load_file_config
    from daydream.test_execution import MissingTestCommandError, canonical_test_command

    target = str(tmp_path)

    # File config loads the key from both sources, dotfile winning.
    (tmp_path / "pyproject.toml").write_text('[tool.daydream]\ntest_command = "make test"\n')
    (tmp_path / ".daydream.toml").write_text('test_command = "pytest -q"\n')
    fc = load_file_config(tmp_path)
    assert fc.test_command == "pytest -q"

    # CLI flag wins over the file value.
    cfg = _parse_args([target, "--test-command", "uv run pytest"])
    assert cfg.test_command == "uv run pytest"
    assert canonical_test_command(fc, cfg) == ["uv", "run", "pytest"]

    # No CLI flag -> file value applies, shell-word-split.
    cfg_noflag = _parse_args([target])
    assert canonical_test_command(fc, cfg_noflag) == ["pytest", "-q"]

    # Nothing set anywhere -> fail closed with a diagnostic naming key + sources.
    empty_fc = load_file_config(tmp_path / "does-not-exist")
    with pytest.raises(MissingTestCommandError) as e:
        canonical_test_command(empty_fc, SimpleNamespace(test_command=None))
    msg = str(e.value)
    assert "test_command" in msg
    assert "tool.daydream" in msg
    assert "--test-command" in msg
