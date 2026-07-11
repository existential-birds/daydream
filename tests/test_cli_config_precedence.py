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

from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig, _resolved_backend_name, _resolved_model


def test_model_precedence_cli_over_file_over_table(monkeypatch, tmp_path: Path) -> None:
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
    assert _resolved_model(bare, "per_stack_review") == "claude-sonnet-4-6"  # table default
    assert _resolved_model(bare, "arbiter") == "claude-opus-4-8"             # table default
    assert _resolved_model(bare, "review") == "claude-opus-4-8"              # unchanged

    fc = DaydreamFileConfig(
        model=None,
        backend=None,
        phases={"per_stack_review": {"model": "file-psr"}},
    )
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_model(cfg, "per_stack_review") == "file-psr"   # file phase override wins
    assert _resolved_model(cfg, "review") == "claude-opus-4-8"      # review untouched by the override
    assert _resolved_model(cfg, "arbiter") == "claude-opus-4-8"     # arbiter untouched


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


def test_env_vars_are_not_a_precedence_tier(monkeypatch, tmp_path: Path) -> None:
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


def test_pi_native_model_is_not_replaced_by_glm_fallback(tmp_path: Path) -> None:
    """Pi's own default remains available when daydream has no model setting."""
    cfg = RunConfig(target=str(tmp_path), backend="pi", model=None)
    assert _resolved_model(cfg, "review") is None

    cfg.model = "custom-model"
    assert _resolved_model(cfg, "review") == "custom-model"
