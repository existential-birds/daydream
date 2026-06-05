"""Phase-model resolution precedence: CLI > env > config-file > table default.

Drives the ``_resolved_model`` helper split out of ``_resolve_backend`` so the
model decision is unit-testable without constructing a Backend. The source
tiers are (highest first): explicit per-phase field, global ``--model``,
``DAYDREAM_MODEL`` env, file-config phase override, file-config global, then the
``PHASE_DEFAULT_MODELS`` table default.
"""

from __future__ import annotations

from pathlib import Path

from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig, _resolved_model


def test_precedence_cli_over_env_over_file_over_table(monkeypatch, tmp_path: Path) -> None:
    fc = DaydreamFileConfig(model="file-model", backend=None, phases={"fix": {"model": "file-fix"}})
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_model(cfg, "fix") == "file-fix"        # file phase override, nothing higher
    cfg.model = "cli-global"
    assert _resolved_model(cfg, "fix") == "cli-global"      # global --model beats file phase override
    cfg.fix_model = "cli-fix"
    assert _resolved_model(cfg, "fix") == "cli-fix"         # explicit per-phase beats global --model
    cfg.model = None
    cfg.fix_model = None
    monkeypatch.setenv("DAYDREAM_MODEL", "env-model")
    assert _resolved_model(cfg, "review") == "env-model"    # env beats file
    monkeypatch.delenv("DAYDREAM_MODEL")
    cfg2 = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_model(cfg2, "parse") == "claude-haiku-4-5"   # falls through to table default
