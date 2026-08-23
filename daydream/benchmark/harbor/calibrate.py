"""Calibration gate for configured semantic-match judges.

Drives the exact packaged Harbor judge path (``score_review.judge_pairs``,
loaded via importlib from ``templates/tests/`` so the bare ``import
verifier_core`` resolves to the sibling copy) against a fixed 24-pair labeled
fixture, three times per pair (72 judge calls). Computes a three-part pass
gate — majority correctness, balanced accuracy, and per-pair retained-edge
threshold-flip stability — and, on pass, records a private deterministic
invalidation-aware receipt at ``<workspace>/runtime/calibration-receipt.json``.
Measuring, never tuning: no weights or thresholds are derived here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from daydream.benchmark import storage
from daydream.benchmark.schema import BenchmarkManifest

_HARBOR = Path(__file__).parent
_TEMPLATES = _HARBOR / "templates"
_CALIBRATION_DIR = _HARBOR / "calibration"


def _load_template_asset(path: Path, name: str) -> Any:
    """Load a non-package template asset so a bare sibling import resolves to it."""
    sys.path.insert(0, str(path.parent))  # bare `import verifier_core` -> sibling template copy
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_judge_template() -> Any:
    """Load the packaged ``score_review.py`` judge template as ``score_review``."""
    return _load_template_asset(_TEMPLATES / "tests" / "score_review.py", "score_review")


def _load_fixture() -> list[dict[str, Any]]:
    """Read and return the fixed 24-pair calibration fixture."""
    path = _CALIBRATION_DIR / "pairs.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"could not parse calibration fixture at {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"calibration fixture at {path} must be a JSON list")
    return data


def _build_calibration_client(env: dict[str, Any], *, http: Any = None) -> Any:
    """Build a judge client threading an injectable ``http=`` seam.

    Mirrors ``score_review._build_client``'s provider branch exactly (same
    ``DAYDREAM_JUDGE_*`` reads, same fail-closed provider set, same
    ``_effective_allowlist``/``_validate_base_url``/``resolve_base_url``
    helpers from the loaded module) and additionally passes ``http=http`` into
    the ``AnthropicJudgeClient`` / ``OpenAIJudgeClient`` constructor. With
    ``http=None`` it is behaviorally identical to ``_build_client`` (a real
    httpx client).
    """
    sr = _load_judge_template()
    provider = env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic"
    model = env.get("DAYDREAM_JUDGE_MODEL")
    api_key = env.get("DAYDREAM_JUDGE_API_KEY")
    if not model or not api_key:
        raise sr.VerifierError("missing DAYDREAM_JUDGE_MODEL or DAYDREAM_JUDGE_API_KEY")
    if provider not in {"anthropic", "openai-compatible"}:
        raise sr.VerifierError(
            f"unsupported DAYDREAM_JUDGE_PROVIDER '{provider}'; expected anthropic or openai-compatible"
        )
    if provider == "anthropic":
        allowlist = sr._effective_allowlist(sr._ANTHROPIC_MESSAGES_URL, env)
        sr._validate_base_url(sr._ANTHROPIC_MESSAGES_URL, allowlist)
        return sr.AnthropicJudgeClient(
            api_key, model, http=http, allowlist=allowlist
        )
    base_url = sr.resolve_base_url(api_key, env.get("DAYDREAM_JUDGE_BASE_URL"))
    allowlist = sr._effective_allowlist(base_url, env)
    initial_url = base_url.rstrip("/") + "/chat/completions"
    sr._validate_base_url(initial_url, allowlist)
    base_url = sr.resolve_base_url(api_key, env.get("DAYDREAM_JUDGE_BASE_URL"))
    allowlist = sr._effective_allowlist(base_url, env)
    initial_url = base_url.rstrip("/") + "/chat/completions"
    sr._validate_base_url(initial_url, allowlist)
    return sr.OpenAIJudgeClient(
        api_key, model, base_url=base_url, http=http, allowlist=allowlist
    )


def _judge_host_from_env(env: dict[str, Any]) -> str:
    """Return the normalized-lowercase judge host for ``env``.

    The anthropic provider (or absent -> anthropic default) always routes to
    ``api.anthropic.com``; the openai-compatible provider resolves its base URL
    via the packaged ``resolve_base_url`` and returns that URL's host.
    """
    sr = _load_judge_template()
    provider = env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic"
    if provider == "anthropic":
        return "api.anthropic.com"
    base_url = sr.resolve_base_url(
        env.get("DAYDREAM_JUDGE_API_KEY") or "", env.get("DAYDREAM_JUDGE_BASE_URL")
    )
    return str(urllib.parse.urlsplit(base_url).hostname or "").lower()


def _load_workspace_allowlist(workspace: Path) -> list[str]:
    """Read ``<workspace>/benchmark.yaml`` and return its judge allowlist.

    Validates the manifest strictly via ``BenchmarkManifest.model_validate``;
    a malformed or missing manifest raises the project's existing workspace
    error (``WorkspaceCorrupt``) — never a silent default.
    """
    try:
        raw = storage.load_yaml_strict(workspace / "benchmark.yaml")
    except storage.WorkspaceCorrupt:
        raise
    try:
        manifest = BenchmarkManifest.model_validate(raw)
    except Exception as exc:
        raise storage.WorkspaceCorrupt(
            f"{workspace}: invalid benchmark.yaml: {exc}"
        ) from exc
    return list(manifest.privacy.judge_allowed_hosts)


def _validate_workspace_host(allowlist: list[str] | set[str], host: str) -> None:
    """Fail closed when ``host`` is not in the workspace judge host allowlist."""
    if host not in allowlist:
        raise ValueError(
            f"judge host {host!r} is not in the workspace judge_allowed_hosts allowlist"
        )
