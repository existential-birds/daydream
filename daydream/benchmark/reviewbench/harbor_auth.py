"""Native credentials stay in canonical stores; role settings and sessions are disposable.
Also runs standalone in the verifier on Python 3.11+."""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

REVIEWER_PI_BOOTSTRAP = '/opt/pi/node_modules/@earendil-works/pi-coding-agent/daydream-auth.mjs'


class SubscriptionAuthError(ValueError):
    """An intentionally credential-free subscription capability failure."""


def _read(path: Path) -> dict[str, Any]:
    # Native in-place writes may briefly overlap admission; refresh locks remain native.
    deadline = time.monotonic() + 1
    while True:
        try:
            if path.is_symlink() or not path.is_file():
                raise SubscriptionAuthError("subscription auth must be a regular private file")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise SubscriptionAuthError("invalid subscription auth object")
            return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise SubscriptionAuthError("subscription auth is unavailable or malformed") from None
            time.sleep(0.01)


def _require_text(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise SubscriptionAuthError("subscription OAuth credentials are incomplete")


def validate_subscription_home(home: Path) -> None:
    """Admit native refresh-capable OAuth without returning credential bytes."""
    value = _read(home / "auth.json")
    if value.get("auth_mode") != "chatgpt" or value.get("OPENAI_API_KEY"):
        raise SubscriptionAuthError("public ReviewBench requires ChatGPT subscription OAuth, without API keys")
    tokens = value.get("tokens")
    if not isinstance(tokens, dict):
        raise SubscriptionAuthError("subscription OAuth credentials are incomplete")
    for key in ("access_token", "refresh_token", "account_id", "id_token"):
        _require_text(tokens.get(key))
    try:
        payload = tokens["access_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if not isinstance(claims, dict) or type(claims.get("exp")) not in (int, float):
            raise ValueError
    except (ValueError, IndexError, UnicodeError):
        raise SubscriptionAuthError("subscription access token has no usable expiry") from None
    # Expired access is valid: the native client refreshes under its own lock.


def validate_judge_home(home: Path) -> None:
    """Admit only Pi's native subscription credential, never API fallback state."""
    value = _read(home / "auth.json")
    credential = value.get("openai-codex")
    if set(value) != {"openai-codex"} or not isinstance(credential, dict) or credential.get("type") != "oauth":
        raise SubscriptionAuthError("judge requires a dedicated Pi openai-codex OAuth login")
    for key in ("access", "refresh", "accountId"):
        _require_text(credential.get(key))
    if type(credential.get("expires")) not in (int, float):
        raise SubscriptionAuthError("judge subscription access token has no usable expiry")


def validate_pi_reviewer_home(home: Path, provider: str | None) -> None:
    """Require the selected native Pi credential without translating its format."""
    if provider != 'nous':
        raise SubscriptionAuthError('public Pi reviewer requires the configured nous provider')
    value = _read(home / 'auth.json')
    credential = value.get(provider)
    if set(value) != {provider} or not isinstance(credential, dict) or credential.get('type') != 'api_key':
        raise SubscriptionAuthError('selected native Pi provider credential is unavailable')
    _require_text(credential.get('key'))


def validate_credential_stores(reviewer: Path, judge: Path, *,
                               backend: str = 'codex', provider: str | None = None) -> None:
    """Require separate auth-only stores and independently signed-in reviewer/judge sessions."""
    if reviewer == judge or reviewer.is_relative_to(judge) or judge.is_relative_to(reviewer):
        raise SubscriptionAuthError("reviewer and judge credential stores must be separate")
    for home, role in ((reviewer, "reviewer"), (judge, "judge")):
        if home.is_symlink() or not home.is_dir():
            raise SubscriptionAuthError("dedicated credential directory is unavailable")
        # Legacy Codex lock files remain admissible but are no longer created or used.
        allowed = ({"auth.json", ".reviewer.lock"} if role == "reviewer" and backend == 'codex'
                   else {"auth.json", "auth.json.lock"})
        if any(path.name not in allowed or path.is_symlink() for path in home.iterdir()):
            raise SubscriptionAuthError("mount only dedicated credential state; remove configuration and sessions")
    validate_judge_home(judge)
    if backend == 'pi':
        validate_pi_reviewer_home(reviewer, provider)
        return
    if backend != 'codex':
        raise SubscriptionAuthError('unsupported native reviewer credential capability')
    validate_subscription_home(reviewer)
    reviewer_refresh = _read(reviewer / "auth.json")["tokens"]["refresh_token"]
    if reviewer_refresh == _read(judge / "auth.json")["openai-codex"]["refresh"]:
        raise SubscriptionAuthError("copied OAuth session; create independent reviewer and judge sign-ins")


@contextmanager
def reviewer_credentials(canonical_home: Path, role_home: Path, *,
                          backend: str = 'codex', provider: str | None = None) -> Iterator[Path]:
    """Link canonical credentials into a disposable role home without a review-wide lock."""
    if (canonical_home.resolve() == role_home.resolve() or canonical_home.is_symlink()
            or role_home.is_symlink() or not canonical_home.is_dir()):
        raise SubscriptionAuthError("canonical and private reviewer homes must be separate regular directories")
    role_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    role_home.chmod(0o700)
    if backend == 'pi':
        validate_pi_reviewer_home(canonical_home, provider)
        # The controlled Node bootstrap binds AuthStorage to the canonical path;
        # Pi keeps its own refresh locks there while settings/sessions are private.
        yield role_home
        return
    if backend != 'codex':
        raise SubscriptionAuthError('unsupported native reviewer credential capability')
    role_path = role_home / "auth.json"
    if role_path.exists() or role_path.is_symlink():
        raise SubscriptionAuthError("refusing stale reviewer subscription credentials")
    validate_subscription_home(canonical_home)
    role_path.symlink_to((canonical_home / "auth.json").resolve())
    try:
        yield role_home
    finally:
        role_path.unlink(missing_ok=True)


def subscription_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Remove alternative provider routes; token delivery uses only private files."""
    return {key: value for key, value in environment.items() if not (
        key.endswith(("_API_KEY", "_TOKEN", "_SECRET"))
        or key.startswith(("OPENAI_", "CODEX_", "CHATGPT_"))
        or key in {"PI_CODING_AGENT_DIR", "NODE_OPTIONS"}
    )}

def _run_judge(command: Sequence[str], environment: Mapping[str, str]) -> int:
    """Join the complete judge process group before removing private state."""
    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with subprocess.Popen(command, env=environment, start_new_session=True) as process:
            try:
                return process.wait()
            finally:
                if process.poll() is None:
                    with suppress(subprocess.TimeoutExpired, ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                # A leader can exit while a tool descendant ignores SIGTERM.
                # Terminate the remaining group before disposing of its state,
                # including descendants left behind after an ordinary exit.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unchanged upstream judge using its own persistent Pi credential store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("judge",))
    parser.add_argument("--canonical-home", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--pi-bootstrap", type=Path, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    if not command:
        parser.error("judge command is required after --")
    try:
        with tempfile.TemporaryDirectory(prefix="daydream-judge-") as temporary:
            home = Path(temporary)
            shutil.copyfile(args.models, home / "models.json")
            validate_judge_home(args.canonical_home)
            environment = subscription_environment(os.environ)
            environment["PI_CODING_AGENT_DIR"] = str(home)
            environment["DAYDREAM_RB_PI_AUTH_PATH"] = str((args.canonical_home / "auth.json").resolve())
            # verify.ts spawns eval.ts with process.execPath and fresh execArgv.
            # Controlled NODE_OPTIONS reaches that child too; ambient options
            # were scrubbed above. This path contains code, never credentials.
            environment["NODE_OPTIONS"] = "--import=" + json.dumps(str(args.pi_bootstrap.resolve(strict=True)))
            return _run_judge(command, environment)
    except (OSError, UnicodeError, SubscriptionAuthError):
        print("ReviewBench judge authentication or execution failed; "
              "no API fallback was attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
