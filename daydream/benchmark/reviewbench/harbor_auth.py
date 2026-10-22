"""Role-private ChatGPT subscription credentials for the public Harbor profile.

This file is also copied verbatim into the verifier image and runs on Python 3.11+.
Each runtime owns an independently signed-in persistent store. Native Codex
invocations serialize under a reviewer-only lock; Pi owns its refresh locking
at one canonical path, including shared sidecars. Only credential state persists.
Credential contents are never configuration, fingerprints, or retained evidence.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from daydream.json_utils import atomic_write_bytes, publish_staged_directory
from daydream.redaction import redact_value


class SubscriptionAuthError(ValueError):
    """An intentionally credential-free subscription capability failure."""


def _read(path: Path) -> dict[str, Any]:
    # Both native stores write in place. Read-only admission and redaction
    # snapshots can overlap another process's locked truncate/write. Tolerate
    # that short window without adding a second refresh-lock implementation.
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


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SubscriptionAuthError("subscription OAuth credentials are incomplete")
    return value


def _claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        value = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, IndexError, UnicodeError):
        raise SubscriptionAuthError("subscription access token has no usable expiry") from None


def _codex_auth(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("auth_mode") != "chatgpt" or value.get("OPENAI_API_KEY"):
        raise SubscriptionAuthError("public ReviewBench requires ChatGPT subscription OAuth, without API keys")
    tokens = value.get("tokens")
    if not isinstance(tokens, dict):
        raise SubscriptionAuthError("subscription OAuth credentials are incomplete")
    for key in ("access_token", "refresh_token", "account_id", "id_token"):
        _text(tokens.get(key))
    claims = _claims(tokens["access_token"])
    if type(claims.get("exp")) not in (int, float):
        raise SubscriptionAuthError("subscription access token has no usable expiry")
    # Expired access is admissible only with refresh capability; the owning native
    # client performs refresh under the reviewer lock, including failure cases.
    return value


def validate_subscription_home(home: Path) -> None:
    """Validate OAuth capability without returning or recording credential bytes."""
    _codex_auth(_read(home / "auth.json"))


def validate_judge_home(home: Path) -> None:
    """Admit only Pi's native subscription credential, never API fallback state."""
    value = _read(home / "auth.json")
    credential = value.get("openai-codex")
    if set(value) != {"openai-codex"} or not isinstance(credential, dict) or credential.get("type") != "oauth":
        raise SubscriptionAuthError("judge requires a dedicated Pi openai-codex OAuth login")
    for key in ("access", "refresh", "accountId"):
        _text(credential.get(key))
    if type(credential.get("expires")) not in (int, float):
        raise SubscriptionAuthError("judge subscription access token has no usable expiry")


def validate_credential_stores(reviewer: Path, judge: Path) -> None:
    """Require disjoint auth-only mounts and reject an obviously copied session.

    Different tokens cannot prove independent sign-ins; setup must create them.
    Lock sidecars belong beside their auth file and must survive invocations.
    """
    if reviewer == judge or reviewer.is_relative_to(judge) or judge.is_relative_to(reviewer):
        raise SubscriptionAuthError("reviewer and judge credential stores must be separate")
    for home, role in ((reviewer, "reviewer"), (judge, "judge")):
        if home.is_symlink() or not home.is_dir():
            raise SubscriptionAuthError("dedicated credential directory is unavailable")
        allowed = {"auth.json", ".reviewer.lock"} if role == "reviewer" else {"auth.json", "auth.json.lock"}
        if any(path.name not in allowed or path.is_symlink() for path in home.iterdir()):
            raise SubscriptionAuthError("mount only dedicated credential state; remove configuration and sessions")
    validate_subscription_home(reviewer)
    validate_judge_home(judge)
    reviewer_refresh = _read(reviewer / "auth.json")["tokens"]["refresh_token"]
    if reviewer_refresh == _read(judge / "auth.json")["openai-codex"]["refresh"]:
        raise SubscriptionAuthError("copied OAuth session; create independent reviewer and judge sign-ins")


@contextmanager
def reviewer_credentials(canonical_home: Path, role_home: Path) -> Iterator[Path]:
    """Serialize native Codex invocations and link only the persistent credential.

    Codex 0.163.0 writes its file store in place. No credential copies, conversion,
    reconciliation or recovery markers are needed. Configuration and sessions
    remain disposable. The OS releases the lock on failure or process death.
    """
    if (canonical_home.resolve() == role_home.resolve() or canonical_home.is_symlink()
            or role_home.is_symlink() or not canonical_home.is_dir()):
        raise SubscriptionAuthError("canonical and private reviewer homes must be separate regular directories")
    role_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    role_home.chmod(0o700)
    role_path = role_home / "auth.json"
    if role_path.exists() or role_path.is_symlink():
        raise SubscriptionAuthError("refusing stale reviewer subscription credentials")
    descriptor = os.open(canonical_home / ".reviewer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        owner, current = canonical_home.stat(), os.fstat(lock.fileno())
        if (current.st_uid, current.st_gid) != (owner.st_uid, owner.st_gid):
            os.fchown(lock.fileno(), owner.st_uid, owner.st_gid)
        fcntl.flock(lock, fcntl.LOCK_EX)
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
        or key in {"OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID",
                   "PI_CODING_AGENT_DIR", "CODEX_HOME", "NODE_OPTIONS"}
    )}



_SESSION_NAME = re.compile(
    r"[0-9]{4}(?:-[0-9]{2}){2}T[0-9]{2}(?:-[0-9]{2}){2}-[0-9]{3}Z_"
    r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\.jsonl"
)


def _credential_values(auth: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key, value in auth.items():
        if isinstance(value, dict):
            values.update(_credential_values(value))
        elif key in {"access", "refresh", "access_token", "refresh_token", "id_token", "account_id", "accountId"}:
            if isinstance(value, str) and value:
                values.add(value)
    return values



def _embedded_auth_values(value: Any) -> set[str]:
    """Recognize our two auth-store formats even inside JSON tool-result text."""
    if isinstance(value, dict):
        values = (_credential_values(value) if value.get("type") == "oauth"
                  or value.get("auth_mode") == "chatgpt" else set())
        for item in value.values():
            values.update(_embedded_auth_values(item))
        return values
    if isinstance(value, list):
        return set().union(*(_embedded_auth_values(item) for item in value))
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return _embedded_auth_values(json.loads(value))
        except json.JSONDecodeError:
            pass
    return set()


def _replace_credentials(value: Any, credentials: set[str]) -> Any:
    if isinstance(value, str):
        for credential in sorted(credentials, key=len, reverse=True):
            value = value.replace(credential, "[REDACTED_CREDENTIAL]")
        return value
    if isinstance(value, dict):
        return {_replace_credentials(key, credentials): _replace_credentials(item, credentials)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_credentials(item, credentials) for item in value]
    return value


def _session_roots(source: Path, destination: Path) -> None:
    """Admit fresh explicit verifier roots before model work, never an auth home."""
    if (source.is_symlink() or destination.is_symlink() or destination.exists()
            or source.resolve().is_relative_to(destination.resolve())
            or destination.resolve().is_relative_to(source.resolve())):
        raise SubscriptionAuthError("classifier session evidence roots are stale or unsafe")
    if source.exists() and (not source.is_dir() or any(source.iterdir())):
        raise SubscriptionAuthError("classifier sessions require a fresh regular directory")


def _retain_classifier_sessions(source: Path, destination: Path, credentials: set[str]) -> None:
    """Publish only fresh Pi JSONL files through shared credential redaction.

    The supplied classifier can read its auth capability. Redact recursively and
    replace known initial/refreshed secrets before hashing retained bytes. Never
    copy the role home, configuration, or other session roots.
    """
    if source.is_symlink() or (source.exists() and not source.is_dir()):
        raise SubscriptionAuthError("classifier session evidence source is unsafe")
    paths = sorted(source.iterdir()) if source.exists() else []
    if any(path.is_symlink() or not path.is_file() or not _SESSION_NAME.fullmatch(path.name) for path in paths):
        raise SubscriptionAuthError("classifier session evidence includes an unexpected file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".judge-evidence-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        sessions = []
        for path in paths:
            content = path.read_bytes()
            text = content.decode("utf-8")
            try:
                entries = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError:
                raise SubscriptionAuthError("classifier session evidence is not valid JSONL") from None
            if (not entries or any(not isinstance(entry, dict) for entry in entries)
                    or entries[0].get("type") != "session"):
                raise SubscriptionAuthError("classifier session evidence is not a Pi session")
            session_credentials = credentials | _embedded_auth_values(entries)
            sanitized = [_replace_credentials(redact_value(entry), session_credentials) for entry in entries]
            if sanitized != entries:
                content = ("\n".join(json.dumps(entry) for entry in sanitized) + "\n").encode("utf-8")
            if b"[REDACTION_FAILED]" in content or any(value.encode() in content for value in session_credentials):
                raise SubscriptionAuthError("classifier session credential redaction failed")
            metadata = {"name": path.name, "sha256": hashlib.sha256(content).hexdigest(),
                        "format": "sanitized-pi-jsonl", "redacted": sanitized != entries}
            atomic_write_bytes(staged / path.name, content, mode=0o600)
            sessions.append(metadata)
        index = {"schema_version": 1, "classifier_sessions": sessions,
                 "classifier_availability": "retained" if sessions else "no_persisted_session",
                 "matcher_sessions": "unavailable_upstream_in_memory", "matcher_usage": "not_retained_by_eval",
                 "monetary_usage_source": "api-equivalent-estimate",
                 "subscription_cash_charges": None}
        atomic_write_bytes(staged / "index.json", json.dumps(index, indent=2).encode("utf-8"), mode=0o600)
        publish_staged_directory(staged, destination, commit_last="index.json")
        destination.chmod(0o700)


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
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    except ProcessLookupError:
                        pass
                # A leader can exit while a tool descendant ignores SIGTERM.
                # Terminate the remaining group before disposing of its state,
                # including descendants left behind after an ordinary exit.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
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
    parser.add_argument("--classifier-sessions", type=Path)
    parser.add_argument("--judge-evidence", type=Path)
    arguments = list(sys.argv[1:] if argv is None else argv)
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    if not command:
        parser.error("judge command is required after --")
    if (args.classifier_sessions is None) != (args.judge_evidence is None):
        parser.error("--classifier-sessions and --judge-evidence must be supplied together")
    try:
        if args.classifier_sessions is not None:
            _session_roots(args.classifier_sessions, args.judge_evidence)
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
            credentials = _credential_values(_read(args.canonical_home / "auth.json"))
            try:
                return _run_judge(command, environment)
            finally:
                if args.classifier_sessions is not None:
                    credentials.update(_credential_values(_read(args.canonical_home / "auth.json")))
                    _retain_classifier_sessions(args.classifier_sessions, args.judge_evidence, credentials)
    except (OSError, UnicodeError, SubscriptionAuthError):
        print("ReviewBench judge authentication, execution, or evidence retention failed; "
              "no API fallback was attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
