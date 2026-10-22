"""Role-private ChatGPT subscription credentials for the public Harbor profile.

This file is also copied verbatim into the verifier image and runs on Python 3.11+.
One persistent canonical Codex auth directory is shared by both environments.
An exclusive lease lasts for the *whole invocation*: native Codex and Pi each own
refresh, so locking just the initial copy would race their rotating credentials.
Harbor trial concurrency can queue, but this account's model invocations serialize.
The dedicated canonical login must not be used by unrelated Codex processes.
Credential contents are never configuration, fingerprints, or retained evidence.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from daydream.redaction import redact_value

try:
    from daydream.json_utils import atomic_write_bytes, publish_staged_directory
except ModuleNotFoundError as missing:
    if missing.name not in {"daydream", "daydream.json_utils"}:
        raise
    # The verifier ships this exact shared-owner module beside its bootstrap.
    shared_writer = importlib.import_module("daydream_json_utils")
    atomic_write_bytes = shared_writer.atomic_write_bytes
    publish_staged_directory = shared_writer.publish_staged_directory


class SubscriptionAuthError(ValueError):
    """An intentionally credential-free subscription capability failure."""


def _read(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise SubscriptionAuthError("subscription auth must be a regular private file")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise SubscriptionAuthError("invalid subscription auth object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SubscriptionAuthError("subscription auth is unavailable or malformed") from None


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
    # client performs refresh under the invocation lease, including failure cases.
    return value


def validate_subscription_home(home: Path) -> None:
    """Validate OAuth capability without returning or recording credential bytes."""
    _codex_auth(_read(home / "auth.json"))


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise SubscriptionAuthError("refusing a symlink subscription auth destination")
    owner = (path if path.exists() else path.parent).stat()
    atomic_write_bytes(path, json.dumps(value).encode("utf-8"), fsync=True, mode=0o600,
                       owner=(owner.st_uid, owner.st_gid))


def _judge_auth(codex: dict[str, Any]) -> dict[str, Any]:
    tokens = codex["tokens"]
    return {"openai-codex": {
        "type": "oauth", "access": tokens["access_token"], "refresh": tokens["refresh_token"],
        "expires": _claims(tokens["access_token"])["exp"] * 1000,
        "accountId": tokens["account_id"],
    }}


def _judge_update(codex: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    credential = judge.get("openai-codex")
    if not isinstance(credential, dict) or credential.get("type") != "oauth":
        raise SubscriptionAuthError("judge lost its subscription OAuth capability")
    access, refresh, account = (_text(credential.get(key)) for key in ("access", "refresh", "accountId"))
    if account != codex["tokens"]["account_id"]:
        raise SubscriptionAuthError("judge subscription account changed during invocation")
    if (access, refresh) != (codex["tokens"]["access_token"], codex["tokens"]["refresh_token"]):
        codex["tokens"].update(access_token=access, refresh_token=refresh)
        codex["last_refresh"] = datetime.now(timezone.utc).isoformat()
    return _codex_auth(codex)


@contextmanager
def subscription_lease(
    canonical_home: Path, role_home: Path, *, role: Literal["reviewer", "judge"] = "reviewer",
) -> Iterator[Path]:
    """Borrow exact OAuth capability and reconcile refresh even after model failure.

    The canonical directory is an explicit runtime mount, never the personal home.
    Role directories must be private and contain no earlier auth. Their auth file
    links to a private durable credential file on the canonical mount: the pinned
    Codex 0.163.0 file store and Pi 0.75.5 file store update auth in place. Their
    conversations, configuration and logs remain in ephemeral role directories.
    A failed writeback or process death retains the durable role-format auth and
    marker for recovery even after the role directory/container is removed.
    """
    if role not in {"reviewer", "judge"}:
        raise SubscriptionAuthError("unsupported subscription role")
    if (canonical_home.resolve() == role_home.resolve() or canonical_home.is_symlink()
            or role_home.is_symlink()):
        raise SubscriptionAuthError("canonical and private role homes must be separate")
    if not canonical_home.is_dir():
        raise SubscriptionAuthError("canonical subscription home is unavailable")
    role_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    role_home.chmod(0o700)
    role_path = role_home / "auth.json"
    if role_path.exists() or role_path.is_symlink():
        raise SubscriptionAuthError("refusing stale role subscription credentials")
    lock_path = canonical_home / ".lease.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        owner, current = canonical_home.stat(), os.fstat(lock.fileno())
        if (current.st_uid, current.st_gid) != (owner.st_uid, owner.st_gid):
            os.fchown(lock.fileno(), owner.st_uid, owner.st_gid)
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = canonical_home / ".lease-state.json"
        durable = canonical_home / ".lease-auth.json"
        if state.exists() or durable.exists():
            raise SubscriptionAuthError("interrupted subscription lease requires credential recovery")
        original = _codex_auth(_read(canonical_home / "auth.json"))
        _write(state, {"role": role, "credential_file": durable.name})
        _write(durable, original if role == "reviewer" else _judge_auth(original))
        # Pinned Codex file-store save uses OpenOptions::truncate/write (not
        # rename), and pinned Pi FileAuthStorageBackend uses writeFileSync.
        # Only credential bytes follow this link; sessions stay in role_home.
        # github.com/openai/codex/blob/rust-v0.163.0/codex-rs/login/src/auth/storage.rs
        role_path.symlink_to(durable.resolve())
        try:
            yield role_home
        finally:
            if _read(canonical_home / "auth.json") != original:
                raise SubscriptionAuthError("canonical auth changed outside the reserved subscription lease")
            updated = _read(durable)
            updated = _codex_auth(updated) if role == "reviewer" else _judge_update(original, updated)
            if updated["tokens"]["account_id"] != original["tokens"]["account_id"]:
                raise SubscriptionAuthError("reviewer subscription account changed during invocation")
            _write(canonical_home / "auth.json", updated)
            role_path.unlink()
            durable.unlink()
            state.unlink()



@contextmanager
def subscription_session(source_home: Path, canonical_home: Path) -> Iterator[Path]:
    """Import only auth for one host job, then return native refresh updates.

    The source lock coordinates other public ReviewBench jobs, but cannot control
    unrelated Codex clients. The operator must reserve this login for the session.
    Source settings and all unrelated personal files stay outside the containers.
    On reconciliation failure canonical credentials remain available for recovery.
    """
    if source_home.resolve() == canonical_home.resolve() or canonical_home.is_symlink():
        raise SubscriptionAuthError("source and canonical auth homes must be separate")
    validate_subscription_home(source_home)
    canonical_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(canonical_home.iterdir()):
        raise SubscriptionAuthError("canonical subscription session requires a fresh directory")
    canonical_home.chmod(0o700)
    descriptor = os.open(source_home / ".reviewbench-subscription.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = _codex_auth(_read(source_home / "auth.json"))
        _write(canonical_home / "auth.json", original)
        try:
            yield canonical_home
        finally:
            if (canonical_home / ".lease-state.json").exists():
                raise SubscriptionAuthError("interrupted subscription lease prevents source credential reconciliation")
            current = _codex_auth(_read(source_home / "auth.json"))
            # Never clobber credentials updated by an unrelated, noncooperating
            # client. Preserve both stores instead and require reconciliation.
            if current != original:
                raise SubscriptionAuthError("source auth changed outside the reserved benchmark session")
            updated = _codex_auth(_read(canonical_home / "auth.json"))
            if updated["tokens"]["account_id"] != original["tokens"]["account_id"]:
                raise SubscriptionAuthError("canonical subscription account changed during session")
            if updated != original:
                _write(source_home / "auth.json", updated)
            (canonical_home / "auth.json").unlink()


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unchanged upstream judge under the same refresh lease as Codex."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("judge",))
    parser.add_argument("--canonical-home", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
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
            with subscription_lease(args.canonical_home, home, role="judge"):
                environment = subscription_environment(os.environ)
                environment["PI_CODING_AGENT_DIR"] = str(home)
                credentials: set[str] = set()
                if args.classifier_sessions is not None:
                    # The role path is an intentional symlink; read only its
                    # admitted regular durable target while this lease is held.
                    credentials.update(_credential_values(_read(args.canonical_home / ".lease-auth.json")))
                    credentials.update(_credential_values(_read(args.canonical_home / "auth.json")))
                try:
                    return subprocess.call(command, env=environment)
                finally:
                    if args.classifier_sessions is not None:
                        credentials.update(_credential_values(_read(args.canonical_home / ".lease-auth.json")))
                        _retain_classifier_sessions(args.classifier_sessions, args.judge_evidence, credentials)
    except (OSError, UnicodeError, SubscriptionAuthError):
        print("ReviewBench judge authentication, execution, or evidence retention failed; "
              "no API fallback was attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
