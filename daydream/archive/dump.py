"""Publish diagnostic dumps without changing the original archive evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from daydream.archive import scan
from daydream.archive._console import warn
from daydream.archive.sanitize import sanitize_bundle_files


def _refuse_dump(dump_path: Path, session_id: str, reason: str) -> bool:
    # The caller owns this empty private finalization stage, never the public
    # destination. Absence tells artifact routing to preserve its prior baseline;
    # an empty directory would instead publish an empty new dump. rmdir refuses
    # unexpected content rather than deleting evidence we do not own.
    dump_path.rmdir()
    warn(f"Withholding the dump for {session_id}: {reason}; preserving the completed run")
    return False


def publish_dump(assembly_dir: Path, dump_path: Path, session_id: str, *, sanitize: bool = False) -> bool:
    """Copy accepted bytes into an empty private stage, returning whether ready.

    By default, copy the assembled bundle's exact bytes without scanning.
    With explicit sanitization, clean/advisory bundles retain their bytes; blocking bundles get a
    separate sanitized derivative and a second publication scan. Sanitizer or
    scanner failures withhold the dump without invalidating the original run.
    Copy/publication and staging-integrity errors propagate to strict archive
    finalization. The caller handles rollback of any partially copied stage.
    """
    if not sanitize:
        shutil.copytree(assembly_dir, dump_path, dirs_exist_ok=True)
        return True

    try:
        source_scan = scan.scan_run_dir(assembly_dir)
    except Exception as exc:  # noqa: BLE001 - optional diagnostics fail closed
        return _refuse_dump(dump_path, session_id, f"secret scanner failed ({type(exc).__name__})")

    if not source_scan.blocking:
        if source_scan.findings:
            warn(
                f"Publishing the dump for {session_id} with advisory secret-scan findings "
                f"({source_scan.summary()})"
            )
        shutil.copytree(assembly_dir, dump_path, dirs_exist_ok=True)
        return True

    warn(f"Sanitizing the dump for {session_id} ({source_scan.summary()})")
    with TemporaryDirectory(prefix=".dump-", dir=assembly_dir.parent) as temporary:
        derivative = Path(temporary) / "bundle"
        shutil.copytree(assembly_dir, derivative)
        try:
            sanitize_bundle_files(derivative)
            (derivative / "dump-sanitization.json").write_text(
                json.dumps({"schema_version": 1, "session_id": session_id, "sanitized": True}) + "\n",
                encoding="utf-8",
            )
            release_scan = scan.scan_run_dir(derivative)
        except Exception as exc:  # noqa: BLE001 - no exception text may expose credentials
            return _refuse_dump(dump_path, session_id, f"sanitization failed ({type(exc).__name__})")
        if release_scan.blocking:
            return _refuse_dump(
                dump_path, session_id,
                f"sanitized artifact secret scan refused publication ({release_scan.summary()})",
            )
        if release_scan.findings:
            warn(
                f"Sanitized dump for {session_id} carries advisory secret-scan findings "
                f"({release_scan.summary()})"
            )
        shutil.copytree(derivative, dump_path, dirs_exist_ok=True)
    return True
