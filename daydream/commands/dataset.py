"""Publish queued JSONL evidence and download exact private Hub snapshots."""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from daydream.commands import common
from daydream.dataset import LocalRecordStore, StoreError
from daydream.ui import create_console, print_error


def _handle_dataset_operation(argv: list[str], *, operation: str) -> int:
    from daydream.dataset_hub import DatasetUploader, download_snapshot

    parser = argparse.ArgumentParser(prog=f"daydream corpus dataset {operation}")
    configured_repo = os.environ.get("DAYDREAM_TRAJECTORY_HUB_REPO") or None
    parser.add_argument("--trajectory-hub-repo", default=configured_repo,
                        required=configured_repo is None, metavar="OWNER/REPO",
                        help="Private Hugging Face dataset; overrides DAYDREAM_TRAJECTORY_HUB_REPO.")
    if operation == "download":
        parser.add_argument("--revision", required=True, help="Exact 40-character Hub commit SHA.")
        parser.add_argument("--output", type=Path, required=True, help="Fresh local record-store directory.")
    else:
        parser.add_argument("--store", type=Path, default=Path.home() / ".daydream" / "dataset",
                            help="Private local JSONL store (default: ~/.daydream/dataset).")
    args = parser.parse_args(argv)
    try:
        if operation == "download":
            if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
                raise StoreError("exact_revision_required")
            store = download_snapshot(args.trajectory_hub_repo, args.revision, args.output)
            records = store.read_records()
            summary = {"revision": args.revision, "runs": len(records["runs"]),
                       "observations": len(records["observations"])}
            failed = False
        else:
            uploader = DatasetUploader(LocalRecordStore(args.store), args.trajectory_hub_repo)
            status = uploader.upload() if operation == "publish" else uploader.status()
            summary = asdict(status)
            failed = bool(status.failed or status.error)
        print(json.dumps(summary, sort_keys=True))
        return 1 if failed else 0
    except Exception as exc:
        # No exception payloads, repository content, credentials, or local paths.
        code = exc.code if isinstance(exc, StoreError) else type(exc).__name__
        print_error(create_console(), "Data Collection", f"Dataset {operation} failed ({code}).")
        return 1


def _handle_publish(argv: list[str]) -> int:
    return _handle_dataset_operation(argv, operation="publish")


def _handle_status(argv: list[str]) -> int:
    return _handle_dataset_operation(argv, operation="status")


def _handle_download(argv: list[str]) -> int:
    return _handle_dataset_operation(argv, operation="download")


def handle_dataset(argv: list[str]) -> int:
    return common._dispatch_namespace(argv, {
        "publish": _handle_publish, "status": _handle_status, "download": _handle_download,
    }, "usage: daydream corpus dataset {publish,status,download} ...\n"
       "\nPublish the local queue, inspect upload status, or download an exact private Hub commit.")
