"""Offline snapshot preview preserves complete captured host finding populations."""

import json
from pathlib import Path

from daydream.dataset import LocalRecordStore
from tests.harness.adjudication import snapshot_id
from tests.harness.record_projection import projection_run
from tests.harness.scripts import cli_main


def test_preview_keeps_distinct_host_findings_with_same_fingerprint(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    run = projection_run("same-fingerprint", dispositions=("unanswered", "missing"))
    run["findings"]["value"]["items"][1]["fingerprint"] = run["findings"]["value"]["items"][0]["fingerprint"]
    store.commit_run(run)
    state = tmp_path / "state"
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "build",
                "--store",
                str(store.root),
                "--snapshot-id",
                snapshot_id(store),
                "--state-dir",
                str(state),
            ]
        )
        == 0
    )
    items = json.loads((state / "queue.json").read_text())
    assert len(items) == 2 and len({i["record_id"] for i in items}) == 2
    assert len({i["fingerprint"] for i in items}) == 1 and {i["item_uid"] for i in items} == {"item:0", "item:1"}
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "label",
                "--state-dir",
                str(state),
                "--record-id",
                items[0]["record_id"],
                "--labeler",
                "alice",
                "--disposition",
                "accepted",
                "--rationale",
                "one host finding",
            ]
        )
        == 0
    )
    judgments = [o for o in store.read_records()["observations"] if o["payload"]["type"] == "finding-judgment"]
    assert len(judgments) == 1 and judgments[0]["item_uid"] == items[0]["item_uid"]
