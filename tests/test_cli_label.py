"""Unknown run labels refuse without appending observations."""
from pathlib import Path

from daydream.dataset import LocalRecordStore
from tests.harness.dataset import run_record
from tests.harness.scripts import cli_main


def test_label_unknown_run_refuses_without_observations(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    assert cli_main(["corpus", "label", "absent", "--store", str(store.root), "--outcome", "accepted"]) == 1
    assert store.read_records()["observations"] == ()
