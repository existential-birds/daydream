"""Archive ownership checks shared by harvest and annotation backfill."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

from daydream.archive.index import label_observation_history
from daydream.training.backfill import run_backfill
from daydream.training.harvest import (
    HarvestConfig,
    HarvestServices,
    make_harvest_services,
    run_harvest,
)
from tests.harness.harvest_services import HarvestTestServices
from tests.test_training_harvest import _seed_archived_deep_run

Entrypoint = Literal["harvest", "backfill"]
_SESSION_ID = "shared-session"


class _ObservedServices(HarvestTestServices):
    """Record the archive query boundary while retaining production adapters."""

    def __init__(
        self,
        delegate: HarvestServices,
        *,
        github: Callable[..., Any],
        events: list[str],
    ) -> None:
        super().__init__(delegate, github=github)
        self._events = events

    def query_rows(self, session_filter: str | None) -> list[Mapping[str, Any]]:
        self._events.append("query")
        return super().query_rows(session_filter)


def _github(events: list[str]) -> Callable[..., Any]:
    def respond(repo: str, endpoint: str, **kwargs: Any) -> Any:
        events.append("github")
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            return []
        return {"merged": True, "merged_at": "2026-02-01T00:00:00Z"}

    return respond


def _seed_archive(path: Path) -> None:
    path.mkdir(exist_ok=True)
    _seed_archived_deep_run(
        path,
        _SESSION_ID,
        merged_at="2026-02-01T00:00:00Z",
    )


async def _invoke(
    entrypoint: Entrypoint,
    requested_archive: Path,
    services: HarvestServices,
    *,
    cache_dir: Path,
    report_path: Path,
) -> None:
    if entrypoint == "harvest":
        await run_harvest(
            HarvestConfig(
                archive_dir=requested_archive,
                cache_dir=cache_dir,
                gh_request_spacing_sec=0,
            ),
            services=services,
        )
        return
    run_backfill(
        requested_archive,
        services=services,
        report_path=report_path,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("entrypoint", ["harvest", "backfill"])
async def test_entrypoint_rejects_services_owned_by_another_archive_before_side_effects(
    tmp_path: Path,
    entrypoint: Entrypoint,
) -> None:
    archive_a = tmp_path / "archive-a"
    archive_b = tmp_path / "archive-b"
    _seed_archive(archive_a)
    _seed_archive(archive_b)
    service_cache = tmp_path / "service-cache"
    requested_cache = tmp_path / "requested-cache"
    report_path = tmp_path / "backfill-report.json"
    events: list[str] = []
    services = _ObservedServices(
        make_harvest_services(
            HarvestConfig(archive_dir=archive_a, cache_dir=service_cache)
        ),
        github=_github(events),
        events=events,
    )
    histories_before = {
        archive: label_observation_history(archive, _SESSION_ID)
        for archive in (archive_a, archive_b)
    }

    with pytest.raises(ValueError, match="archive"):
        await _invoke(
            entrypoint,
            archive_b,
            services,
            cache_dir=requested_cache,
            report_path=report_path,
        )

    assert events == []
    assert {
        archive: label_observation_history(archive, _SESSION_ID)
        for archive in (archive_a, archive_b)
    } == histories_before
    assert not service_cache.exists()
    assert not requested_cache.exists()
    assert not report_path.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("entrypoint", ["harvest", "backfill"])
async def test_entrypoint_accepts_services_owned_by_symlink_alias(
    tmp_path: Path,
    entrypoint: Entrypoint,
) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)
    archive_alias = tmp_path / "archive-alias"
    archive_alias.symlink_to(archive, target_is_directory=True)
    events: list[str] = []
    services = _ObservedServices(
        make_harvest_services(
            HarvestConfig(
                archive_dir=archive,
                cache_dir=tmp_path / "service-cache",
            )
        ),
        github=_github(events),
        events=events,
    )

    await _invoke(
        entrypoint,
        archive_alias,
        services,
        cache_dir=tmp_path / "requested-cache",
        report_path=tmp_path / "backfill-report.json",
    )

    assert events[0] == "query"
    assert "github" in events
    assert len(label_observation_history(archive, _SESSION_ID)) == 1
    assert label_observation_history(archive_alias, _SESSION_ID) == (
        label_observation_history(archive, _SESSION_ID)
    )
