"""Taskset scaffold — implemented in Phase 1."""

from __future__ import annotations

import verifiers.v1 as vf


class DaydreamReviewData(vf.TaskData):
    """Placeholder; Phase 1 adds the harvested-corpus fields."""


class DaydreamReviewTaskConfig(vf.TaskConfig):
    """Placeholder; Phase 1 adds the reward weights."""


class DaydreamReviewTask(vf.Task[DaydreamReviewData, vf.State, DaydreamReviewTaskConfig]):
    """Placeholder; Phase 3 adds the rewards and metrics."""


class DaydreamReviewConfig(vf.TasksetConfig):
    """Placeholder; Phase 1 adds corpus_dir / manifest_path."""

    task: DaydreamReviewTaskConfig = DaydreamReviewTaskConfig()


class DaydreamReviewTaskset(vf.Taskset[DaydreamReviewTask, DaydreamReviewConfig]):
    def load(self) -> list[DaydreamReviewTask]:
        raise NotImplementedError("Phase 1")
