"""Typed live view over deep-flow ``FlowContext.data`` state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from daydream.backends import ContinuationToken
    from daydream.deep.detection import StackAssignment
    from daydream.deep.fix_steps import FixCycleState, RetainedTreeSnapshot
    from daydream.deep.prompts import DeepDiffBoundInfo
    from daydream.exploration_runner import Tier
    from daydream.phases import PushReceipt


_MISSING = object()


class DeepState:
    """Checked access to the exact extension-visible deep-flow state mapping.

    Construction deliberately does not validate or copy the mapping. A value is
    checked when a built-in stage consumes its property, and setters write to the
    same dictionary so extension and built-in mutations remain mutually visible.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @staticmethod
    def _check(key: str, value: object, expected: type[object], expected_name: str) -> object:
        if not isinstance(value, expected):
            raise TypeError(
                f"deep state key '{key}' expected {expected_name}, got {type(value).__name__}"
            )
        return value

    def _required(self, key: str, expected: type[object], expected_name: str) -> object:
        value: object = self._data[key]
        return self._check(key, value, expected, expected_name)

    def _optional(self, key: str, expected: type[object], expected_name: str) -> object | None:
        value: object | None = self._data.get(key)
        if value is None:
            return None
        return self._check(key, value, expected, expected_name)

    @property
    def mode(self) -> str:
        return str(self._data.get("mode", "loop"))

    @property
    def diff(self) -> str:
        return cast(str, self._required("diff", str, "str"))

    @property
    def diff_or_empty(self) -> str:
        value: object = self._data.get("diff", _MISSING)
        if value is _MISSING or value is None:
            return ""
        return cast(str, self._check("diff", value, str, "str"))

    @property
    def diff_path(self) -> Path:
        return cast(Path, self._required("diff_path", Path, "Path"))

    @property
    def diff_path_or_none(self) -> Path | None:
        return cast(Path | None, self._optional("diff_path", Path, "Path or None"))

    @property
    def diff_truncated(self) -> bool:
        value: object = self._data.get("diff_truncated", _MISSING)
        if value is _MISSING:
            return False
        return cast(bool, self._check("diff_truncated", value, bool, "bool"))

    @property
    def diff_truncation(self) -> DeepDiffBoundInfo | None:
        value: object | None = self._data.get("diff_truncation")
        if value is None:
            return None
        from daydream.deep.prompts import DeepDiffBoundInfo

        return cast(
            DeepDiffBoundInfo,
            self._check("diff_truncation", value, DeepDiffBoundInfo, "DeepDiffBoundInfo or None"),
        )

    @property
    def tier(self) -> Tier:
        value: object = self._data["tier"]
        if value not in ("parallel", "single", "skip"):
            raise TypeError(
                "deep state key 'tier' expected one of: parallel, single, skip, "
                f"got {type(value).__name__}"
            )
        return value

    @property
    def exploration_dir(self) -> Path | None:
        value: object | None = self._data["exploration_dir"]
        if value is None:
            return None
        return cast(Path, self._check("exploration_dir", value, Path, "Path or None"))

    @exploration_dir.setter
    def exploration_dir(self, value: Path | None) -> None:
        self._data["exploration_dir"] = value

    @property
    def exploration_dir_or_none(self) -> Path | None:
        return cast(
            Path | None,
            self._optional("exploration_dir", Path, "Path or None"),
        )

    @property
    def intent_path(self) -> Path:
        return cast(Path, self._required("intent_path", Path, "Path"))

    @intent_path.setter
    def intent_path(self, value: Path) -> None:
        self._data["intent_path"] = value

    @property
    def alts_path(self) -> Path:
        return cast(Path, self._required("alts_path", Path, "Path"))

    @alts_path.setter
    def alts_path(self, value: Path) -> None:
        self._data["alts_path"] = value

    @property
    def items_file(self) -> Path:
        return cast(Path, self._required("items_file", Path, "Path"))

    @items_file.setter
    def items_file(self, value: Path) -> None:
        self._data["items_file"] = value

    @property
    def items(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._required("items", list, "list"))

    @items.setter
    def items(self, value: list[dict[str, Any]]) -> None:
        self._data["items"] = value

    @property
    def items_or_empty(self) -> list[dict[str, Any]]:
        value: object | None = self._data.get("items")
        if value is None:
            return []
        return cast(list[dict[str, Any]], self._check("items", value, list, "list"))

    @property
    def diagrams(self) -> dict[str, Any] | None:
        return cast(
            dict[str, Any] | None,
            self._optional("diagrams", dict, "dict or None"),
        )

    @diagrams.setter
    def diagrams(self, value: dict[str, Any]) -> None:
        self._data["diagrams"] = value

    @property
    def import_graph(self) -> dict[str, set[str]]:
        value: object | None = self._data.get("import_graph")
        if value is None:
            return {}
        return cast(
            dict[str, set[str]],
            self._check("import_graph", value, dict, "dict"),
        )

    @property
    def intent_authoritative(self) -> bool:
        value: object = self._data.get("intent_authoritative", _MISSING)
        if value is _MISSING:
            return False
        return cast(
            bool,
            self._check("intent_authoritative", value, bool, "bool"),
        )

    @intent_authoritative.setter
    def intent_authoritative(self, value: bool) -> None:
        self._data["intent_authoritative"] = value

    @property
    def changed_files(self) -> set[str]:
        return cast(set[str], self._required("changed_files", set, "set"))

    @property
    def changed_files_or_none(self) -> set[str] | None:
        return cast(
            set[str] | None,
            self._optional("changed_files", set, "set or None"),
        )

    @property
    def dd(self) -> Path:
        return cast(Path, self._required("dd", Path, "Path"))

    @property
    def stacks(self) -> list[StackAssignment]:
        return cast("list[StackAssignment]", self._required("stacks", list, "list"))

    @property
    def single_stack_mode(self) -> bool:
        return cast(bool, self._required("single_stack_mode", bool, "bool"))

    @property
    def log(self) -> str:
        return cast(str, self._required("log", str, "str"))

    @property
    def branch(self) -> str:
        return cast(str, self._required("branch", str, "str"))

    @property
    def failed_stacks(self) -> dict[str, str]:
        return cast(
            dict[str, str],
            self._required("failed_stacks", dict, "dict"),
        )

    @failed_stacks.setter
    def failed_stacks(self, value: dict[str, str]) -> None:
        self._data["failed_stacks"] = value

    @property
    def failed_stacks_or_none(self) -> dict[str, str] | None:
        value: object | None = self._data.get("failed_stacks")
        if value is None:
            return None
        checked = cast(
            dict[str, str],
            self._check("failed_stacks", value, dict, "dict or None"),
        )
        return checked or None

    @property
    def intent_summary(self) -> str:
        return cast(str, self._required("intent_summary", str, "str"))

    @intent_summary.setter
    def intent_summary(self, value: str) -> None:
        self._data["intent_summary"] = value

    @property
    def per_stack_outputs(self) -> dict[str, Path]:
        return cast(
            dict[str, Path],
            self._required("per_stack_outputs", dict, "dict"),
        )

    @per_stack_outputs.setter
    def per_stack_outputs(self, value: dict[str, Path]) -> None:
        self._data["per_stack_outputs"] = value

    @property
    def records_paths(self) -> list[Path]:
        return cast(list[Path], self._required("records_paths", list, "list"))

    @records_paths.setter
    def records_paths(self, value: list[Path]) -> None:
        self._data["records_paths"] = value

    @property
    def records(self) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._required("records", list, "list"),
        )

    @records.setter
    def records(self, value: list[dict[str, Any]]) -> None:
        self._data["records"] = value

    @property
    def record_sources(self) -> list[str]:
        return cast(list[str], self._required("record_sources", list, "list"))

    @record_sources.setter
    def record_sources(self, value: list[str]) -> None:
        self._data["record_sources"] = value

    @property
    def structural_records_path(self) -> Path | None:
        value: object | None = self._data["structural_records_path"]
        if value is None:
            return None
        return cast(
            Path,
            self._check("structural_records_path", value, Path, "Path or None"),
        )

    @structural_records_path.setter
    def structural_records_path(self, value: Path | None) -> None:
        self._data["structural_records_path"] = value

    @property
    def structural_records_path_or_none(self) -> Path | None:
        return cast(
            Path | None,
            self._optional("structural_records_path", Path, "Path or None"),
        )

    @property
    def structural_records(self) -> list[dict[str, Any]]:
        value: object | None = self._data.get("structural_records")
        if value is None:
            return []
        return cast(
            list[dict[str, Any]],
            self._check("structural_records", value, list, "list"),
        )

    @structural_records.setter
    def structural_records(self, value: list[dict[str, Any]]) -> None:
        self._data["structural_records"] = value

    @property
    def structural_record_sources(self) -> list[str]:
        value: object | None = self._data.get("structural_record_sources")
        if value is None:
            return []
        return cast(
            list[str],
            self._check("structural_record_sources", value, list, "list"),
        )

    @structural_record_sources.setter
    def structural_record_sources(self, value: list[str]) -> None:
        self._data["structural_record_sources"] = value

    @property
    def arbiter_continuation(self) -> ContinuationToken | None:
        value: object | None = self._data.get("arbiter_continuation")
        if value is None:
            return None
        from daydream.backends import ContinuationToken

        return cast(
            ContinuationToken,
            self._check(
                "arbiter_continuation",
                value,
                ContinuationToken,
                "ContinuationToken or None",
            ),
        )

    @arbiter_continuation.setter
    def arbiter_continuation(self, value: ContinuationToken) -> None:
        self._data["arbiter_continuation"] = value

    @property
    def merged_report(self) -> Path:
        return cast(Path, self._required("merged_report", Path, "Path"))

    @merged_report.setter
    def merged_report(self, value: Path) -> None:
        self._data["merged_report"] = value

    @property
    def merged_report_or_none(self) -> Path | None:
        return cast(
            Path | None,
            self._optional("merged_report", Path, "Path or None"),
        )

    @property
    def fix_cycle_state(self) -> FixCycleState:
        value: object | None = self._data.get("fix_cycle_state")
        from daydream.deep.fix_steps import FixCycleState

        if not isinstance(value, FixCycleState):
            raise RuntimeError("fix cycle was not initialized at the accepted gate")
        return value

    @fix_cycle_state.setter
    def fix_cycle_state(self, value: FixCycleState) -> None:
        self._data["fix_cycle_state"] = value

    @property
    def iteration(self) -> int | None:
        value: object | None = self._data.get("iteration")
        if value is None:
            return None
        if type(value) is not int:
            raise TypeError(
                f"deep state key 'iteration' expected int or None, got {type(value).__name__}"
            )
        return value

    @property
    def fix_outcomes(self) -> dict[str, dict[str, Any]]:
        value: object | None = self._data.get("fix_outcomes")
        if value is None:
            return {}
        return cast(
            dict[str, dict[str, Any]],
            self._check("fix_outcomes", value, dict, "dict"),
        )

    @fix_outcomes.setter
    def fix_outcomes(self, value: dict[str, dict[str, Any]]) -> None:
        self._data["fix_outcomes"] = value

    @property
    def fix_round_items(self) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._required("fix_round_items", list, "list"),
        )

    @fix_round_items.setter
    def fix_round_items(self, value: list[dict[str, Any]]) -> None:
        self._data["fix_round_items"] = value

    @property
    def fix_round_snapshot(self) -> RetainedTreeSnapshot | None:
        value: object | None = self._data.get("fix_round_snapshot")
        from daydream.deep.fix_steps import RetainedTreeSnapshot

        if not isinstance(value, RetainedTreeSnapshot):
            return None
        return value

    @fix_round_snapshot.setter
    def fix_round_snapshot(self, value: RetainedTreeSnapshot) -> None:
        self._data["fix_round_snapshot"] = value

    @property
    def push_receipt(self) -> PushReceipt | None:
        value: object | None = self._data.get("push_receipt")
        if value is None:
            return None
        from daydream.phases import PushReceipt

        if not isinstance(value, PushReceipt):
            return None
        return value

    @push_receipt.setter
    def push_receipt(self, value: PushReceipt) -> None:
        self._data["push_receipt"] = value


__all__ = ["DeepState"]
