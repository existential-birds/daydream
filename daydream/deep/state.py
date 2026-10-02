"""Typed live view over deep-flow ``FlowContext.data`` state."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

from daydream.deep.records import RecordPool
from daydream.review_result import ReviewCoverage

if TYPE_CHECKING:
    from daydream.backends import ContinuationToken
    from daydream.deep.detection import StackAssignment
    from daydream.deep.diff import DeepDiffBoundInfo
    from daydream.deep.fix_state import FixCycleState, RetainedTreeSnapshot
    from daydream.deep.latency import ArbiterPlan, LatencyRoute, RiskSummary
    from daydream.deep.reuse_store import ReuseCache
    from daydream.exploration_runner import Tier
    from daydream.phases import PushReceipt
    from daydream.test_execution import TestRecipe


_T = TypeVar("_T")


class _StateField(Generic[_T]):
    """A checked mapping field with an optional fresh default and opt-in writes."""

    def __init__(
        self,
        expected: type[object],
        *,
        key: str | None = None,
        label: str | None = None,
        default: Callable[[], _T] | None = None,
        writable: bool = False,
    ) -> None:
        self.expected = expected
        self.key = key
        self.label = label or expected.__name__
        self.default = default
        self.writable = writable

    def __set_name__(self, owner: type[DeepState], name: str) -> None:
        self.name = name
        self.key = self.key or name

    @overload
    def __get__(self, instance: None, owner: type[DeepState]) -> _StateField[_T]: ...

    @overload
    def __get__(self, instance: DeepState, owner: type[DeepState] | None = None) -> _T: ...

    def __get__(
        self, instance: DeepState | None, owner: type[DeepState] | None = None
    ) -> _T | _StateField[_T]:
        if instance is None:
            return self
        assert self.key is not None
        value = instance._data[self.key] if self.default is None else instance._data.get(self.key)
        if value is None and self.default is not None:
            return self.default()
        return cast(_T, instance._check(self.key, value, self.expected, self.label))

    def __set__(self, instance: DeepState, value: _T) -> None:
        if not self.writable:
            raise AttributeError(f"property '{self.name}' of 'DeepState' object has no setter")
        assert self.key is not None
        instance._data[self.key] = value


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

    def _optional(
        self, key: str, expected: type[object], expected_name: str, default: object | None = None
    ) -> object | None:
        value: object | None = self._data.get(key)
        if value is None:
            return default
        return self._check(key, value, expected, expected_name)

    review_coverage = _StateField[ReviewCoverage](ReviewCoverage)

    @property
    def mode(self) -> str:
        return str(self._data.get("mode", "loop"))

    diff = _StateField[str](str)

    @property
    def diff_or_empty(self) -> str:
        return cast(str, self._optional("diff", str, "str")) or ""

    diff_path = _StateField[Path](Path)

    diff_path_or_none = _StateField[Path | None](Path, key='diff_path', label='Path or None', default=lambda: None)

    diff_truncated = _StateField[bool](bool, default=bool)

    @property
    def diff_truncation(self) -> DeepDiffBoundInfo | None:
        from daydream.deep.diff import DeepDiffBoundInfo

        return cast(
            DeepDiffBoundInfo | None,
            self._optional("diff_truncation", DeepDiffBoundInfo, "DeepDiffBoundInfo or None"),
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

    exploration_dir_or_none = _StateField[Path | None](
        Path, key='exploration_dir', label='Path or None', default=lambda: None
    )

    intent_path = _StateField[Path](Path, writable=True)

    alts_path = _StateField[Path](Path, writable=True)

    items_file = _StateField[Path](Path, writable=True)

    items = _StateField[list[dict[str, Any]]](list, writable=True)

    items_or_empty = _StateField[list[dict[str, Any]]](list, key='items', default=list)

    diagrams = _StateField[dict[str, Any] | None](dict, label='dict or None', default=lambda: None, writable=True)

    import_graph = _StateField[dict[str, set[str]]](dict, default=dict)

    intent_authoritative = _StateField[bool](bool, default=bool, writable=True)

    changed_files = _StateField[set[str]](set)

    changed_files_or_none = _StateField[set[str] | None](
        set, key='changed_files', label='set or None', default=lambda: None
    )

    dd = _StateField[Path](Path)

    stacks: _StateField[list[StackAssignment]] = _StateField(list)

    single_stack_mode = _StateField[bool](bool)

    @property
    def latency_route(self) -> LatencyRoute | None:
        from daydream.deep.latency import LatencyRoute

        return cast(
            LatencyRoute | None,
            self._optional("latency_route", LatencyRoute, "LatencyRoute or None"),
        )

    @property
    def reuse_cache(self) -> ReuseCache | None:
        from daydream.deep.reuse_store import ReuseCache

        return cast(
            ReuseCache | None,
            self._optional("reuse_cache", ReuseCache, "ReuseCache or None"),
        )

    @property
    def risk_summary(self) -> RiskSummary | None:
        from daydream.deep.latency import RiskSummary

        return cast(
            RiskSummary | None,
            self._optional("risk_summary", RiskSummary, "RiskSummary or None"),
        )

    @property
    def arbiter_plan(self) -> ArbiterPlan | None:
        from daydream.deep.latency import ArbiterPlan

        return cast(
            ArbiterPlan | None,
            self._optional("arbiter_plan", ArbiterPlan, "ArbiterPlan or None"),
        )

    @arbiter_plan.setter
    def arbiter_plan(self, value: ArbiterPlan) -> None:
        self._data["arbiter_plan"] = value

    log = _StateField[str](str)

    branch = _StateField[str](str)

    @property
    def unfinished_scopes(self) -> dict[str, str]:
        return self.review_coverage.unfinished_scopes

    intent_summary = _StateField[str](str, writable=True)

    intent_summary_or_none = _StateField[str | None](
        str, key='intent_summary', label='str or None', default=lambda: None
    )

    record_pool = _StateField[RecordPool](RecordPool, writable=True)

    @property
    def arbiter_continuation(self) -> ContinuationToken | None:
        from daydream.backends import ContinuationToken

        return cast(
            ContinuationToken | None,
            self._optional(
                "arbiter_continuation", ContinuationToken, "ContinuationToken or None"
            ),
        )

    @arbiter_continuation.setter
    def arbiter_continuation(self, value: ContinuationToken) -> None:
        self._data["arbiter_continuation"] = value

    merged_report = _StateField[Path](Path, writable=True)

    merged_report_or_none = _StateField[Path | None](
        Path, key='merged_report', label='Path or None', default=lambda: None
    )

    @property
    def fix_cycle_state(self) -> FixCycleState:
        value: object | None = self._data.get("fix_cycle_state")
        from daydream.deep.fix_state import FixCycleState

        if not isinstance(value, FixCycleState):
            raise RuntimeError("fix cycle was not initialized at the accepted gate")
        return value

    @fix_cycle_state.setter
    def fix_cycle_state(self, value: FixCycleState) -> None:
        self._data["fix_cycle_state"] = value

    @property
    def test_recipe(self) -> TestRecipe | None:
        from daydream.test_execution import TestRecipe

        return cast(
            TestRecipe | None,
            self._optional("test_recipe", TestRecipe, "TestRecipe or None"),
        )

    @test_recipe.setter
    def test_recipe(self, value: TestRecipe | None) -> None:
        self._data["test_recipe"] = value

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

    fix_outcomes = _StateField[dict[str, dict[str, Any]]](dict, default=dict, writable=True)

    @property
    def fix_round_snapshot(self) -> RetainedTreeSnapshot | None:
        value: object | None = self._data.get("fix_round_snapshot")
        from daydream.deep.fix_state import RetainedTreeSnapshot

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
