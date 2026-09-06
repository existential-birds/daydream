"""Normalized authorization policy and audit events for the deep fix loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from daydream.repository_paths import (
    InvalidRepositoryFilePath,
    canonicalize_repository_file_path,
    git_observed_path_is_confined,
)

FixFootprintAction = Literal[
    "authorize",
    "approve_generated",
    "rejected_retarget",
    "restore",
    "remove",
    "stage",
    "reject",
]
FixFootprintPathKind = Literal["model", "git"]
FixFootprintOrigin = Literal[
    "reviewed",
    "primary",
    "related",
    "generated",
    "guard",
    "retarget",
    "staging",
]

_ACTIONS = frozenset(
    {"authorize", "approve_generated", "rejected_retarget", "restore", "remove", "stage", "reject"}
)
_ORIGINS = frozenset({"reviewed", "primary", "related", "generated", "guard", "retarget", "staging"})


@dataclass(frozen=True)
class FixFootprintEvent:
    """One monotonically ordered authorization or enforcement decision."""

    sequence: int
    action: FixFootprintAction
    path: str
    path_kind: FixFootprintPathKind
    origin: FixFootprintOrigin
    item_uid: str | None
    phase: str
    round_number: int | None
    reason: str


@dataclass
class AuthorizedFixFootprint:
    """Run-wide and per-item normalized edit authorization.

    The run union is used only after parallel work joins. Mutating fixer groups
    receive the exact union of their member items through :meth:`group_paths`.
    """

    run_allowed_paths: frozenset[str]
    policy_revision: int
    _item_paths: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _item_primary: dict[str, str] = field(default_factory=dict, repr=False)
    _events: list[FixFootprintEvent] = field(default_factory=list, repr=False)

    @classmethod
    def build(
        cls,
        repo: Path,
        reviewed_paths: set[str] | None,
        items: list[dict[str, Any]],
    ) -> AuthorizedFixFootprint:
        """Build a fail-closed footprint from reviewed paths and canonical items."""
        # Reviewed paths come from the host's Git diff, not model output.
        # Preserve their exact native spelling while retaining confinement;
        # item targets and related/retarget paths still use the model grammar.
        reviewed_set: set[str] = set()
        for path in reviewed_paths or set():
            if not isinstance(path, str):
                raise InvalidRepositoryFilePath("invalid reviewed repository path")
            normalized = path[2:] if path.startswith("./") else path
            if not git_observed_path_is_confined(repo, normalized):
                raise InvalidRepositoryFilePath("invalid reviewed repository path")
            reviewed_set.add(normalized)
        reviewed = sorted(reviewed_set, key=lambda path: path.encode("utf-8", errors="surrogateescape"))

        normalized_items: list[tuple[str, str, tuple[str, ...]]] = []
        seen_uids: set[str] = set()
        for item in items:
            uid = item.get("item_uid")
            if not isinstance(uid, str) or not uid or uid in seen_uids:
                raise ValueError("fix footprint requires unique non-empty item_uid values")
            seen_uids.add(uid)
            primary = canonicalize_repository_file_path(repo, item.get("file"))
            raw_related = item.get("related_files")
            if raw_related is None:
                related_values: list[object] = []
            elif isinstance(raw_related, list):
                related_values = raw_related
            else:
                raise InvalidRepositoryFilePath("invalid repository file path")
            related = tuple(canonicalize_repository_file_path(repo, value) for value in related_values)
            normalized_items.append((uid, primary, related))

        footprint = cls(run_allowed_paths=frozenset(), policy_revision=1)
        allowed: set[str] = set()
        for path in reviewed:
            allowed.add(path)
            footprint._append_event(
                action="authorize",
                path=path,
                path_kind="git",
                origin="reviewed",
                item_uid=None,
                phase="initialization",
                round_number=None,
                reason="path is part of the reviewed change",
            )
        for uid, primary, related in normalized_items:
            item_set = {primary, *related}
            footprint._item_paths[uid] = frozenset(item_set)
            footprint._item_primary[uid] = primary
            allowed.update(item_set)
            footprint._append_event(
                action="authorize",
                path=primary,
                path_kind="model",
                origin="primary",
                item_uid=uid,
                phase="initialization",
                round_number=None,
                reason="finding primary path",
            )
            for path in related:
                footprint._append_event(
                    action="authorize",
                    path=path,
                    path_kind="model",
                    origin="related",
                    item_uid=uid,
                    phase="initialization",
                    round_number=None,
                    reason="finding related path",
                )
        footprint.run_allowed_paths = frozenset(allowed)
        return footprint

    @property
    def events(self) -> tuple[FixFootprintEvent, ...]:
        """Return the immutable ordered audit-event view."""
        return tuple(self._events)

    def item_paths(self, item_uid: str) -> frozenset[str]:
        """Return the exact edit paths authorized to one durable item identity."""
        try:
            return self._item_paths[item_uid]
        except KeyError as exc:
            raise ValueError("unknown fix-footprint item_uid") from exc

    def group_paths(self, items: list[dict[str, Any]]) -> frozenset[str]:
        """Return the exact transitive union for a dispatched item group."""
        paths: set[str] = set()
        for item in items:
            uid = item.get("item_uid")
            if not isinstance(uid, str):
                raise ValueError("fix footprint requires item_uid")
            paths.update(self.item_paths(uid))
        return frozenset(paths)

    def accept_retarget(
        self,
        repo: Path,
        item_uid: str,
        candidate: object,
        *,
        phase: str,
        round_number: int,
    ) -> str | None:
        """Accept only a normalized retarget already authorized to this item."""
        own_paths = self.item_paths(item_uid)
        try:
            normalized = canonicalize_repository_file_path(repo, candidate)
        except InvalidRepositoryFilePath:
            self._append_event(
                action="rejected_retarget",
                path=self._item_primary[item_uid],
                path_kind="model",
                origin="retarget",
                item_uid=item_uid,
                phase=phase,
                round_number=round_number,
                reason="retarget path is invalid; original target retained",
            )
            return None
        if normalized in own_paths:
            self._append_event(
                action="authorize",
                path=normalized,
                path_kind="model",
                origin="retarget",
                item_uid=item_uid,
                phase=phase,
                round_number=round_number,
                reason="retarget selected an existing item-authorized path; policy unchanged",
            )
            return normalized
        self._append_event(
            action="rejected_retarget",
            path=normalized,
            path_kind="model",
            origin="retarget",
            item_uid=item_uid,
            phase=phase,
            round_number=round_number,
            reason="retarget is outside the item's authorized paths; original target retained",
        )
        return None

    def authorize_new_generated(
        self,
        repo: Path,
        path: str,
        *,
        phase: str,
        round_number: int | None,
        reason: str,
    ) -> None:
        """Widen the run policy once for a policy-approved generated path."""
        normalized = canonicalize_repository_file_path(repo, path)
        if normalized in self.run_allowed_paths:
            return
        self.run_allowed_paths = frozenset((*self.run_allowed_paths, normalized))
        self.policy_revision += 1
        self._append_event(
            action="approve_generated",
            path=normalized,
            path_kind="model",
            origin="generated",
            item_uid=None,
            phase=phase,
            round_number=round_number,
            reason=reason,
        )

    def record_git_event(
        self,
        *,
        action: str,
        path: str,
        origin: str,
        phase: str,
        round_number: int | None,
        reason: str,
    ) -> None:
        """Record a decision involving an exact Git-observed path."""
        if action not in _ACTIONS or origin not in _ORIGINS:
            raise ValueError("invalid fix-footprint audit event")
        self._append_event(
            action=cast(FixFootprintAction, action),
            path=path,
            path_kind="git",
            origin=cast(FixFootprintOrigin, origin),
            item_uid=None,
            phase=phase,
            round_number=round_number,
            reason=reason,
        )

    def audit_payload(self, session_id: str, **evidence: Any) -> dict[str, Any]:
        """Return the complete session-bound JSON-serializable audit payload."""
        return {
            "session_id": session_id,
            "policy_revision": self.policy_revision,
            "run_allowed_paths": sorted(
                self.run_allowed_paths,
                key=lambda path: path.encode("utf-8", errors="surrogateescape"),
            ),
            "item_paths": {
                uid: sorted(paths, key=lambda path: path.encode("utf-8", errors="surrogateescape"))
                for uid, paths in sorted(self._item_paths.items())
            },
            "events": [asdict(event) for event in self._events],
            **evidence,
        }

    def _append_event(
        self,
        *,
        action: FixFootprintAction,
        path: str,
        path_kind: FixFootprintPathKind,
        origin: FixFootprintOrigin,
        item_uid: str | None,
        phase: str,
        round_number: int | None,
        reason: str,
    ) -> None:
        self._events.append(
            FixFootprintEvent(
                sequence=len(self._events) + 1,
                action=action,
                path=path,
                path_kind=path_kind,
                origin=origin,
                item_uid=item_uid,
                phase=phase,
                round_number=round_number,
                reason=reason,
            )
        )


__all__ = ["AuthorizedFixFootprint", "FixFootprintEvent"]
