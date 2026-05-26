"""Harvest pass — assemble immutable bronze signals into reward inputs.

The harvest pass is the single deferred *annotate* step of the corpus
pipeline: it reads an archived run's immutable bronze artifacts, reduces
them to a :class:`~daydream.training.reward.ScoringInputs`, scores an
intrinsic :class:`~daydream.training.reward.RewardBreakdown`, derives the
outcome label, and appends one bitemporal annotation. There is no separate
"labeling" step: a single annotate pass writes label + reward together.

This module carries the bronze-signal assembly step, the per-run annotation
builder, and the :func:`run_harvest` orchestrator that walks the archive index
and appends one fresh annotation generation per run.

Signal sources (all under the archived run directory):

* ``deep/recommendation-verdicts.json`` — the ``verdicts`` list produced by
  the recommendation-verification stage (verdict shape mirrors
  :data:`daydream.phases.RECOMMENDATION_VERDICTS_SCHEMA`).
* ``deep/stack-*-records.json`` — per-stack finding records (shape mirrors
  the reader in :mod:`daydream.eval.analyzer`).
* ``review-output.md`` (root) falling back to ``deep/review-output.md`` —
  the char-count length proxy (matching the back-compat fallback in the
  former exporter).

Failure-propagation rules:

* Absent structured artifacts ⇒ ``verifier_verdicts=None`` (the shallow-run
  path); ``format_valid`` stays ``True`` because nothing failed to parse.
* A *present* verdicts/records file that is malformed JSON ⇒ caught as
  :class:`json.JSONDecodeError` and surfaced as ``format_valid=False``;
  assembly never crashes on bad data.
* ``grounding_rate`` is read from the indexed manifest row
  (``row["grounding_rate"]``), never re-derived here.
* ``length`` is the documented review-output char-count proxy, ``None`` when
  no review output exists.

Annotation builder:

* :func:`build_annotation` composes the posterior rubric (PR vs local-branch,
  mirroring the former labeler's assembly), derives the outcome label, scores
  the intrinsic reward, and returns a frozen :class:`AnnotationPayload`. It is
  *pure of DB writes* — the orchestrator persists the payload. Posterior-fetch
  and git errors propagate to the caller, which isolates per-row.
* ``valid_at`` is the PR merge timestamp for PR rows and ``None`` for
  non-PR/local rows (the write layer collapses ``None`` → ``observed_at``).

Orchestrator (:func:`run_harvest`):

* **Append-only and re-runnable:** every indexed run is considered on every
  pass; rows that already carry an annotation are *not* skipped — a re-harvest
  appends a fresh generation so a ``REWARD_VERSION`` bump can re-score the whole
  archive while older ``as_of`` pins still resolve their original generation.
  Only the ``cache``/``dry_run`` paths suppress writes.
* **Per-row error isolation:** an exception on one row counts in ``errors`` and
  does not derail subsequent rows. Configuration errors (missing
  ``archive_dir``) raise before the loop begins.
* **Capture-time ``base_sha``:** materialized into the manifest when missing
  (the only fallible git I/O of the annotate pass lives here, not in the pure
  build-corpus projection).

The rubric-assembly helpers live here as harvest's own (the legacy
``labeler.py`` is retired in plan Task 13); the git/``gh`` wrappers are
module-level so the orchestrator and tests can monkeypatch them as
injection seams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.archive.index import append_label_observation, query_runs
from daydream.git_ops import GitError
from daydream.training import reward
from daydream.training.backfill_cache import BackfillCache
from daydream.training.base_sha import materialize_base_sha
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    pr_merge_signal,
)
from daydream.training.reward import ScoringInputs, score_trajectory
from daydream.training.rubric import Rubric, derive_outcome_label
from daydream.ui import create_console, print_warning

_VERDICTS_FILE = "recommendation-verdicts.json"
"""Bronze artifact (under ``deep/``) carrying the ``verdicts`` list."""

_RECORDS_GLOB = "stack-*-records.json"
"""Bronze per-stack finding-record artifacts (under ``deep/``)."""

_REVIEW_OUTPUT_FILE = "review-output.md"
"""Length-proxy artifact; at the run root for shallow runs, under ``deep/`` for deep runs."""


def _read_review_output(run_dir: Path) -> str | None:
    r"""Return the review-output text, or ``None`` when absent.

    Tries ``review-output.md`` at the run root first (shallow-loop layout),
    then ``deep/review-output.md`` (deep-mode layout), mirroring the former
    exporter's back-compat fallback order.

    Args:
        run_dir: The archived run directory.

    Returns:
        The text of the first review-output file found, or ``None`` when
        neither location exists. Non-``FileNotFoundError`` ``OSError``\s
        propagate to the caller.
    """
    for candidate in (run_dir / _REVIEW_OUTPUT_FILE, run_dir / "deep" / _REVIEW_OUTPUT_FILE):
        try:
            return candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return None


def _read_review_output_length(run_dir: Path) -> int | None:
    """Return the review-output char count, or ``None`` when absent.

    Delegates to :func:`_read_review_output`; see that function for the
    fallback order and error semantics.

    Args:
        run_dir: The archived run directory.

    Returns:
        The character count of the first review-output file found, or
        ``None`` when neither location exists.
    """
    text = _read_review_output(run_dir)
    return len(text) if text is not None else None


def assemble_scoring_inputs(run_dir: Path, row: dict[str, Any]) -> ScoringInputs:
    """Reduce one run's bronze artifacts to intrinsic :class:`ScoringInputs`.

    Reads the structured bronze artifacts under ``run_dir/deep`` and the
    review-output length proxy, combining them with the indexed
    ``grounding_rate`` into the capture-time signals the reward reducer
    consumes. Absent artifacts yield the shallow-run path
    (``verifier_verdicts=None``); a present-but-malformed structured
    artifact sets ``format_valid=False`` without raising.

    Args:
        run_dir: The archived run directory (bronze bundle root).
        row: The indexed manifest row; ``row["grounding_rate"]`` supplies the
            grounding axis (``None`` when unavailable).

    Returns:
        A :class:`ScoringInputs` with the verdicts list (or ``None``), the
        passed-through grounding rate, the format-validity gate, and the
        char-count length proxy (or ``None``).
    """
    deep_dir = run_dir / "deep"

    verifier_verdicts: list[dict[str, Any]] | None = None
    format_valid = True

    verdicts_path = deep_dir / _VERDICTS_FILE
    try:
        data = json.loads(verdicts_path.read_text(encoding="utf-8"))
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if isinstance(verdicts, list):
            verifier_verdicts = verdicts
    except FileNotFoundError:
        # Shallow run — no structured verdicts. Nothing failed to parse.
        pass
    except json.JSONDecodeError:
        # Present but malformed structured artifact ⇒ format gate floors.
        format_valid = False

    # Per-stack records are structural bronze too: a present-but-malformed
    # records file also trips the format gate, even though it doesn't feed a
    # reward axis in the minimal reducer.
    if deep_dir.is_dir():
        for records_path in sorted(deep_dir.glob(_RECORDS_GLOB)):
            try:
                json.loads(records_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except json.JSONDecodeError:
                format_valid = False

    return ScoringInputs(
        verifier_verdicts=verifier_verdicts,
        grounding_rate=row.get("grounding_rate"),
        format_valid=format_valid,
        length=_read_review_output_length(run_dir),
    )


# ---------------------------------------------------------------------------
# git / gh wrappers — module-level so they double as monkeypatch seams.
# ---------------------------------------------------------------------------


def _gh_api(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """Proxy to :func:`daydream.git_ops.gh_api` keyed by ``repo`` slug.

    The PR posterior signal extractors call ``gh_api(repo, endpoint, **kwargs)``
    with ``repo`` as a slug string (``"owner/name"``). :func:`git_ops.gh_api`
    takes a ``Path`` as its first argument because it uses ``cwd=repo`` for the
    shell-out. We adapt by using ``Path(".")`` — ``gh api`` works from any cwd
    because it authenticates against the GitHub host configured in ``gh auth``,
    not the local repo.

    Limitation: ``repo`` is accepted for API compatibility but is not used to
    resolve the GitHub host; all requests go to the single host configured in
    ``gh auth`` (typically ``github.com``). Mixing repos from different GitHub
    hosts in a single harvest run would silently use the wrong host.
    """
    return git_ops.gh_api(Path("."), endpoint, **kwargs)


def _diff_name_only(repo: Path, base: str, head: str) -> list[str]:
    """Proxy to :func:`daydream.git_ops.diff_name_only`."""
    return git_ops.diff_name_only(repo, base, head)


def _commits_in_window(repo: Path, head: str, base: str, days: int) -> list[str]:
    """Return commits on ``base`` since ``head``'s ancestor, within ``days``.

    Used by the fix-applied cascade to bound the upstream review window.
    """
    return git_ops.log_shas_since(repo, head, base, since_days=days)


def _commits_since(repo: Path, branch: str, since: str) -> list[str]:
    """Return commits on ``branch`` after ``since``.

    Used by the local-branch posterior path to walk commits pushed after
    the daydream-recorded ``head_sha``.
    """
    return git_ops.log_shas(repo, branch, since=since)


def _file_at(repo: Path, path: str, sha: str) -> str:
    """Return file content at ``sha``; empty string on missing path."""
    try:
        return git_ops.show(repo, sha, path).decode("utf-8", errors="replace")
    except git_ops.GitError:
        return ""


# ---------------------------------------------------------------------------
# Rubric assembly — PR vs local-branch (harvest's own, formerly in labeler.py)
# ---------------------------------------------------------------------------


_FIX_APPLIED_STUB = FixAppliedSignal(
    verdict="unknown",
    hunks_applied=0,
    hunks_total=0,
    window_commits=[],
)
"""Returned when the fix-applied cascade cannot run (missing diff.patch,
empty changed_files, or any subprocess error). The rubric still carries
the field for schema stability; outcome derivation does not depend on it
for the PR-review path."""


def _added_lines(diff_text: str) -> list[str]:
    """Extract the content of ``+`` lines from a unified-diff blob.

    Strips the leading ``+`` and any single leading space (the conventional
    unified-diff content marker). Excludes ``+++`` file headers. Whitespace-
    only lines are dropped so they don't poison the substring check.
    """
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++"):
            continue
        if not line.startswith("+"):
            continue
        content = line[1:]
        if content.startswith(" "):
            content = content[1:]
        content = content.strip()
        if content:
            out.append(content)
    return out


def _local_branch_verdict(
    *,
    diff_text: str,
    commits: list[str],
    repo_clone: Path,
    changed_files: list[str],
    file_at_fetcher: Any,
) -> LocalCommitAppliedSignal:
    """Lenient posterior signal for PR-less runs.

    Returns ``"applied"`` if any commit in ``commits`` yields file content
    (via ``file_at_fetcher``) that contains every added line from
    ``diff_text`` as a substring. ``"rejected"`` if commits exist but none
    match; ``"unknown"`` when there are no commits to inspect.
    """
    if not commits:
        return LocalCommitAppliedSignal(verdict="unknown")
    added = _added_lines(diff_text)
    if not added:
        return LocalCommitAppliedSignal(verdict="rejected")
    paths = changed_files if changed_files else [""]
    for commit in commits:
        for path in paths:
            try:
                content = file_at_fetcher(repo_clone, path, commit)
            except Exception:  # noqa: BLE001 - extractor isolation
                continue
            if all(needle in content for needle in added):
                return LocalCommitAppliedSignal(verdict="applied")
    return LocalCommitAppliedSignal(verdict="rejected")


def _safe_fix_applied(
    row: dict[str, Any],
    *,
    changed_files: list[str],
    repo_clone: Path,
    window_days: int,
) -> FixAppliedSignal:
    """Run :func:`fix_applied_signal`, swallowing missing-data errors.

    The cascade needs ``archive_path/diff.patch`` to exist. When the archive
    directory is missing (older runs, dry fixtures), or when ``changed_files``
    is empty, return :data:`_FIX_APPLIED_STUB`.
    """
    if not changed_files:
        return _FIX_APPLIED_STUB
    try:
        return fix_applied_signal(
            row,
            changed_files=changed_files,
            repo_clone=repo_clone,
            diff_fetcher=_diff_name_only,
            commits_in_window_fetcher=_commits_in_window,
            file_at_fetcher=_file_at,
            window_days=window_days,
        )
    except (FileNotFoundError, OSError, GitError):
        return _FIX_APPLIED_STUB


def _row_changed_files(row: dict[str, Any]) -> list[str]:
    """Return ``changed_files`` from a sqlite row, decoding the JSON column."""
    raw = row.get("changed_files")
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        decoded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return list(decoded) if isinstance(decoded, list) else []


def _row_is_pr(row: dict[str, Any]) -> bool:
    """Return ``True`` when the row has both ``pr_repo`` and ``pr_number``."""
    return bool(row.get("pr_repo")) and row.get("pr_number") is not None


def _build_rubric_pr(
    row: dict[str, Any],
    *,
    gh_api: Any,
    repo_clone: Path,
    window_days: int,
) -> Rubric:
    """Compose all four signals for a row that originated from a PR."""
    changed_files = _row_changed_files(row)
    signal_row = {**row, "changed_files": changed_files}
    pr_merge = pr_merge_signal(signal_row, gh_api=gh_api)
    comments = comment_resolution_signal(signal_row, gh_api=gh_api)
    fix = _safe_fix_applied(
        signal_row,
        changed_files=changed_files,
        repo_clone=repo_clone,
        window_days=window_days,
    )
    return Rubric(
        pr_merge=pr_merge,
        fix_applied=fix,
        comment_resolution=comments,
        local_commit_applied=None,
        posterior_source="pr_review",
    )


def _build_rubric_local(
    row: dict[str, Any],
    *,
    repo_clone: Path,
) -> Rubric:
    """Compose signals for a PR-less row (local-branch posterior)."""
    archive_path = Path(row["archive_path"])
    diff_text = ""
    try:
        diff_text = (archive_path / "diff.patch").read_text()
    except (FileNotFoundError, OSError):
        diff_text = ""
    commits = _commits_since(repo_clone, row.get("branch") or "HEAD", row.get("head_sha") or "")
    local = _local_branch_verdict(
        diff_text=diff_text,
        commits=commits,
        repo_clone=repo_clone,
        changed_files=_row_changed_files(row),
        file_at_fetcher=_file_at,
    )
    pr_merge = PRMergeSignal(merged=False, merged_at=None)
    comments = CommentResolutionSignal(total=0, replied=0, unresolved=0)
    return Rubric(
        pr_merge=pr_merge,
        fix_applied=_FIX_APPLIED_STUB,
        comment_resolution=comments,
        local_commit_applied=local,
        posterior_source="local_branch",
    )


def _pr_state_for_rubric(rubric: Rubric) -> str | None:
    """Map a PR-review rubric to a sqlite ``pr_state`` discriminator.

    For local-branch rubrics returns ``None`` so the column reflects "no PR
    associated".
    """
    if rubric.posterior_source != "pr_review":
        return None
    return "merged" if rubric.pr_merge.merged else "closed"


# ---------------------------------------------------------------------------
# Per-run annotation builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationPayload:
    """One run's bitemporal annotation, ready to persist (no DB writes here).

    Attributes:
        labels: Outcome labels (``[]`` when the derived label is
            ``"unknown"``, else a single-element list).
        pr_state: sqlite ``pr_state`` discriminator (``"merged"``/``"closed"``
            for PR rows, ``None`` for local-branch rows).
        valid_at: The valid-time of the posterior outcome — the PR merge
            timestamp for PR rows, ``None`` for non-PR/local rows (the write
            layer collapses ``None`` → ``observed_at``).
        reward_version: The :data:`daydream.training.reward.REWARD_VERSION`
            observed at scoring time.
        reward_json: ``json.dumps`` of the full
            :class:`~daydream.training.reward.RewardBreakdown.to_dict` so
            re-projection has every axis.
        composite_reward: The cached intrinsic composite scalar (or ``None``
            when uncomputable).
        evidence_sha: The run's ``head_sha`` (the evidence anchor for the
            posterior signals), or ``None``.
    """

    labels: list[str]
    pr_state: str | None
    valid_at: str | None
    reward_version: str
    reward_json: str
    composite_reward: float | None
    evidence_sha: str | None
    rubric_json: str | None


def build_annotation(
    row: dict[str, Any],
    *,
    run_dir: Path,
    gh_api: Any,
    repo_clone: Path,
    window_days: int,
) -> AnnotationPayload:
    """Build one run's bitemporal annotation payload (pure of DB writes).

    Composes the posterior rubric (PR vs local-branch, mirroring the former
    labeler's assembly), derives the outcome label, assembles the intrinsic
    :class:`ScoringInputs` from bronze, and scores a
    :class:`~daydream.training.reward.RewardBreakdown`. Returns a frozen
    :class:`AnnotationPayload`; the orchestrator persists it. Posterior-fetch
    and git errors propagate to the caller (the orchestrator isolates per-row).

    Args:
        row: The indexed manifest row (carries ``session_id``, the PR
            discriminators, ``head_sha``, ``grounding_rate``, etc.).
        run_dir: The archived run directory (bronze bundle root) feeding the
            intrinsic reward signals.
        gh_api: Callable invoked as ``gh_api(repo, endpoint, **kwargs)`` by
            the PR posterior signals; unused on the local-branch path.
        repo_clone: Local clone root for the fix-applied / local-commit
            cascades.
        window_days: Lookback window for the fix-applied cascade.

    Returns:
        A frozen :class:`AnnotationPayload`. ``valid_at`` is the PR merge
        timestamp for PR rows and ``None`` for non-PR/local rows.

    Raises:
        Exception: Any posterior-fetch (``gh_api``) or git error from the
            rubric assembly propagates unchanged to the caller.
    """
    if _row_is_pr(row):
        rubric = _build_rubric_pr(
            row,
            gh_api=gh_api,
            repo_clone=repo_clone,
            window_days=window_days,
        )
        valid_at = rubric.pr_merge.merged_at
    else:
        rubric = _build_rubric_local(row, repo_clone=repo_clone)
        valid_at = None

    outcome_label = derive_outcome_label(rubric)
    labels = [outcome_label] if outcome_label != "unknown" else []

    inputs = assemble_scoring_inputs(run_dir, row)
    rb = score_trajectory(inputs)

    return AnnotationPayload(
        labels=labels,
        pr_state=_pr_state_for_rubric(rubric),
        valid_at=valid_at,
        reward_version=reward.REWARD_VERSION,
        reward_json=json.dumps(rb.to_dict()),
        composite_reward=rb.composite,
        evidence_sha=row.get("head_sha"),
        rubric_json=json.dumps(rubric.to_dict()),
    )


# ---------------------------------------------------------------------------
# Repo resolution — three-tier priority: source_path → clone cache → None
# ---------------------------------------------------------------------------


def _resolve_repo_for_row(
    row: dict[str, Any],
    clone_cache: Path | None,
    *,
    fetched_repos: set[Path] | None = None,
    console: Console | None = None,
) -> Path | None:
    """Resolve a local repo working tree for a manifest row.

    Priority:
        1. ``row["source_path"]`` when it exists on disk with a ``.git`` dir.
        2. Clone cache: ``clone_cache/<owner>/<repo>/`` — fetch if present, clone if not.
        3. ``None`` when no source is available.

    Clone/fetch failures are caught and logged; they never block harvest.

    Args:
        row: An indexed manifest row (supplies ``source_path``, ``remote_url``, ``repo_slug``).
        clone_cache: Root directory for cached clones, or ``None`` to skip cloning.

    Returns:
        Path to a usable working tree, or ``None``.
    """
    source_path = row.get("source_path")
    if source_path and (Path(source_path) / ".git").exists():
        return Path(source_path)

    remote_url = row.get("remote_url")
    repo_slug = row.get("repo_slug")
    if not remote_url or not repo_slug or clone_cache is None:
        return None

    parts = repo_slug.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None

    cached_repo = clone_cache / parts[0] / parts[1]
    try:
        if (cached_repo / ".git").exists():
            if fetched_repos is None or cached_repo not in fetched_repos:
                git_ops.fetch(cached_repo)
                if fetched_repos is not None:
                    fetched_repos.add(cached_repo)
        else:
            cached_repo.parent.mkdir(parents=True, exist_ok=True)
            git_ops.clone(remote_url, cached_repo, blobless=True)
    except (GitError, OSError) as exc:
        print_warning(
            console or create_console(),
            f"harvest: repo resolution failed for {repo_slug}: {type(exc).__name__}: {exc}",
        )
        if not (cached_repo / ".git").exists():
            return None
    return cached_repo


def _materialize_base_sha_if_missing(
    row: dict[str, Any], run_dir: Path, repo_clone: Path | None, *, console: Console | None = None
) -> None:
    """Opportunistically backfill ``code_context.base_sha`` into the manifest.

    Only acts when ``manifest.json`` exists AND ``repo_clone`` is available.
    Any failure is swallowed (opportunistic), leaving ``base_sha`` as ``None``.

    Args:
        row: The indexed manifest row.
        run_dir: The archived run directory (holds ``manifest.json``).
        repo_clone: Resolved repo working tree, or ``None``.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    if repo_clone is None:
        return
    try:
        materialize_base_sha(manifest_path, repo_clone=repo_clone)
    except (OSError, json.JSONDecodeError, GitError) as exc:
        print_warning(
            console or create_console(),
            f"harvest: base_sha backfill failed for {manifest_path}: {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Orchestrator — append-only, re-runnable, per-row isolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarvestConfig:
    """Configuration for a single :func:`run_harvest` invocation.

    Mirrors the retired labeler's config shape (plan Task 6).

    Attributes:
        archive_dir: Path to the daydream archive root (contains ``index.db``).
        dry_run: When ``True``, the loop builds annotations but suppresses the
            write to ``label_observations`` and the resume log.
        cache_dir: Optional directory backing
            :class:`~daydream.training.backfill_cache.BackfillCache`. When
            ``None``, ``gh_api`` calls hit the network on every row.
        repo_clone_root: Optional root under which per-repo clones live (used
            by the fix-applied / local-commit cascades). Falls back to
            ``cache_dir / 'repos'`` when unset (or ``None`` if ``cache_dir``
            is also unset).
        session_filter: Optional ``session_id`` prefix to restrict the queue.
        fix_applied_window_days: Lookback window for upstream commits
            considered by the fix-applied cascade.
        gh_request_spacing_sec: Sleep duration between rows to spread
            ``gh api`` calls under GitHub's secondary rate limits.
    """

    archive_dir: Path
    dry_run: bool = False
    cache_dir: Path | None = None
    repo_clone_root: Path | None = None
    session_filter: str | None = None
    fix_applied_window_days: int = 30
    gh_request_spacing_sec: float = 0.8


async def run_harvest(config: HarvestConfig) -> dict[str, int]:
    """Walk the archive and append one fresh annotation per indexed run.

    The single deferred annotate pass: for every indexed run, materialize the
    capture-time ``base_sha`` (when missing), build the bitemporal annotation
    (label + intrinsic reward + ``valid_at``) via :func:`build_annotation`, and
    append it through :func:`daydream.archive.index.append_label_observation`.

    **Append-only and re-runnable:** rows that already carry an annotation are
    *not* skipped — a re-harvest appends a fresh generation, so a later
    ``REWARD_VERSION`` bump can re-score the whole archive while older
    ``as_of`` pins still resolve their original generation. Only the
    ``cache``/``dry_run`` paths suppress writes.

    Per-row error isolation: an exception on one row counts in ``errors`` and
    does not derail subsequent rows. Configuration errors (missing
    ``archive_dir``) raise before the loop begins.

    Args:
        config: A :class:`HarvestConfig` instance.

    Returns:
        Summary dict with keys ``considered``, ``annotated``,
        ``would_annotate``, ``skipped``, ``errors``.

    Raises:
        FileNotFoundError: When ``config.archive_dir`` does not exist.
            Configuration errors deliberately surface before the loop.
    """
    if not config.archive_dir.exists():
        msg = f"archive_dir does not exist: {config.archive_dir}"
        raise FileNotFoundError(msg)

    # Build the queue: every indexed run, optionally prefix-filtered. Unlike
    # the old labeler, we do NOT drop rows that already carry an annotation —
    # harvest is append-only and re-runnable.
    if config.session_filter:
        queue = query_runs(
            config.archive_dir,
            "session_id LIKE ? || '%'",
            (config.session_filter,),
        )
    else:
        queue = query_runs(config.archive_dir)

    # Cache + resume integration. The cache wraps the gh seam and tracks
    # completed sessions so an interrupted run can resume without re-fetching.
    cache: BackfillCache | None = None
    gh_api_callable: Any = _gh_api
    if config.cache_dir is not None:
        cache = BackfillCache(cache_dir=config.cache_dir, inner=_gh_api)
        gh_api_callable = cache
        done = cache.completed_sessions()
        queue = [row for row in queue if row["session_id"] not in done]

    clone_cache = config.repo_clone_root or (config.cache_dir / "repos" if config.cache_dir else None)
    fetched_repos: set[Path] = set()

    summary = {
        "considered": len(queue),
        "annotated": 0,
        "would_annotate": 0,
        "skipped": 0,
        "errors": 0,
    }

    console = create_console()

    for row in queue:
        try:
            run_dir = Path(row["archive_path"])
            row_repo_clone = _resolve_repo_for_row(
                row, clone_cache=clone_cache, fetched_repos=fetched_repos, console=console
            )
            _materialize_base_sha_if_missing(row, run_dir, repo_clone=row_repo_clone, console=console)
            payload = build_annotation(
                row,
                run_dir=run_dir,
                gh_api=gh_api_callable,
                repo_clone=row_repo_clone or config.archive_dir,
                window_days=config.fix_applied_window_days,
            )
            if config.dry_run:
                summary["would_annotate"] += 1
            else:
                append_label_observation(
                    config.archive_dir,
                    row["session_id"],
                    labels=payload.labels,
                    pr_state=payload.pr_state,
                    labeler_version=reward.REWARD_VERSION,
                    evidence_sha=payload.evidence_sha,
                    rubric_json=payload.rubric_json,
                    valid_at=payload.valid_at,
                    reward_version=payload.reward_version,
                    reward_json=payload.reward_json,
                    composite_reward=payload.composite_reward,
                )
                summary["annotated"] += 1
                if cache is not None:
                    cache.mark_session_done(row["session_id"])
        except Exception as exc:  # noqa: BLE001 - per-row isolation by design
            summary["errors"] += 1
            print_warning(
                console,
                f"harvest: session {row.get('session_id', '<unknown>')} failed: "
                f"{type(exc).__name__}: {exc}",
            )
            continue

        if not config.dry_run:
            await anyio.sleep(config.gh_request_spacing_sec)

    return summary
