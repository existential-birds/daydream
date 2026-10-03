"""Validate immutable diagram evidence and publish standalone diagrams."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jsonschema

from daydream import git_ops
from daydream.config import DIAGRAM_KINDS
from daydream.git_ops import INHERIT_GITHUB_AUTH, GitError, GitHubAuth, PathAbsentError
from daydream.repository_paths import strip_dot_slash, valid_repository_file_path
from daydream.reviews.identity import DAYDREAM_FOOTER, diagram_marker
from daydream.reviews.models import PRInfo
from daydream.ui import print_error, print_success, print_warning

if TYPE_CHECKING:
    from rich.console import Console

    from daydream.deep.diagram_types import SequenceSpec
    from daydream.findings import FindingsArtifact


def post_diagram_comment_to_pr(
    target_dir: Path,
    pr: PRInfo,
    *,
    body: str,
    kinds: list[str],
    bot_login: str | None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> tuple[str | None, str | None]:
    """Post an attributed diagram comment, then minimize prior comments of the same kinds.

    A failed replacement leaves prior diagrams visible. Missing bot identity disables
    minimization, and minimization failures warn without undoing the new comment.
    Return (URL, None) on success or (None, optional error) on failure."""
    from daydream.agent import console
    from daydream.reconcile import fetch_prior_diagram_comments, minimize_comment

    repo_slug = f"{pr.owner}/{pr.repo}"
    prior = []
    if bot_login is None:
        print_warning(
            console,
            "BOT_LOGIN_UNRESOLVED: no bot login resolvable; skipping minimization of "
            "prior diagram comments (a stale diagram may stay unfolded, but no "
            "comment is ever minimized on an unproven author).",
        )
    else:
        try:
            prior = fetch_prior_diagram_comments(
                target_dir, repo_slug, pr.number, bot_login=bot_login, auth=auth
            )
        except GitError as exc:
            print_warning(console, f"Could not inventory prior diagram comments: {exc}")
            prior = []

    markers = "\n".join(diagram_marker(kind, pr.head_sha) for kind in kinds)
    chunks = [chunk for chunk in (markers, body.strip(), DAYDREAM_FOOTER) if chunk]
    endpoint = f"/repos/{pr.owner}/{pr.repo}/issues/{pr.number}/comments"
    try:
        data = git_ops.gh_api(
            target_dir,
            endpoint,
            method="POST",
            input_data={"body": "\n\n".join(chunks)},
            auth=auth,
        )
    except GitError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, None
    url = data.get("html_url")
    if not url:
        return None, None

    current_kinds = set(kinds)
    for comment in prior:
        if not set(comment.kinds) & current_kinds:
            continue
        if not minimize_comment(target_dir, comment.node_id, auth=auth):
            print_warning(
                console,
                f"Failed to minimize prior diagram comment {comment.node_id}",
            )
    return str(url), None


def diagram_comment_kinds(payload: dict[str, Any]) -> list[str]:
    """Return attempted eligible kinds in render order, including omitted results.

    A replacement supersedes prior comments only for its own requested kinds."""
    eligibility = payload.get("eligibility")
    if not isinstance(eligibility, dict):
        return []
    kinds: list[str] = []
    for kind in DIAGRAM_KINDS:
        decision = eligibility.get(kind)
        if isinstance(decision, dict) and decision.get("eligible"):
            kinds.append(kind)
    return kinds


def _diagram_results(payload: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Treat absent or malformed per-kind results as missing blocks."""
    raw = payload.get("results")
    results: dict[str, dict[str, Any] | None] = {}
    for kind in DIAGRAM_KINDS:
        value = raw.get(kind) if isinstance(raw, dict) else None
        results[kind] = value if isinstance(value, dict) else None
    return results


def render_diagram_blocks_from_payload(payload: dict[str, Any]) -> str:
    """Render folded review blocks; explicit diagram requests also need omission notices."""
    from daydream.deep.diagram_render import render_diagram_blocks

    return render_diagram_blocks(_diagram_results(payload))


def render_diagram_comment_body(payload: dict[str, Any]) -> str:
    """Re-render specs for explicit requests, including eligible kinds' omission notices.

    Both live and artifact posting use this renderer. Stored mermaid is never trusted."""
    from daydream.deep.diagram_render import render_omission_notice

    results = _diagram_results(payload)
    chunks: list[str] = []
    blocks = render_diagram_blocks_from_payload(payload)
    if blocks:
        chunks.append(blocks)
    for kind in diagram_comment_kinds(payload):
        result = results.get(kind)
        if result is None or result.get("status") == "rendered":
            continue
        notice = render_omission_notice(kind, result)
        if notice:
            chunks.append(notice)
    if not chunks:
        chunks.append(
            "No grounded diagram was eligible for this pull request: the change "
            "does not cross a module or service boundary and no changed function "
            "gained enough branch points to be worth charting."
        )
    return "\n\n".join(chunks)


class _HeadEvidence:
    """Read immutable head bytes from local Git or, without checkout, GitHub's contents API."""

    def __init__(
        self,
        target_dir: Path,
        head_sha: str,
        repo_slug: str | None,
        *,
        auth: GitHubAuth,
    ) -> None:
        self._target_dir = target_dir
        self._head_sha = head_sha
        self._repo_slug = repo_slug
        self._auth = auth
        self._local: bool | None = None
        self._bytes: dict[str, bytes] = {}
        self._lines: dict[str, list[str]] = {}

    def _reads_locally(self) -> bool:
        """Whether ``target_dir`` is a repository that already holds the head commit."""
        if self._local is None:
            try:
                self._local = git_ops.commit_exists(self._target_dir, self._head_sha)
            except GitError:
                self._local = False
        return self._local

    def read(self, path: str) -> bytes:
        """Return *path*'s bytes at the head SHA.

        Raises:
            PathAbsentError: If either source proved *path* absent at that commit.
            GitError: If the file could not be read at all.
        """
        cached = self._bytes.get(path)
        if cached is not None:
            return cached
        # The spec schema's ``pattern`` is applied with ``re.search``, whose
        # ``$`` matches before a trailing newline, so the grammar is re-checked
        # here as a fullmatch — the sequence branch writes these paths into a
        # snapshot tree, and the API branch into an endpoint.
        if not valid_repository_file_path(path):
            raise GitError("invalid repository file path")
        if self._reads_locally():
            data = git_ops.show(self._target_dir, self._head_sha, path)
        elif self._repo_slug is None:
            raise GitError(f"{self._target_dir} does not contain commit {self._head_sha}")
        else:
            data = git_ops.gh_file_at_ref(
                self._target_dir, self._repo_slug, self._head_sha, path, auth=self._auth
            )
        self._bytes[path] = data
        return data

    def lines(self, path: str) -> list[str]:
        """Return *path*'s decoded lines at the head SHA (same raises as :meth:`read`)."""
        cached = self._lines.get(path)
        if cached is None:
            cached = self.read(path).decode(errors="replace").splitlines()
            self._lines[path] = cached
        return cached


def _diagram_head_evidence_problem(
    kind: str,
    spec: dict[str, Any],
    head: _HeadEvidence,
) -> str | None:
    """Return a problem when a rendered diagram citation is absent at the head SHA."""
    from daydream.tree_sitter_index import (
        StatementLines,
        definitions_in_file,
        language_for_path,
        statement_lines,
    )

    def unreadable(path: str, exc: GitError) -> str:
        """Separate invalid paths, proven absence, and operational read failures."""
        if not valid_repository_file_path(path):
            return f"{kind} diagram evidence cites an invalid repository path: {path!r}"
        if isinstance(exc, PathAbsentError):
            return f"{kind} diagram evidence is missing from immutable head: {path}"
        return f"{kind} diagram evidence could not be read from immutable head: {exc}"

    def check(evidence: dict[str, Any]) -> str | None:
        path = evidence["file"]
        try:
            lines = head.lines(path)
        except GitError as exc:
            return unreadable(path, exc)
        if evidence["line"] > len(lines):
            return (
                f"{kind} diagram evidence line {evidence['line']} is missing from "
                f"immutable head: {path}"
            )
        return None

    if kind == "sequence":
        paths = {
            path
            for participant in spec["participants"]
            for path in participant["files"]
        }
        paths.update(message["evidence"]["file"] for message in spec["messages"])
        paths.update(
            branch["evidence"]["file"]
            for block in spec["blocks"]
            for branch in block["branches"]
        )
        with tempfile.TemporaryDirectory() as snapshot_dir:
            snapshot_root = Path(snapshot_dir)
            for path in paths:
                try:
                    source = head.read(path)
                except GitError as exc:
                    return unreadable(path, exc)
                snapshot_path = snapshot_root / path
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.write_bytes(source)

            # Strict wire schemas admit empty names, empty service strings,
            # path aliases and integral floats. Those are not the canonical
            # sequence values the old grounding projection compared to the
            # original artifact. Preserve that refusal here without applying
            # the author's tolerant text stripping or duplicate-index salvage.
            if any(
                not participant["name"]
                or participant["service"] == ""
                or any(strip_dot_slash(path) != path for path in participant["files"])
                for participant in spec["participants"]
            ) or any(
                branch["condition"].strip() != branch["condition"]
                or any(not isinstance(index, int) for index in branch["messages"])
                for block in spec["blocks"]
                for branch in block["branches"]
            ):
                return "sequence diagram evidence is not grounded in immutable head"

            from daydream.deep.diagram_grounding import RepoSymbols, ground_sequence

            report = ground_sequence(
                cast("SequenceSpec", spec),
                repo_root=snapshot_root,
                hunk_ranges={},
                symbols=RepoSymbols(snapshot_root),
            )
        if report.spec_final != spec:
            return "sequence diagram evidence is not grounded in immutable head"
    else:
        root = spec["root"]
        problem = check(root)
        if problem is not None:
            return problem
        with tempfile.TemporaryDirectory() as snapshot_dir:
            snapshot_root = Path(snapshot_dir)
            snapshot_path = snapshot_root / root["file"]
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(head.read(root["file"]))
            try:
                definitions = definitions_in_file(snapshot_root, root["file"])
            except Exception:
                definitions = []  # Node checks below keep their parser fallback.
        root_end = next((
            definition["end_line"] for definition in definitions
            if definition["kind"] == "function"
            and definition["name"] == root["name"] and definition["line"] == root["line"]
        ), None)
        if definitions and root_end is None:
            return "flowchart root is not a function in immutable head"
        source_lines: dict[str, StatementLines] = {}
        for node in spec["nodes"]:
            evidence = node["evidence"]
            problem = check(evidence)
            if problem is not None:
                return problem
            line = evidence["line"]
            if isinstance(root_end, int) and not root["line"] <= line <= root_end:
                return "flowchart diagram node lies outside its root in immutable head"
            try:
                path = evidence["file"]
                if path not in source_lines:
                    source_lines[path] = statement_lines(
                        language_for_path(path), "\n".join(head.lines(path)).encode(),
                    )
                captured = source_lines[path]
                grounded = line in (
                    captured.terminals if node["kind"] == "end"
                    else captured.branches if node["kind"] == "decision"
                    else captured.executable
                )
            except Exception:
                grounded = False
            if not grounded:
                return (
                    f"flowchart diagram evidence line {line} is not a valid "
                    f"{node['kind']} node in immutable head: {evidence['file']}"
                )
    return None


def validate_diagram_payload(
    payload: dict[str, Any],
    *,
    target_dir: Path | None = None,
    head_sha: str | None = None,
    repo_slug: str | None = None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> str | None:
    """Validate model-derived specs, render caps, references, and immutable head evidence.

    Findings artifacts allow arbitrary diagram payloads, so posting must validate before
    re-rendering. Without a local head commit, evidence comes from GitHub at the event SHA.
    Return the first problem, or None when all rendered kinds validate."""
    from daydream.config import (
        DIAGRAM_MAX_BLOCKS,
        DIAGRAM_MAX_EDGES,
        DIAGRAM_MAX_MESSAGES,
        DIAGRAM_MAX_NODES,
        DIAGRAM_MAX_PARTICIPANTS,
    )
    from daydream.deep.diagram_schema import FLOWCHART_SPEC_SCHEMA, SEQUENCE_SPEC_SCHEMA

    schemas = {"sequence": SEQUENCE_SPEC_SCHEMA, "flowchart": FLOWCHART_SPEC_SCHEMA}
    caps = {
        "sequence": {
            "participants": DIAGRAM_MAX_PARTICIPANTS,
            "messages": DIAGRAM_MAX_MESSAGES,
            "blocks": DIAGRAM_MAX_BLOCKS,
        },
        "flowchart": {"nodes": DIAGRAM_MAX_NODES, "edges": DIAGRAM_MAX_EDGES},
    }
    results = payload.get("results")
    if not isinstance(results, dict):
        return "diagrams payload has no 'results' object"
    head = (
        _HeadEvidence(target_dir, head_sha, repo_slug, auth=auth)
        if target_dir is not None and head_sha is not None
        else None
    )
    for kind in DIAGRAM_KINDS:
        result = results.get(kind)
        if not isinstance(result, dict) or result.get("status") != "rendered":
            continue
        try:
            jsonschema.validate(result.get("spec_final"), schemas[kind])
        except jsonschema.ValidationError as exc:
            return f"{kind} spec_final failed schema validation: {exc.message}"
        spec = result["spec_final"]
        for collection, cap in caps[kind].items():
            size = len(spec[collection])
            if size > cap:
                return f"{kind} spec_final exceeds {collection} render cap: {size} > {cap}"
        if result.get("reason") is not None or result.get("omit_reasons") not in (None, []):
            return f"{kind} rendered result contradicts its status"
        if kind == "sequence":
            names = {participant["name"] for participant in spec["participants"]}
            if len(names) != len(spec["participants"]):
                return "sequence spec_final has duplicate participant names"
            for index, message in enumerate(spec["messages"]):
                if message["from"] not in names or message["to"] not in names:
                    return f"sequence spec_final message {index} has an unknown participant"
            for index, block in enumerate(spec["blocks"]):
                if any(
                    message >= len(spec["messages"])
                    for branch in block["branches"] for message in branch["messages"]
                ):
                    return f"sequence spec_final block {index} cites an unknown message"
        else:
            node_ids = {node["id"] for node in spec["nodes"]}
            if len(node_ids) != len(spec["nodes"]):
                return "flowchart spec_final has duplicate node ids"
            for index, node in enumerate(spec["nodes"]):
                if node["evidence"]["file"] != spec["root"]["file"]:
                    return f"flowchart spec_final node {index} lies outside its root"
            for index, edge in enumerate(spec["edges"]):
                if edge["from"] not in node_ids or edge["to"] not in node_ids:
                    return f"flowchart spec_final edge {index} has an unknown endpoint"
        if head is not None:
            problem = _diagram_head_evidence_problem(kind, spec, head)
            if problem is not None:
                return problem
    return None


def post_diagram_artifact(
    artifact: "FindingsArtifact",
    pr: PRInfo,
    target_dir: Path,
    *,
    console: Console,
    bot_login: str | None,
    auth: GitHubAuth,
) -> int:
    """Validate and post a standalone diagram artifact; return 0 on success, 1 on failure."""
    payload = artifact.diagrams
    if not isinstance(payload, dict):
        print_error(
            console,
            "Diagram Artifact Rejected",
            "artifact declares kind 'diagram' but carries no 'diagrams' payload",
        )
        return 1
    problem = validate_diagram_payload(
        payload,
        target_dir=target_dir,
        head_sha=pr.head_sha,
        repo_slug=f"{pr.owner}/{pr.repo}",
        auth=auth,
    )
    if problem is not None:
        print_error(console, "Diagram Artifact Rejected", problem)
        return 1
    body = render_diagram_comment_body(payload)
    kinds = diagram_comment_kinds(payload)
    url, error = post_diagram_comment_to_pr(
        target_dir, pr, body=body, kinds=kinds, bot_login=bot_login, auth=auth
    )
    if url is None:
        suffix = f" ({error})" if error else ""
        print_error(console, "Diagram Comment Post Failed", f"No comment was posted.{suffix}")
        return 1
    print_success(console, f"Posted diagram comment: {url}")
    return 0
