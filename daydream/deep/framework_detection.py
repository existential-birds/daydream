"""Bounded, local framework evidence for JavaScript/TypeScript review routing.

Only React has a built-in framework reviewer. All other, conflicting, unavailable,
or unsupported evidence deliberately selects the generic reviewer.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from tree_sitter import Node

from daydream._tree_sitter_safety import TreeSitterBadVersionError
from daydream.tree_sitter_index.runtime import get_parser

_MAX_SOURCE_FILES = 256
_MAX_SOURCE_BYTES = 256 * 1024
_MAX_METADATA_FILES = 128
_MAX_METADATA_BYTES = 64 * 1024
_MAX_TOTAL_BYTES = 4 * 1024 * 1024
JS_SOURCE_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})
_JSON_COMMENTS = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n\r]*|/\*[\s\S]*?\*/')
_JSON_TRAILING_COMMAS = re.compile(r'("(?:\\.|[^"\\])*")|,\s*(?=[}\]])')
_JSX_PRAGMA = re.compile(r"@jsx(ImportSource|Runtime|Factory)?\s+([^\s*]+)")


def _framework(module: str) -> str | None:
    if module in {"react", "react-dom", "react-native"} or module.startswith(("react/", "react-dom/")):
        return "react"
    if module.startswith("@angular/"):
        return "angular"
    if module == "vue" or module.startswith(("vue/", "@vue/")):
        return "vue"
    if module == "svelte" or module.startswith("svelte/"):
        return "svelte"
    if module in {"preact", "solid-js", "inferno", "@builder.io/qwik", "@stencil/core"} or module.startswith(
        ("preact/", "solid-js/", "inferno/", "@builder.io/qwik/")
    ):
        return "other-jsx"
    if module.endswith(("/jsx-runtime", "/jsx-dev-runtime")):
        return "other-jsx"
    return None


def _runtime(module: str) -> str:
    return _framework(module) or "other-jsx"


def _json_object(text: str) -> dict[str, Any] | None:
    # tsconfig accepts JSON comments/trailing commas; preserve quoted bytes.
    text = _JSON_COMMENTS.sub(lambda match: match.group(1) or " ", text)
    text = _JSON_TRAILING_COMMAS.sub(lambda match: match.group(1) or "", text)
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _dependencies(package: dict[str, Any]) -> set[str] | None:
    found: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        dependencies = package.get(key, {})
        if not isinstance(dependencies, dict):
            return None
        for name, version in dependencies.items():
            if not isinstance(name, str) or not isinstance(version, str):
                return None
            framework = _framework(name)
            if framework is not None:
                found.add(framework)
    return found


def _jsx_options(options: dict[str, Any]) -> set[str] | None:
    found: set[str] = set()
    source = options.get("jsxImportSource")
    if source is not None:
        if not isinstance(source, str) or not source:
            return None
        found.add(_runtime(source))
    for key in ("jsxFactory", "jsxFragmentFactory", "reactNamespace"):
        factory = options.get(key)
        if factory is not None:
            if not isinstance(factory, str) or not factory:
                return None
            found.add("react" if factory.split(".")[0] == "React" else "other-jsx")
    jsx = options.get("jsx")
    if jsx is not None:
        if not isinstance(jsx, str) or jsx not in {"react", "react-jsx", "react-jsxdev", "preserve", "react-native"}:
            return None
        if not found and jsx in {"react", "react-jsx", "react-jsxdev", "react-native"}:
            found.add("react")
    return found


class FrameworkInspection:
    """Per-detection-run reads and budgets; callers inspect changed paths sorted."""

    def __init__(self, target_dir: Path | None) -> None:
        self.root = target_dir.resolve() if target_dir is not None else None
        self._source_count = 0
        self._metadata_count = 0
        self._bytes = 0
        self._metadata: dict[Path, dict[str, Any] | None] = {}
        self._sources: dict[str, str] = {}

    def _safe_path(self, path: Path) -> bool:
        if self.root is None:
            return False
        try:
            relative = path.relative_to(self.root)
            if ".." in relative.parts:
                return False
            current = self.root
            for part in relative.parts:
                current /= part
                if current.is_symlink():
                    return False
            return True
        except (OSError, ValueError):
            return False

    def _path(self, path: str) -> Path | None:
        relative = PurePosixPath(path)
        if self.root is None or relative.is_absolute() or ".." in relative.parts or "\\" in path:
            return None
        result = self.root.joinpath(*relative.parts)
        return result if self._safe_path(result) else None

    @staticmethod
    def _present(path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            pass  # An inaccessible candidate is a boundary, never permission to inherit.
        return True

    def _read(self, path: Path, limit: int) -> str | None:
        if not self._safe_path(path):
            return None
        try:
            # Open without following the final component; reject parent symlinks above.
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not path.is_file() or before.st_size > min(limit, _MAX_TOTAL_BYTES - self._bytes):
                    return None
                content = stream.read(before.st_size)
                self._bytes += len(content)
                after = os.fstat(stream.fileno())
                if len(content) != before.st_size or (before.st_size, before.st_mtime_ns) != (
                    after.st_size, after.st_mtime_ns
                ):
                    return None
            return content.decode("utf-8")
        except (OSError, UnicodeError):
            return None

    def _meta(self, path: Path) -> dict[str, Any] | None:
        if path not in self._metadata:
            value = None
            if self._metadata_count < _MAX_METADATA_FILES:
                self._metadata_count += 1
                text = self._read(path, _MAX_METADATA_BYTES)
                if text is not None:
                    if path.name == "pnpm-workspace.yaml":
                        try:
                            parsed = yaml.safe_load(text)
                            value = parsed if isinstance(parsed, dict) else None
                        except (yaml.YAMLError, RecursionError):
                            pass
                    elif path.name == "package.json":
                        try:
                            parsed = json.loads(text)
                            value = parsed if isinstance(parsed, dict) else None
                        except (ValueError, RecursionError):
                            pass
                    else:
                        value = _json_object(text)
            self._metadata[path] = value
        return self._metadata[path]

    def _workspace_boundary(self, directory: Path, source_dir: Path, patterns: object) -> bool | None:
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            return None
        for pattern in patterns:
            pattern = pattern.removeprefix("!").rstrip("/")
            if not pattern or PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts:
                return None
            current = source_dir
            while current != directory:
                if fnmatch.fnmatchcase(current.relative_to(directory).as_posix(), pattern):
                    return True
                current = current.parent
        return False

    def _owner(self, directory: Path) -> tuple[Path, dict[str, Any] | None]:
        current = directory
        while True:
            for name in ("pnpm-workspace.yaml", "package.json"):
                candidate = current / name
                if not self._present(candidate):
                    continue
                metadata = self._meta(candidate)
                if metadata is None:
                    return current, None
                if name == "pnpm-workspace.yaml":
                    patterns = metadata.get("packages")
                else:
                    patterns = metadata.get("workspaces", [])
                    if isinstance(patterns, dict):
                        patterns = patterns.get("packages")
                boundary = self._workspace_boundary(current, directory, patterns)
                if boundary is not False:
                    return current, None
                if name == "package.json":
                    return current, metadata
            if current == self.root:
                return current, {}
            current = current.parent

    def _tsconfig(self, path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any] | None:
        if path in seen:
            return None
        metadata = self._meta(path)
        if metadata is None:
            return None
        options: dict[str, Any] = {}
        extends = metadata.get("extends")
        if extends is not None:
            # Package specifiers/arrays/remote configs need a resolver; fall back.
            if not isinstance(extends, str) or not extends.startswith("."):
                return None
            parent = Path(os.path.normpath(path.parent / extends))
            if parent.suffix != ".json":
                parent = Path(str(parent) + ".json")
            if not self._safe_path(parent):
                return None
            inherited = self._tsconfig(parent, seen | {path})
            if inherited is None:
                return None
            options.update(inherited)
        local = metadata.get("compilerOptions", {})
        if not isinstance(local, dict):
            return None
        options.update(local)
        return options

    def _local_evidence(self, path: Path) -> set[str] | None:
        owner_dir, package = self._owner(path.parent)
        if package is None:
            return None
        found = _dependencies(package)
        if found is None:
            return None
        current = path.parent
        while True:
            candidate = current / "tsconfig.json"
            if self._present(candidate):
                options = self._tsconfig(candidate)
                runtime = _jsx_options(options) if options is not None else None
                return found | runtime if runtime is not None else None
            if current == owner_dir:
                return found
            current = current.parent

    def config_stack(self, path: str) -> str:
        """Use a package/config's own local ownership, never co-changed source."""
        target = self._path(path)
        if target is None or not self._present(target):
            return "generic"
        if target.name == "package.json":
            metadata = self._meta(target)
            evidence = _dependencies(metadata) if metadata is not None else None
        elif target.name == "tsconfig.json":
            evidence = self._local_evidence(target)
        else:
            evidence = None
        return "react" if evidence == {"react"} else "generic"

    def source_stack(self, path: str) -> str:
        """Direct source evidence wins; only actual JSX consults local metadata."""
        if path in self._sources:
            return self._sources[path]
        result = "generic"
        target = self._path(path)
        if (target is not None and target.suffix.lower() in JS_SOURCE_SUFFIXES
                and self._source_count < _MAX_SOURCE_FILES):
            self._source_count += 1
            text = self._read(target, _MAX_SOURCE_BYTES)
            if text is not None:
                result = self._source_evidence(target, text)
        self._sources[path] = result
        return result

    def _source_evidence(self, path: Path, text: str) -> str:
        try:
            parser = get_parser("typescript" if path.suffix.lower() in {".ts", ".mjs", ".cjs"} else "tsx")
        except TreeSitterBadVersionError:
            return "generic"
        if parser is None:
            return "generic"
        tree = parser.parse(text.encode("utf-8"))
        if tree.root_node.has_error:
            return "generic"
        evidence: set[str] = set()
        jsx = False
        pragma_runtime = False
        pragma_factory = False
        nodes = [tree.root_node]
        while nodes:
            node = nodes.pop()
            nodes.extend(node.named_children)
            if node.type in {"jsx_element", "jsx_self_closing_element", "jsx_fragment"}:
                jsx = True
            module: Node | None = None
            if node.type in {"import_statement", "export_statement"}:
                module = node.child_by_field_name("source")
            elif node.type == "call_expression":
                function = node.child_by_field_name("function")
                arguments = node.child_by_field_name("arguments")
                if function is not None and function.text in {b"require", b"import"} and arguments is not None:
                    if len(arguments.named_children) == 1 and arguments.named_children[0].type == "string":
                        module = arguments.named_children[0]
            if module is not None and module.text is not None:
                framework = _framework(module.text.decode("utf-8")[1:-1])
                if framework is not None:
                    evidence.add(framework)
            if node.type == "comment" and node.text is not None:
                for match in _JSX_PRAGMA.finditer(node.text.decode("utf-8")):
                    kind, value = match.groups()
                    if kind == "ImportSource":
                        pragma_factory = True
                        evidence.add(_runtime(value))
                    elif kind in {None, "Factory"}:
                        pragma_factory = True
                        evidence.add("react" if value.split(".")[0] == "React" else "other-jsx")
                    elif value in {"automatic", "classic"}:
                        pragma_runtime = True
                    else:
                        evidence.add("other-jsx")
        if pragma_runtime and not pragma_factory:
            evidence.add("react")
        if evidence:
            return "react" if evidence == {"react"} else "generic"
        if jsx:
            local = self._local_evidence(path)
            return "react" if local == {"react"} else "generic"
        return "generic"
