"""Bound controlled ReviewBench identities to the candidate's shipped packages.

This does not change ordinary extension discovery. Controlled benchmarks require
extensions to ship beside Daydream, so their complete package bytes can be checked
against the reviewer wheel without traversing arbitrary operator directories.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any

_EXTENSION_GUIDANCE = (
    'Controlled ReviewBench requires a packaged daydream_ext beside Daydream; '
    'package the extension in the candidate wheel and unset DAYDREAM_EXT_DIR'
)


def _files(root: Path, prefix: str) -> dict[str, Path]:
    if root.is_symlink():
        raise ValueError('Controlled ReviewBench cannot freeze symlinked candidate package assets')
    paths = [root] if root.is_file() else sorted(root.rglob('*'))
    result = {}
    for path in paths:
        relative = Path(path.name) if root.is_file() else path.relative_to(root)
        if '__pycache__' in relative.parts or path.suffix in {'.pyc', '.pyo'} or path.name == '.DS_Store':
            continue
        if path.is_symlink():
            raise ValueError('Controlled ReviewBench cannot freeze symlinked candidate package assets')
        if path.is_file():
            name = prefix if root.is_file() else f'{prefix}/{relative.as_posix()}'
            result[name] = path
    return result


def _extension_files() -> tuple[str | None, dict[str, Path]]:
    if os.environ.get('DAYDREAM_EXT_DIR'):
        raise ValueError('Controlled ReviewBench does not support DAYDREAM_EXT_DIR. ' + _EXTENSION_GUIDANCE)
    package = Path(__file__).resolve().parents[2]
    directory, module = package.parent / 'daydream_ext', package.parent / 'daydream_ext.py'
    # Establish the owned boundary before discovering or traversing any source.
    # An imported extension elsewhere is not an invitation to scan its directory.
    if directory.is_symlink() or module.is_symlink() or (directory.exists() and module.exists()):
        raise ValueError(_EXTENSION_GUIDANCE + '; ambiguous or symlinked extension source')
    owned = directory / '__init__.py' if directory.is_dir() else module
    try:
        spec = importlib.util.find_spec('daydream_ext')
    except (ImportError, ValueError) as exc:
        raise ValueError(_EXTENSION_GUIDANCE + '; extension discovery is not reproducible') from exc
    if spec is None:
        if directory.exists() or module.exists():
            raise ValueError(_EXTENSION_GUIDANCE + '; owned extension is not importable')
        return None, {}
    if not spec.origin or not owned.is_file() or Path(spec.origin).resolve() != owned.resolve():
        raise ValueError(_EXTENSION_GUIDANCE + '; selected default extension is outside the owned candidate package')
    if directory.is_dir():
        return 'package', _files(directory, 'daydream_ext')
    return 'module', _files(module, 'daydream_ext.py')


def _digest(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def selected_extension_identity() -> dict[str, Any]:
    """Identify the admitted default or explicit absence without executing it."""
    kind, files = _extension_files()
    return {'source_kind': 'packaged' if kind else 'none', 'package_kind': kind,
            'sha256': _digest(files) if kind else None, 'files': len(files)}


def candidate_package_files() -> dict[str, Path]:
    """Return exactly the bounded core/default-extension package inventory."""
    _, extension = _extension_files()
    package = Path(__file__).resolve().parents[2]
    return {**_files(package, 'daydream'), **extension}


def candidate_package_digest() -> str:
    """Hash both shipped package namespaces and all their ordinary resources."""
    return _digest(candidate_package_files())
