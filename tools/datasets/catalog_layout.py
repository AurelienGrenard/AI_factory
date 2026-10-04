"""Resolve historical recipe paths into the prod/other physical layout.

Recipes and generation receipts keep their original logical paths and hashes.
The manifests are the explicit routing table for both catalog and datasets.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=8)
def _routes(root: Path) -> dict[str, str]:
    manifests = [root / 'catalog' / branch / 'manifest.json' for branch in ('prod', 'other')]
    manifests = [path for path in manifests if path.is_file()]
    if not manifests:
        return {}
    routes = {}
    for manifest in manifests:
        branch = manifest.parent.name
        for row in json.loads(manifest.read_text())['datasets']:
            source = row['source']
            for name in ('generator.cpp', 'recipe.yaml', 'generation.yaml'):
                routes[f'catalog/{source}/{name}'] = f'catalog/{branch}/{source}/{name}'
            routes[f'catalog/{source}'] = f'catalog/{branch}/{source}'
            routes[row['originalOutput']] = row['output']
    return routes


def physical_path(root: Path, relative: str) -> Path:
    """Return the physical location of a logical catalog or dataset path."""
    part = Path(relative)
    if part.is_absolute() or not part.parts or '..' in part.parts:
        raise ValueError(f'Expected a repository-relative path: {relative}')
    base = root.resolve()
    routes = _routes(base)
    route = routes.get(part.as_posix(), part.as_posix())
    if routes and route == part.as_posix() and len(part.parts) > 1:
        if part.parts[0] in {'catalog', 'datasets'} and part.parts[1] in {'model', 'product', 'curve'}:
            route = str(Path(part.parts[0]) / 'other' / Path(*part.parts[1:]))
    path = base / route
    if path.resolve() != path:
        raise ValueError(f'Artifact path traverses a symlink: {path}')
    return path
