"""Resolve historical logical paths to the published catalog or local work.

Recipes and generation receipts retain their original logical paths and hashes.
The manifests route selected catalog leaves and datasets to their physical homes.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=8)
def _routes(root: Path) -> dict[str, str]:
    manifests = (root / "catalog/manifest.json", root / "work/catalog/manifest.json")
    routes = {}
    for manifest in manifests:
        if not manifest.is_file():
            continue
        for row in json.loads(manifest.read_text())["datasets"]:
            source = row["source"]
            for name in ("generator.cpp", "recipe.yaml", "generation.yaml"):
                routes[f"catalog/{source}/{name}"] = f"{row['catalogPath']}/{name}"
            routes[f"catalog/{source}"] = row["catalogPath"]
            routes[row["originalOutput"]] = row["output"]
    return routes


def physical_path(root: Path, relative: str) -> Path:
    """Return the physical location of a logical catalog or dataset path."""
    part = Path(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts:
        raise ValueError(f"Expected a repository-relative path: {relative}")
    base = root.resolve()
    routes = _routes(base)
    route = routes.get(part.as_posix(), part.as_posix())
    if routes and part.as_posix() not in routes and len(part.parts) > 1:
        if part.parts[0] == "catalog" and part.parts[1] in {"model", "product", "curve"}:
            route = str(Path("work/catalog") / Path(*part.parts[1:]))
        elif part.parts[0] == "datasets" and part.parts[1] in {"model", "product", "curve"}:
            route = str(Path("datasets/other") / Path(*part.parts[1:]))
    path = base / route
    if path.resolve() != path:
        raise ValueError(f"Artifact path traverses a symlink: {path}")
    return path
