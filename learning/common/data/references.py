"""Resolve and join model, curve and product rows referenced by price data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping


_ROLE_ORDER = {"model": 0, "curve": 1, "product": 2}


@dataclass(frozen=True)
class EntityRow:
    values: tuple[float, ...]
    ordinal: int


@dataclass(frozen=True)
class EntityTable:
    role: str
    dataset_id: str
    path: Path
    feature_names: tuple[str, ...]
    rows: Mapping[str, EntityRow]


def find_repository_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "catalog").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise ValueError(f"Cannot locate repository root from {path}")


def resolve_catalog_reference(dataset_path: Path, catalog: str) -> Path:
    catalog_path = Path(catalog)
    if catalog_path.is_absolute():
        candidate = catalog_path
    else:
        parts = catalog_path.parts
        if not parts or parts[0] != "catalog":
            raise ValueError(f"Unsupported dataset catalog reference: {catalog}")
        candidate = find_repository_root(dataset_path) / "datasets" / Path(*parts[1:])
    return candidate.with_suffix(".json")


def _flatten_scalars(value: object, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        raise ValueError(f"Boolean feature {prefix!r} needs an explicit encoder")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite feature {prefix!r}")
        return {prefix: number}
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            result.update(_flatten_scalars(child, child_prefix))
        return result
    raise ValueError(
        f"Feature {prefix!r} has type {type(value).__name__}; "
        "register a categorical or sequence encoder before using it"
    )


def _row_array(document: dict, role: str) -> list[dict]:
    expected = {"model": "models", "curve": "curves", "product": "products"}.get(role)
    if expected is not None and isinstance(document.get(expected), list):
        return document[expected]
    candidates = [
        value
        for value in document.values()
        if isinstance(value, list)
        and (not value or isinstance(value[0], dict))
        and (not value or "id" in value[0])
        and (not value or "parameters" in value[0])
    ]
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify parameter rows for role {role!r}")
    return candidates[0]


def load_entity_tables(dataset_path: Path, header: dict) -> tuple[EntityTable, ...]:
    references: list[tuple[str, dict]] = []
    for name, value in header.items():
        if name.endswith("_dataset") and isinstance(value, dict) and "catalog" in value:
            references.append((name.removesuffix("_dataset"), value))
    references.sort(key=lambda item: (_ROLE_ORDER.get(item[0], 100), item[0]))
    if not references:
        raise ValueError("Price dataset does not reference any parameter dataset")

    tables: list[EntityTable] = []
    for role, reference in references:
        path = resolve_catalog_reference(dataset_path, str(reference["catalog"]))
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} parameter dataset: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        dataset_id = str(document.get("database_id", ""))
        if dataset_id != str(reference.get("id", "")):
            raise ValueError(
                f"{role} dataset id mismatch: expected {reference.get('id')!r}, "
                f"found {dataset_id!r}"
            )
        rows = _row_array(document, role)
        if not rows:
            raise ValueError(f"Empty {role} parameter dataset")
        first = _flatten_scalars(rows[0].get("parameters"), role)
        names = tuple(sorted(first))
        mapped: dict[str, EntityRow] = {}
        for ordinal, row in enumerate(rows):
            row_id = str(row.get("id", ""))
            if not row_id or row_id in mapped:
                raise ValueError(f"Missing or duplicate {role} row id {row_id!r}")
            flattened = _flatten_scalars(row.get("parameters"), role)
            if tuple(sorted(flattened)) != names:
                raise ValueError(f"Inconsistent features in {role} row {row_id}")
            mapped[row_id] = EntityRow(
                values=tuple(flattened[name] for name in names), ordinal=ordinal
            )
        tables.append(
            EntityTable(
                role=role,
                dataset_id=dataset_id,
                path=path,
                feature_names=names,
                rows=mapped,
            )
        )
    return tuple(tables)
