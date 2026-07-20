"""Deterministic slices of production registry databases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tools.registry.common.paths import registry_database_path
from tools.registry.common.schema import PARAMETER_YAML_KEYS, VALIDATION_SOURCE_KEYS

DEFAULT_VALIDATION_ROW_COUNT = 100


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_database_slice(
    *,
    project_root: Path,
    kind: str,
    source_id: str,
    target_id: str,
    row_key: str,
    row_count: int = DEFAULT_VALIDATION_ROW_COUNT,
) -> Path:
    """Write the first rows of a production database into validation."""
    if row_count <= 0:
        raise ValueError("row_count must be strictly positive.")

    source_json = registry_database_path(
        project_root, "production", kind, "data", source_id, "json"
    )
    source_yaml = registry_database_path(
        project_root, "production", kind, "specifications", source_id, "yaml"
    )
    target_json = registry_database_path(
        project_root, "validation", kind, "data", target_id, "json"
    )
    target_yaml = registry_database_path(
        project_root, "validation", kind, "specifications", target_id, "yaml"
    )
    generator = registry_database_path(
        project_root, "validation", kind, "generators", target_id, "py"
    )

    source = json.loads(source_json.read_text(encoding="utf-8"))
    source_rows = source.get(row_key)
    if not isinstance(source_rows, list):
        raise ValueError(f"{source_json} has no list field {row_key!r}.")
    rows = source_rows[:row_count]
    if len(rows) < row_count:
        raise ValueError(
            f"Requested {row_count} rows from {source_id}, which only has "
            f"{len(source_rows)}."
        )

    target = {
        **{key: value for key, value in source.items() if key != row_key},
        "database_id": target_id,
        "specification": _relative(project_root, target_yaml),
        "generation_script": _relative(project_root, generator),
        "row_count": len(rows),
        row_key: rows,
    }
    _write_json(target_json, target)

    source_specification = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    inherited_specification: dict[str, Any] = {
        key: value
        for key, value in source_specification.items()
        if key not in {
            "title",
            "database_id",
            "json_path",
            "generation_script",
            "construction",
        }
    }
    values: dict[str, Any] = {
        "title": f"Production slice {target_id}",
        "format": source["format"],
        "database_id": target_id,
        **inherited_specification,
    }
    family_key = {
        "models": "model_family",
        "products": "product_family",
        "curves": "curve_family",
    }.get(kind)
    if family_key is not None and family_key in source:
        values[family_key] = source[family_key]
    values.update(
        {
            "json_path": _relative(project_root, target_json),
            "generation_script": _relative(project_root, generator),
            "source_json_path": _relative(project_root, source_json),
            "source_yaml_path": _relative(project_root, source_yaml),
            "construction": {
                "row_count": len(rows),
                "method": "production database slice",
                "rule": "first rows",
                "source_database_id": source_id,
            },
        }
    )
    canonical_keys = PARAMETER_YAML_KEYS[kind] + VALIDATION_SOURCE_KEYS
    specification = {key: values[key] for key in canonical_keys if key in values}
    _write_yaml(target_yaml, specification)
    return target_json
