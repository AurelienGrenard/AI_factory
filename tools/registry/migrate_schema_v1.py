"""Normalize existing registry metadata to the canonical v1 contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.schema import (
    ALIGNED_ROW_RULE,
    analytic_time_grid,
    canonical_timing,
    database_reference,
    exact_transition_time_grid,
    primary_source_files,
)


REGISTRY_ROOT = PROJECT_ROOT / "registry"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _specification_path(data_path: Path) -> Path:
    return data_path.parent.parent / "specifications" / f"{data_path.stem}.yaml"


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _normalize_parameter_database(
    tier: str,
    kind: str,
    data_path: Path,
) -> None:
    data = _read_json(data_path)
    spec_path = _specification_path(data_path)
    spec = _read_yaml(spec_path)
    source_spec: dict[str, Any] = {}
    if "source_yaml_path" in spec:
        source_spec = _read_yaml(PROJECT_ROOT / spec["source_yaml_path"])

    family_key, row_key, semantic_keys = {
        "curves": ("curve_family", "curves", ("parameters", "equations")),
        "models": ("model_family", "models", ("parameters", "dynamics")),
        "products": ("product_family", "products", ("parameters", "payoff")),
    }[kind]
    rows = data[row_key]
    construction = spec.get("construction") or source_spec.get("construction", {})
    construction = {**construction, "row_count": len(rows)}
    if tier == "validation":
        source_id = spec.get("source_database_id")
        if source_id is None and "source_json_path" in spec:
            source_id = Path(spec["source_json_path"]).stem
        construction = {
            "row_count": len(rows),
            "method": "production database slice",
            "rule": "first rows",
            "source_database_id": source_id,
        }

    canonical: dict[str, Any] = {
        "title": spec.get("title", source_spec.get("title", data["database_id"])),
        "format": data["format"],
        "database_id": data["database_id"],
        family_key: data[family_key],
        "json_path": _relative(data_path),
        "generation_script": data["generation_script"],
    }
    for key in semantic_keys:
        value = spec.get(key, source_spec.get(key))
        if value is not None:
            canonical[key] = value
    canonical["construction"] = construction
    for key in ("source_json_path", "source_yaml_path"):
        if key in spec:
            canonical[key] = spec[key]

    data["row_count"] = len(rows)
    data["specification"] = _relative(spec_path)
    _write_json(data_path, data)
    _write_yaml(spec_path, canonical)


def _reference_id(
    data: dict[str, Any],
    spec: dict[str, Any],
    key: str,
) -> str | None:
    reference = data.get(key) or spec.get(key)
    if isinstance(reference, dict):
        return reference.get("id")
    legacy = data.get(f"{key}_id")
    return str(legacy) if legacy is not None else None


def _construction(
    data: dict[str, Any], spec: dict[str, Any], *, stochastic: bool
) -> dict[str, Any]:
    source = spec.get("result_construction") or data.get("result_construction") or {}
    rule = str(source.get("rule", ALIGNED_ROW_RULE))
    if "aligned" in rule.lower():
        rule = ALIGNED_ROW_RULE
    construction: dict[str, Any] = {"rule": rule}
    if stochastic:
        first_seed = source.get("first_seed")
        if first_seed is None:
            first_row = next(iter(data.get("results", [])), {})
            first_seed = first_row.get("seed")
        if first_seed is not None:
            construction["first_seed"] = int(first_seed)
    for key, value in source.items():
        if key not in {"rule", "first_seed"}:
            construction[key] = value
    return construction


def _time_grid(
    spec: dict[str, Any], model: str, product: str, engine: str
) -> dict[str, Any]:
    if "time_grid" in spec:
        return spec["time_grid"]
    if "analytic" in engine:
        return analytic_time_grid()
    if model == "hull_white":
        schedule = "exercise dates" if "swaption" in product else "contractual dates"
        return exact_transition_time_grid(schedule)
    return {
        "rule": "nearest integer step count to target dt",
        "target_dt": "1/52",
        "step_count": "round(maturity / target_dt)",
        "effective_dt": "maturity / step_count",
    }


def _normalize_result(tier: str, data_path: Path) -> None:
    data = _read_json(data_path)
    spec_path = _specification_path(data_path)
    spec = _read_yaml(spec_path)
    model = data_path.parents[2].name
    product = data_path.parents[1].name
    engine = str(data.get("engine", spec.get("summary", {}).get("engine", "")))
    model_id = _reference_id(data, spec, "model_database")
    product_id = _reference_id(data, spec, "product_database")
    curve_id = _reference_id(data, spec, "curve_database")
    if model_id is None or product_id is None:
        raise ValueError(f"Missing result database references in {data_path}")

    model_database = database_reference(PROJECT_ROOT, tier, "models", model_id)
    product_database = database_reference(PROJECT_ROOT, tier, "products", product_id)
    curve_database = (
        database_reference(PROJECT_ROOT, tier, "curves", curve_id)
        if curve_id is not None
        else None
    )
    stochastic = "analytic" not in engine
    construction = _construction(data, spec, stochastic=stochastic)
    results = data["results"]
    if not stochastic:
        results = [
            {key: value for key, value in row.items() if key != "seed"}
            for row in results
        ]
    timing = spec.get("timing", data.get("timing"))
    if not isinstance(timing, dict):
        raise ValueError(f"Missing result timing in {data_path}")
    timing = canonical_timing(timing)

    normalized_data: dict[str, Any] = {
        "format": "ai_factory.results.v1",
        "database_id": data["database_id"],
    }
    if "status" in data:
        normalized_data["status"] = data["status"]
    normalized_data.update(
        {
            "specification": _relative(spec_path),
            "generation_script": data["generation_script"],
            "row_count": len(results),
            "model_database": model_database,
        }
    )
    if curve_database is not None:
        normalized_data["curve_database"] = curve_database
    normalized_data.update(
        {
            "product_database": product_database,
            "result_construction": construction,
            "engine": engine,
        }
    )
    if "timing" in data:
        normalized_data["timing"] = timing
    normalized_data["results"] = results

    summary = dict(spec.get("summary", {}))
    summary["row_count"] = len(results)
    summary.setdefault("model", model.replace("_", " ").title())
    summary.setdefault("payoff", product.replace("_", " ").title())
    summary["engine"] = engine
    summary.setdefault("device", "gpu" if "_gpu_" in engine else "cpu")
    summary["source_files"] = primary_source_files(model_id, product_id, engine)

    reserved = {
        "title", "format", "database_id", "status", "json_path",
        "generation_script", "summary", "time_grid", "outputs",
        "model_database", "curve_database", "product_database",
        "result_construction", "timing",
    }
    extras = {key: value for key, value in spec.items() if key not in reserved}
    normalized_spec: dict[str, Any] = {
        "title": spec.get("title", data["database_id"]),
        "format": "ai_factory.results.v1",
        "database_id": data["database_id"],
    }
    if "status" in spec:
        normalized_spec["status"] = spec["status"]
    normalized_spec.update(
        {
            "json_path": _relative(data_path),
            "generation_script": data["generation_script"],
            "summary": summary,
            "time_grid": _time_grid(spec, model, product, engine),
            "outputs": spec["outputs"],
        }
    )
    normalized_spec.update(extras)
    normalized_spec["model_database"] = model_database
    if curve_database is not None:
        normalized_spec["curve_database"] = curve_database
    normalized_spec.update(
        {
            "product_database": product_database,
            "result_construction": construction,
            "timing": timing,
        }
    )
    _write_json(data_path, normalized_data)
    _write_yaml(spec_path, normalized_spec)


def main() -> None:
    counts = {"parameters": 0, "results": 0}
    for tier in ("production", "validation"):
        for kind in ("curves", "models", "products"):
            root = REGISTRY_ROOT / tier / kind
            for data_path in sorted(root.rglob("data/*.json")):
                _normalize_parameter_database(tier, kind, data_path)
                counts["parameters"] += 1
        for data_path in sorted(
            (REGISTRY_ROOT / tier / "results").rglob("data/*.json")
        ):
            _normalize_result(tier, data_path)
            counts["results"] += 1
    print(
        f"Normalized {counts['parameters']} parameter databases and "
        f"{counts['results']} result databases."
    )


if __name__ == "__main__":
    main()
