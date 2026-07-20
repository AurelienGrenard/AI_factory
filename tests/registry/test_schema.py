from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from tools.registry.common.schema import (
    PARAMETER_JSON_KEYS,
    PARAMETER_YAML_KEYS,
    RESULT_JSON_KEYS,
    RESULT_YAML_KEYS,
    VALIDATION_SOURCE_KEYS,
    primary_source_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = PROJECT_ROOT / "registry"

PARAMETER_CONTRACTS = {
    "curves": {
        "format": "ai_factory.curves.v1",
        "family": "curve_family",
        "rows": "curves",
        "semantic": {"parameters", "equations"},
    },
    "models": {
        "format": "ai_factory.models.v1",
        "family": "model_family",
        "rows": "models",
        "semantic": {"parameters", "dynamics"},
    },
    "products": {
        "format": "ai_factory.products.v1",
        "family": "product_family",
        "rows": "products",
        "semantic": {"parameters", "payoff"},
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _assert_relative_file(value: Any, owner: Path) -> Path:
    assert isinstance(value, str) and value, owner
    path = PROJECT_ROOT / value
    assert path.is_file(), (owner, value)
    return path


def _assert_required(mapping: dict[str, Any], keys: set[str], owner: Path) -> None:
    missing = keys - mapping.keys()
    assert missing == set(), (owner, sorted(missing))


def _assert_exact_top_level(
    mapping: dict[str, Any],
    canonical_keys: tuple[str, ...],
    owner: Path,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    allowed = set(canonical_keys)
    required = allowed - optional
    assert required <= mapping.keys(), (owner, sorted(required - mapping.keys()))
    assert mapping.keys() <= allowed, (owner, sorted(mapping.keys() - allowed))
    expected_order = [key for key in canonical_keys if key in mapping]
    assert list(mapping) == expected_order, (owner, list(mapping), expected_order)


def _assert_rows(rows: Any, owner: Path) -> None:
    assert isinstance(rows, list), owner
    identifiers: list[str] = []
    for row in rows:
        _assert_required(row, {"id", "parameters"}, owner)
        assert isinstance(row["id"], str) and row["id"], owner
        assert isinstance(row["parameters"], dict) and row["parameters"], owner
        identifiers.append(row["id"])
    assert len(identifiers) == len(set(identifiers)), owner


def test_parameter_database_contracts() -> None:
    for tier in ("production", "validation"):
        for kind, contract in PARAMETER_CONTRACTS.items():
            for json_path in sorted((REGISTRY_ROOT / tier / kind).rglob("data/*.json")):
                yaml_path = (
                    json_path.parent.parent
                    / "specifications"
                    / f"{json_path.stem}.yaml"
                )
                data = _read_json(json_path)
                specification = _read_yaml(yaml_path)
                _assert_exact_top_level(
                    data, PARAMETER_JSON_KEYS[kind], json_path
                )
                specification_keys = PARAMETER_YAML_KEYS[kind]
                if tier == "validation":
                    specification_keys += VALIDATION_SOURCE_KEYS
                _assert_exact_top_level(
                    specification, specification_keys, yaml_path
                )
                family_key = contract["family"]
                row_key = contract["rows"]
                _assert_required(
                    data,
                    {
                        "format", "database_id", family_key, "specification",
                        "generation_script", "row_count", row_key,
                    },
                    json_path,
                )
                _assert_required(
                    specification,
                    {
                        "title", "format", "database_id", family_key,
                        "json_path", "generation_script", "construction",
                        *contract["semantic"],
                    },
                    yaml_path,
                )
                assert data["format"] == specification["format"] == contract["format"]
                assert data["database_id"] == specification["database_id"] == json_path.stem
                assert data[family_key] == specification[family_key]
                assert data["specification"] == _relative(yaml_path)
                assert specification["json_path"] == _relative(json_path)
                assert data["generation_script"] == specification["generation_script"]
                _assert_relative_file(data["generation_script"], json_path)
                _assert_rows(data[row_key], json_path)
                assert data["row_count"] == len(data[row_key])
                assert specification["construction"]["row_count"] == data["row_count"]
                for key in contract["semantic"]:
                    assert isinstance(specification[key], dict) and specification[key]


def _assert_database_reference(
    reference: Any,
    kind: str,
    owner: Path,
) -> dict[str, Any]:
    assert isinstance(reference, dict), owner
    assert set(reference) == {"id", "json_path"}, owner
    referenced_path = _assert_relative_file(reference["json_path"], owner)
    referenced = _read_json(referenced_path)
    assert referenced_path.parts[-4] == kind, (owner, reference)
    assert reference["id"] == referenced["database_id"], owner
    return referenced


def _assert_timing(timing: Any, owner: Path) -> None:
    assert isinstance(timing, dict), owner
    assert isinstance(timing.get("wall_seconds"), (int, float)), owner
    assert math.isfinite(float(timing["wall_seconds"]))
    assert timing["wall_seconds"] >= 0
    for key, value in timing.items():
        if key.endswith("seconds") and isinstance(value, (int, float)):
            assert math.isfinite(float(value)) and value >= 0, (owner, key)
    if "kernel_seconds" in timing:
        assert timing["wall_seconds"] >= timing["kernel_seconds"], owner
    if "benchmark_kernel_seconds" in timing:
        assert timing["benchmark_seconds"] >= timing["benchmark_kernel_seconds"], owner


def test_result_database_contracts() -> None:
    for tier in ("production", "validation"):
        root = REGISTRY_ROOT / tier / "results"
        for json_path in sorted(root.rglob("data/*.json")):
            yaml_path = (
                json_path.parent.parent
                / "specifications"
                / f"{json_path.stem}.yaml"
            )
            data = _read_json(json_path)
            specification = _read_yaml(yaml_path)
            _assert_exact_top_level(
                data,
                RESULT_JSON_KEYS,
                json_path,
                optional={"curve_database", "timing"},
            )
            _assert_exact_top_level(
                specification,
                RESULT_YAML_KEYS,
                yaml_path,
                optional={
                    "curve_database", "monitoring", "exercise",
                    "exercise_policy", "delta_method",
                    "source_production_result",
                },
            )
            _assert_required(
                data,
                {
                    "format", "database_id", "specification", "generation_script",
                    "row_count", "model_database", "product_database",
                    "result_construction", "engine", "results",
                },
                json_path,
            )
            _assert_required(
                specification,
                {
                    "title", "format", "database_id", "json_path",
                    "generation_script", "summary", "time_grid", "outputs",
                    "model_database", "product_database", "result_construction",
                    "timing",
                },
                yaml_path,
            )
            assert not {
                "model_database_id", "curve_database_id", "product_database_id"
            } & data.keys(), json_path
            assert data["format"] == specification["format"] == "ai_factory.results.v1"
            assert data["database_id"] == specification["database_id"] == json_path.stem
            assert data["specification"] == _relative(yaml_path)
            assert specification["json_path"] == _relative(json_path)
            assert data["generation_script"] == specification["generation_script"]
            _assert_relative_file(data["generation_script"], json_path)
            assert data["model_database"] == specification["model_database"]
            assert data["product_database"] == specification["product_database"]
            models = _assert_database_reference(data["model_database"], "models", json_path)
            products = _assert_database_reference(data["product_database"], "products", json_path)
            curves = None
            if "curve_database" in data or "curve_database" in specification:
                assert data.get("curve_database") == specification.get("curve_database")
                curves = _assert_database_reference(data["curve_database"], "curves", json_path)
            assert data["result_construction"] == specification["result_construction"]
            assert isinstance(data["result_construction"].get("rule"), str)
            assert data["result_construction"]["rule"]
            assert isinstance(specification["time_grid"], dict) and specification["time_grid"]
            assert isinstance(specification["outputs"], dict) and specification["outputs"]
            summary = specification["summary"]
            _assert_required(summary, {"row_count", "source_files"}, yaml_path)
            assert isinstance(summary["source_files"], list) and summary["source_files"]
            assert summary["source_files"] == primary_source_files(
                data["model_database"]["id"],
                data["product_database"]["id"],
                data["engine"],
            ), yaml_path
            for source_file in summary["source_files"]:
                _assert_relative_file(source_file, yaml_path)
            _assert_timing(specification["timing"], yaml_path)

            results = data["results"]
            assert isinstance(results, list)
            assert data["row_count"] == summary["row_count"] == len(results)
            model_ids = {row["id"] for row in models["models"]}
            product_ids = {row["id"] for row in products["products"]}
            curve_ids = {row["id"] for row in curves["curves"]} if curves else set()
            result_ids: list[str] = []
            output_keys = set(specification["outputs"])
            stochastic = "analytic" not in data["engine"]
            for row in results:
                _assert_required(row, {"id", "model_id", "product_id", "outputs"}, json_path)
                result_ids.append(row["id"])
                assert row["model_id"] in model_ids, (json_path, row["model_id"])
                assert row["product_id"] in product_ids, (json_path, row["product_id"])
                if curves is not None:
                    assert row.get("curve_id") in curve_ids, (json_path, row.get("curve_id"))
                else:
                    assert "curve_id" not in row, json_path
                if stochastic:
                    assert isinstance(row.get("seed"), int), (json_path, row["id"])
                else:
                    assert "seed" not in row, (json_path, row["id"])
                assert set(row["outputs"]) == output_keys, (json_path, row["id"])
                for value in row["outputs"].values():
                    assert isinstance(value, (int, float)) and math.isfinite(value), (json_path, row["id"])
                if "standard_error" in row["outputs"]:
                    assert row["outputs"]["standard_error"] >= 0
            assert len(result_ids) == len(set(result_ids)), json_path


def test_stochastic_volatility_american_policies_use_full_markov_state() -> None:
    expected_states = {
        "heston": ["spot", "instantaneous_variance"],
        "rough_heston": ["spot", "markovian_factors_Y1_to_Y8"],
    }
    for tier in ("production", "validation"):
        for model, expected in expected_states.items():
            root = REGISTRY_ROOT / tier / "results" / model / "american_puts"
            for yaml_path in sorted(root.glob("specifications/*.yaml")):
                specification = _read_yaml(yaml_path)
                basis = specification["exercise_policy"]["basis"]
                assert basis["state"] == expected, yaml_path
                assert basis["functions"], yaml_path
                assert "ridge" in basis["regularization"], yaml_path
