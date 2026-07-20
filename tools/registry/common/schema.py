"""Canonical registry metadata shared by database writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.registry.common.paths import (
    model_family,
    product_family,
    registry_relative_path,
)


ALIGNED_ROW_RULE = "aligned row pairing"

PARAMETER_JSON_KEYS = {
    "curves": (
        "format", "database_id", "curve_family", "specification",
        "generation_script", "row_count", "curves",
    ),
    "models": (
        "format", "database_id", "model_family", "specification",
        "generation_script", "row_count", "models",
    ),
    "products": (
        "format", "database_id", "product_family", "specification",
        "generation_script", "row_count", "products",
    ),
}

PARAMETER_YAML_KEYS = {
    "curves": (
        "title", "format", "database_id", "curve_family", "json_path",
        "generation_script", "parameters", "equations", "construction",
    ),
    "models": (
        "title", "format", "database_id", "model_family", "json_path",
        "generation_script", "parameters", "dynamics", "construction",
    ),
    "products": (
        "title", "format", "database_id", "product_family", "json_path",
        "generation_script", "parameters", "payoff", "construction",
    ),
}

VALIDATION_SOURCE_KEYS = ("source_json_path", "source_yaml_path")

RESULT_JSON_KEYS = (
    "format", "database_id", "status", "specification",
    "generation_script", "row_count", "model_database", "curve_database",
    "product_database", "result_construction", "engine", "timing", "results",
)

RESULT_YAML_KEYS = (
    "title", "format", "database_id", "status", "json_path",
    "generation_script", "summary", "time_grid", "outputs", "monitoring",
    "exercise", "exercise_policy", "delta_method", "model_database",
    "curve_database", "product_database", "result_construction", "timing",
    "source_production_result",
)


def database_reference(
    project_root: Path,
    tier: str,
    kind: str,
    database_id: str,
) -> dict[str, str]:
    """Return the canonical reference to an input database."""
    return {
        "id": database_id,
        "json_path": registry_relative_path(
            project_root, tier, kind, "data", database_id, "json"
        ),
    }


def primary_source_files(
    model_database_id: str,
    product_database_id: str,
    engine: str,
) -> list[str]:
    """Return the specialized implementation unit represented by a result."""
    model = model_family(model_database_id)
    product = product_family(product_database_id)
    if engine.startswith("python_"):
        return [f"src_python/ai_factory/pytorch/{model}/{product}.py"]
    if engine.startswith("cpp_cpu"):
        return [f"src_cpp/ai_factory/cpu/{model}/{product}.cpp"]
    if engine.startswith("cpp_gpu"):
        return [f"src_cpp/ai_factory/cuda/{model}/{product}.cu"]
    raise ValueError(f"Unsupported result engine: {engine}")


def aligned_result_construction(first_seed: int | None) -> dict[str, Any]:
    """Describe deterministic row-aligned result construction."""
    construction: dict[str, Any] = {"rule": ALIGNED_ROW_RULE}
    if first_seed is not None:
        construction["first_seed"] = first_seed
    return construction


def analytic_time_grid() -> dict[str, str]:
    """Keep the result schema uniform when no numerical grid is used."""
    return {"rule": "not applicable for analytic pricing"}


def exact_transition_time_grid(schedule: str) -> dict[str, str]:
    """Document exact-transition simulations without a discretization grid."""
    return {
        "rule": "exact model transition on contractual dates",
        "schedule": schedule,
    }


def canonical_timing(timing: dict[str, Any]) -> dict[str, Any]:
    """Return non-negative timings with host time no shorter than CUDA time."""
    normalized = dict(timing)
    for key, value in normalized.items():
        if key.endswith("seconds") and isinstance(value, (int, float)) and value < 0:
            raise ValueError(f"Timing {key} cannot be negative.")
    if "kernel_seconds" in normalized:
        normalized["wall_seconds"] = max(
            float(normalized["wall_seconds"]),
            float(normalized["kernel_seconds"]),
        )
    if "benchmark_kernel_seconds" in normalized:
        normalized["benchmark_seconds"] = max(
            float(normalized["benchmark_seconds"]),
            float(normalized["benchmark_kernel_seconds"]),
        )
    return normalized
