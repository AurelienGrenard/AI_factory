"""Load and compare one production result with its four-engine audit slice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


MIN_RELATIVE_PRICE = 1.0e-4
ENGINE_ORDER = (
    ("cpp_gpu", "cpp cuda"),
    ("python_gpu", "pytorch gpu"),
    ("cpp_cpu", "cpp cpu"),
    ("python_cpu", "pytorch cpu"),
)


@dataclass(frozen=True)
class ResultDocument:
    data: dict[str, Any]
    specification: dict[str, Any]
    json_path: Path
    yaml_path: Path


@dataclass(frozen=True)
class ProductionAudit:
    model_family: str
    product_family: str
    delta_crn: bool
    production: ResultDocument
    validation: dict[str, ResultDocument]
    price_only_validation: dict[str, ResultDocument]


def _read_document(json_path: Path) -> ResultDocument:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    yaml_path = json_path.parent.parent / "specifications" / f"{json_path.stem}.yaml"
    specification = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return ResultDocument(data, specification, json_path, yaml_path)


def _engine_key(engine: str) -> str | None:
    return next((key for key, _ in ENGINE_ORDER if key in engine), None)


def _matches_mode(document: ResultDocument, delta_crn: bool) -> bool:
    return ("delta_crn" in document.data["engine"]) == delta_crn


def _one(documents: list[ResultDocument], label: str) -> ResultDocument:
    if len(documents) != 1:
        raise ValueError(f"Expected one {label}, found {len(documents)}.")
    return documents[0]


def load_production_audit(
    project_root: Path,
    *,
    model_family: str,
    product_family: str,
    delta_crn: bool = False,
) -> ProductionAudit:
    """Discover the canonical production result and four validation engines."""

    production_root = (
        project_root / "registry" / "production" / "results"
        / model_family / product_family / "data"
    )
    production = _one(
        [
            document
            for path in production_root.glob("*.json")
            if (document := _read_document(path)).data["engine"].startswith("cpp_gpu")
            and _matches_mode(document, delta_crn)
        ],
        f"production {model_family}/{product_family} result",
    )

    model_id = production.data["model_database"]["id"] + "__first_100"
    product_id = production.data["product_database"]["id"] + "__first_100"
    curve_reference = production.data.get("curve_database")
    curve_id = curve_reference["id"] + "__first_100" if curve_reference else None
    validation_root = (
        project_root / "registry" / "validation" / "results"
        / model_family / product_family / "data"
    )

    def validation_for_mode(mode: bool) -> dict[str, ResultDocument]:
        grouped: dict[str, list[ResultDocument]] = {
            key: [] for key, _ in ENGINE_ORDER
        }
        for path in validation_root.glob("*.json"):
            document = _read_document(path)
            data = document.data
            if not _matches_mode(document, mode):
                continue
            if data["model_database"]["id"] != model_id:
                continue
            if data["product_database"]["id"] != product_id:
                continue
            if curve_id is not None and data.get("curve_database", {}).get("id") != curve_id:
                continue
            key = _engine_key(data["engine"])
            if key is not None:
                grouped[key].append(document)
        return {
            key: _one(documents, f"{key} validation result")
            for key, documents in grouped.items()
        }

    validation = validation_for_mode(delta_crn)
    price_only = validation_for_mode(False) if delta_crn else {}
    return ProductionAudit(
        model_family,
        product_family,
        delta_crn,
        production,
        validation,
        price_only,
    )


def timing_frame(audit: ProductionAudit) -> pd.DataFrame:
    """Return the fixed four-engine timing view used by every notebook."""

    rows = []
    workloads = set()
    for key, label in ENGINE_ORDER:
        timing = audit.validation[key].specification["timing"]
        workloads.add((
            timing.get("benchmark_row_count"),
            timing.get("benchmark_repetitions"),
            timing.get("benchmark_workload"),
        ))
        rows.append({
            "engine": label,
            "wall seconds": timing.get("benchmark_seconds", timing["wall_seconds"]),
            "kernel seconds": (
                timing.get("benchmark_kernel_seconds", timing.get("kernel_seconds"))
                if key == "cpp_gpu" else None
            ),
        })
    if len(workloads) != 1:
        raise ValueError("Timing engines use different benchmark workloads.")
    frame = pd.DataFrame(rows).set_index("engine")
    frame.attrs["benchmark_row_count"] = next(iter(workloads))[0]
    return frame


def _standard_error_key(output: str) -> str:
    return "standard_error" if output == "price" else f"{output}_standard_error"


def comparison_metrics(
    label: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    output: str,
) -> dict[str, Any]:
    """Return the compact deterministic/statistical comparison metrics."""

    if len(left) != len(right):
        raise ValueError(f"{label} compares different row counts.")
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    z_scores: list[float] = []
    se_key = _standard_error_key(output)
    for left_row, right_row in zip(left, right, strict=True):
        if left_row["id"] != right_row["id"]:
            raise ValueError(f"{label} compares misaligned result ids.")
        left_outputs = left_row["outputs"]
        right_outputs = right_row["outputs"]
        left_value = float(left_outputs[output])
        right_value = float(right_outputs[output])
        error = abs(left_value - right_value)
        absolute_errors.append(error)
        scale = max(abs(left_value), abs(right_value))
        if scale >= MIN_RELATIVE_PRICE:
            relative_errors.append(100.0 * error / scale)
        if se_key in left_outputs and se_key in right_outputs:
            standard_error = math.hypot(
                float(left_outputs[se_key]), float(right_outputs[se_key])
            )
            if standard_error > 0.0:
                z_scores.append(error / standard_error)
            elif error > 1.0e-12:
                z_scores.append(float("inf"))
    return {
        "check": label,
        "output": output,
        "max abs error": max(absolute_errors, default=0.0),
        "max rel error (%)": max(relative_errors) if relative_errors else None,
        "relative rows": len(relative_errors),
        "max z-score": max(z_scores) if z_scores else None,
    }


def coherence_frame(audit: ProductionAudit) -> pd.DataFrame:
    """Build the canonical production/native/statistical coherence table."""

    production_rows = audit.production.data["results"][:100]
    validation = {
        key: document.data["results"] for key, document in audit.validation.items()
    }
    outputs = ["price"]
    if audit.delta_crn:
        outputs.extend(
            key for key in audit.production.specification["outputs"]
            if key not in {"price", "standard_error"} and not key.endswith("standard_error")
        )
    rows = []
    for output in outputs:
        rows.extend((
            comparison_metrics(
                "production stored cpp cuda vs regenerated cpp cuda",
                production_rows, validation["cpp_gpu"], output=output,
            ),
            comparison_metrics(
                "cpp cpu vs cpp cuda",
                validation["cpp_cpu"], validation["cpp_gpu"], output=output,
            ),
            comparison_metrics(
                "pytorch cpu vs pytorch gpu",
                validation["python_cpu"], validation["python_gpu"], output=output,
            ),
            comparison_metrics(
                "cpp cuda vs pytorch gpu",
                validation["cpp_gpu"], validation["python_gpu"], output=output,
            ),
        ))
    if audit.delta_crn:
        for key, label in (("cpp_gpu", "cpp cuda"), ("cpp_cpu", "cpp cpu")):
            rows.append(comparison_metrics(
                f"{label} price-only vs price-and-gradient",
                audit.price_only_validation[key].data["results"],
                validation[key],
                output="price",
            ))
    return pd.DataFrame(rows).set_index("check")
