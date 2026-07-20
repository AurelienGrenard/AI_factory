from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from tools.notebooks.audit_timings import audit as audit_timings
from tools.validation.audit import coherence_frame, load_production_audit, timing_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_RESULTS = PROJECT_ROOT / "registry" / "production" / "results"


def _audit_cases() -> list[tuple[str, str, bool]]:
    cases = []
    for model_directory in sorted(PRODUCTION_RESULTS.iterdir()):
        if not model_directory.is_dir():
            continue
        for product_directory in sorted(model_directory.iterdir()):
            data_directory = product_directory / "data"
            if not data_directory.is_dir():
                continue
            modes = {
                "delta_crn" in path.stem for path in data_directory.glob("*.json")
            }
            cases.extend(
                (model_directory.name, product_directory.name, mode)
                for mode in sorted(modes)
            )
    return cases


@pytest.mark.parametrize("model,product,delta_crn", _audit_cases())
def test_all_production_results_have_certified_validation_slices(
    model: str,
    product: str,
    delta_crn: bool,
) -> None:
    audit = load_production_audit(
        PROJECT_ROOT,
        model_family=model,
        product_family=product,
        delta_crn=delta_crn,
    )
    frame = coherence_frame(audit)
    performance = timing_frame(audit)
    assert len(audit.production.data["results"]) >= 100
    assert all(len(document.data["results"]) == 100 for document in audit.validation.values())
    assert all(math.isfinite(float(value)) for value in frame["max abs error"])
    assert list(performance.index) == [
        "cpp cuda", "pytorch gpu", "cpp cpu", "pytorch cpu"
    ]

    for key, document in audit.validation.items():
        timing = document.specification["timing"]
        assert timing["warmup_calls"] >= 1
        assert timing["benchmark_repetitions"] >= 1
        assert timing["benchmark_statistic"] == "median"
        assert timing["benchmark_row_count"] >= 100
        assert isinstance(timing["benchmark_workload"], str)
        assert timing["benchmark_workload"]
        assert timing["benchmark_seconds"] > 0.0
        if key == "cpp_gpu":
            assert timing["kernel_seconds"] > 0.0
            assert timing["benchmark_kernel_seconds"] > 0.0
            assert timing["wall_seconds"] >= timing["kernel_seconds"]
            assert timing["benchmark_seconds"] >= timing["benchmark_kernel_seconds"]
        else:
            assert "kernel_seconds" not in timing
            assert "benchmark_kernel_seconds" not in timing

    for label, row in frame.iterrows():
        maximum_error = float(row["max abs error"])
        if label.startswith("production stored cpp cuda"):
            assert maximum_error <= 1.0e-12, (model, product, label, maximum_error)
        if label == "cpp cpu vs cpp cuda":
            if product == "american_puts":
                tolerance = 1.0e-4
            elif model == "rough_heston" and product in {
                "lookback_fixed_calls", "volatility_swaps"
            }:
                # Parallel reductions preserve numerical equivalence, not
                # summation order, for these accumulated path statistics.
                tolerance = 1.0e-10
            else:
                tolerance = 1.0e-12
            assert maximum_error <= tolerance, (model, product, label, maximum_error)
        if "price-only vs price-and-gradient" in label:
            assert maximum_error <= 1.0e-12, (model, product, label, maximum_error)
        if label in {"pytorch cpu vs pytorch gpu", "cpp cuda vs pytorch gpu"}:
            z_score = row["max z-score"]
            if z_score is not None and not math.isnan(float(z_score)):
                assert float(z_score) <= 4.0, (model, product, label, z_score)


def test_performance_hierarchy_on_certification_hardware() -> None:
    if os.environ.get("AI_FACTORY_PERFORMANCE_CERTIFICATION") != "1":
        pytest.skip("Performance hierarchy is certified only on the reference GPU host.")
    assert audit_timings(PROJECT_ROOT / "registry/validation/results") == []
