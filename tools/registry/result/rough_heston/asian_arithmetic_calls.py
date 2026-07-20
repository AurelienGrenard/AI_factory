"""Shared Rough Heston asian arithmetic priced result recipes."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PYTHON = PROJECT_ROOT / "src_python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from ai_factory.pytorch.rough_heston.asian_arithmetic_calls import (
    price_batch as pytorch_asian_price_batch,
    price_delta_crn_batch as pytorch_asian_delta_crn_batch,
)
from tools.registry.result.common.execution import (
    rows_grouped_by_step_count as _rows_grouped_by_step_count,
)
from tools.registry.result.common.metadata import (
    DEFAULT_RELATIVE_BUMP,
    DEFAULT_TARGET_DT,
    delta_crn_outputs_documentation,
    delta_method_documentation,
    price_only_outputs_documentation,
    time_grid_documentation,
)
from tools.registry.result.rough_heston.common import (
    cpp_outputs_for_time_grid as _cpp_native_outputs,
)
DEFAULT_NUM_PATHS = 16_384
DEFAULT_FIRST_SEED = 941_700_001
DEFAULT_PYTORCH_BATCH_ROWS = 16


class CRoughHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double),
        ("hurst", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


class CMonteCarloOutput(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("standard_error", ctypes.c_double),
    ]


class CPriceDeltaOutput(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("standard_error", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("delta_standard_error", ctypes.c_double),
    ]


def references_for_engine(engine: str) -> list[dict[str, Any]]:
    references = [
        {
            "topic": "Markovian approximation of rough Heston",
            "reference": {
                "authors": "Abi Jaber, E.; El Euch, O.",
                "year": 2019,
                "title": "Multifactor Approximation of Rough Volatility Models",
            },
        }
    ]
    if engine.startswith("cpp_"):
        references.append(
            {
                "topic": "C++ Philox counter-based random numbers",
                "reference": {
                    "authors": (
                        "Salmon, J. K.; Moraes, M. A.; Dror, R. O.; Shaw, D. E."
                    ),
                    "year": 2011,
                    "title": "Parallel Random Numbers: As Easy as 1, 2, 3",
                },
            }
        )
    if "delta_crn" in engine:
        references.append(
            {
                "topic": "Common random numbers finite-difference delta",
                "reference": {
                    "authors": "Glasserman, P.; Yao, D. D.",
                    "year": 1992,
                    "title": "Some Guidelines and Guarantees for Common Random Numbers",
                },
            }
        )
    return references


def source_files_for_engine(engine: str) -> list[str]:
    common = ["tools/registry/result/rough_heston/asian_arithmetic_calls.py"]
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/rough_heston/common.py",
            "src_python/ai_factory/pytorch/rough_heston/pathwise.py",
            "src_python/ai_factory/pytorch/rough_heston/asian_arithmetic_calls.py",
            *common,
        ]
    if engine in {"cpp_cpu_philox", "cpp_cpu_philox_delta_crn"}:
        return [
            "src_cpp/ai_factory/cpu/common/philox.cpp",
            "src_cpp/ai_factory/cpu/common/payoffs/asian.hpp",
            "src_cpp/ai_factory/cpu/rough_heston/common.cpp",
            "src_cpp/ai_factory/cpu/rough_heston/asian_arithmetic_calls.cpp",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    if engine in {"cpp_gpu_philox", "cpp_gpu_philox_delta_crn"}:
        return [
            "src_cpp/ai_factory/cuda/common/philox.cuh",
            "src_cpp/ai_factory/cuda/common/reductions.cuh",
            "src_cpp/ai_factory/cuda/common/runtime.cuh",
            "src_cpp/ai_factory/cuda/common/types.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/api.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/dynamics.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/asian_arithmetic_calls.cu",
            "src_cpp/ai_factory/cuda/rough_heston/paths.cu",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    raise ValueError(f"Unsupported engine: {engine}")


def python_outputs(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    device: str,
    batch_rows: int = DEFAULT_PYTORCH_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    outputs: list[dict[str, float] | None] = [None] * len(rows)
    timing: dict[str, float] = {"wall_seconds": 0.0}
    started = perf_counter()
    for num_steps, indexed_rows in _rows_grouped_by_step_count(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed_rows]
        group_outputs, group_timing = pytorch_asian_price_batch(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=device,
            batch_rows=batch_rows,
        )
        for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
            outputs[index] = output
        for name, value in group_timing.items():
            if name != "wall_seconds":
                timing[name] = timing.get(name, 0.0) + float(value)
    timing["wall_seconds"] = perf_counter() - started
    return [output for output in outputs if output is not None], timing

def python_delta_crn_outputs(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    device: str,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
    batch_rows: int = DEFAULT_PYTORCH_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    outputs: list[dict[str, float] | None] = [None] * len(rows)
    timing: dict[str, float] = {"wall_seconds": 0.0}
    started = perf_counter()
    for num_steps, indexed_rows in _rows_grouped_by_step_count(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed_rows]
        group_outputs, group_timing = pytorch_asian_delta_crn_batch(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=device,
            relative_bump=relative_bump,
            batch_rows=batch_rows,
        )
        for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
            outputs[index] = output
        for name, value in group_timing.items():
            if name != "wall_seconds":
                timing[name] = timing.get(name, 0.0) + float(value)
    timing["wall_seconds"] = perf_counter() - started
    return [output for output in outputs if output is not None], timing

def python_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    device: str,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return python_outputs(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        target_dt=target_dt,
        device=device,
    )


def python_delta_crn_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    device: str,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return python_delta_crn_outputs(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        target_dt=target_dt,
        device=device,
        relative_bump=relative_bump,
    )


def cpp_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    use_gpu: bool,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return _cpp_native_outputs(
        rows,
        model_by_id,
        product_by_id,
        row_type=CRoughHestonRow,
        output_type=CMonteCarloOutput,
        product_symbol="asian_arithmetic",
        num_paths=num_paths,
        target_dt=target_dt,
        use_gpu=use_gpu,
    )


def cpp_delta_crn_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    use_gpu: bool,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return _cpp_native_outputs(
        rows,
        model_by_id,
        product_by_id,
        row_type=CRoughHestonRow,
        output_type=CPriceDeltaOutput,
        product_symbol="asian_arithmetic",
        num_paths=num_paths,
        target_dt=target_dt,
        use_gpu=use_gpu,
        relative_bump=relative_bump,
    )


from tools.registry.result.common.production_pipeline import (  # noqa: E402
    ProductionPipeline,
    ProductionPipelineConfig,
)

PRODUCTION_MODEL_ID = "rough_heston_01"
PRODUCTION_PRODUCT_ID = "asian_arithmetic_calls_01"
PRODUCTION_RESULT_VERSION = "01"
PRODUCTION_ROW_COUNT = 1_000
AUDIT_ROW_COUNT = 100

_PRODUCTION_PIPELINE = ProductionPipeline(
    ProductionPipelineConfig(
        project_root=PROJECT_ROOT,
        model_id=PRODUCTION_MODEL_ID,
        product_id=PRODUCTION_PRODUCT_ID,
        result_version=PRODUCTION_RESULT_VERSION,
        model_name="Rough Heston",
        payoff_name="Asian Arithmetic Call",
        first_seed=DEFAULT_FIRST_SEED,
        default_num_paths=DEFAULT_NUM_PATHS,
        default_target_dt=DEFAULT_TARGET_DT,
        default_relative_bump=DEFAULT_RELATIVE_BUMP,
        cpp_price=cpp_outputs_for_time_grid,
        python_price=python_outputs_for_time_grid,
        cpp_delta=cpp_delta_crn_outputs_for_time_grid,
        python_delta=python_delta_crn_outputs_for_time_grid,
        source_files_for_engine=source_files_for_engine,
        references_for_engine=references_for_engine,
        time_grid_documentation=time_grid_documentation,
        price_outputs_documentation=price_only_outputs_documentation,
        delta_outputs_documentation=delta_crn_outputs_documentation,
        delta_method_documentation=delta_method_documentation,
        summary_details={"scheme": "eight-factor Markovian approximation"},
    )
)


def generate_production_cpp_gpu_result(**kwargs: Any) -> Path:
    return _PRODUCTION_PIPELINE.generate_production_cpp_gpu_result(**kwargs)


def generate_production_cpp_gpu_delta_crn_result(**kwargs: Any) -> Path:
    return _PRODUCTION_PIPELINE.generate_production_cpp_gpu_result(
        delta_crn=True, **kwargs
    )


def generate_validation_reprice_result(**kwargs: Any) -> Path:
    return _PRODUCTION_PIPELINE.generate_validation_reprice_result(**kwargs)
