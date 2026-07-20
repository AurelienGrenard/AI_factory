"""Shared implementation for Rough Heston American put production-audit recipes."""

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

from ai_factory.pytorch.rough_heston.american_puts import price_batch as python_price_batch
from tools.registry.result.common.execution import (
    rows_grouped_by_step_count as _rows_grouped_by_step_count,
)
from tools.registry.result.common.metadata import (
    DEFAULT_TARGET_DT,
    time_grid_documentation,
)
from tools.registry.result.rough_heston.common import (
    _rough_heston_row,
)
from tools.common.native_library import load_cpp_library as _load_cpp_library

DEFAULT_NUM_PATHS = 16_384
DEFAULT_FIRST_SEED = 931_700_001
DEFAULT_PYTORCH_BATCH_ROWS = 8


class CRoughHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double), ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double), ("hurst", ctypes.c_double),
        ("rho", ctypes.c_double), ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double), ("seed", ctypes.c_uint64),
    ]


def price_only_outputs_documentation() -> dict[str, dict[str, str]]:
    return {
        "price": {"estimator": "Longstaff-Schwartz Monte Carlo price"},
        "standard_error": {
            "estimator": "Monte Carlo standard error of realized discounted cashflows"
        },
    }


def references_for_engine(engine: str) -> list[dict[str, Any]]:
    references = [
        {
            "topic": "Markovian approximation of rough Heston",
            "reference": {
                "authors": "Abi Jaber, E.; El Euch, O.",
                "year": 2019,
                "title": "Multifactor Approximation of Rough Volatility Models",
            },
        },
        {
            "topic": "American option least-squares regression",
            "reference": {
                "authors": "Longstaff, F. A.; Schwartz, E. S.",
                "year": 2001,
                "title": "Valuing American Options by Simulation: A Simple Least-Squares Approach",
            },
        },
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
    return references


def source_files_for_engine(engine: str) -> list[str]:
    common = ["tools/registry/result/rough_heston/american_puts.py"]
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/common/american_options.py",
            "src_python/ai_factory/pytorch/rough_heston/common.py",
            "src_python/ai_factory/pytorch/rough_heston/american_puts.py",
            *common,
        ]
    if engine == "cpp_cpu_philox":
        return [
            "src_cpp/ai_factory/cpu/common/philox.cpp",
            "src_cpp/ai_factory/cpu/rough_heston/common.cpp",
            "src_cpp/ai_factory/cpu/common/payoffs/american_put.hpp",
            "src_cpp/ai_factory/cpu/rough_heston/american_puts.hpp",
            "src_cpp/ai_factory/cpu/rough_heston/american_puts.cpp",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    if engine == "cpp_gpu_philox":
        return [
            "src_cpp/ai_factory/cuda/common/philox.cuh",
            "src_cpp/ai_factory/cuda/common/runtime.cuh",
            "src_cpp/ai_factory/cuda/common/types.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/api.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/dynamics.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/american_puts.cuh",
            "src_cpp/ai_factory/cuda/rough_heston/american_puts.cu",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    raise ValueError(f"Unsupported engine: {engine}")


class CMonteCarloOutput(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("standard_error", ctypes.c_double),
    ]


def _library() -> ctypes.CDLL:
    library = _load_cpp_library()
    _configure_cpp_library(library)
    return library


def _configure_cpp_library(library: ctypes.CDLL) -> None:
    row_ptr = ctypes.POINTER(CRoughHestonRow)
    output_ptr = ctypes.POINTER(CMonteCarloOutput)
    seconds_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    library.ai_factory_price_rough_heston_american_put_cpu_batch.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        output_ptr,
    ]
    library.ai_factory_price_rough_heston_american_put_cpu_batch.restype = ctypes.c_int
    library.ai_factory_price_rough_heston_american_put_gpu_batch.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        output_ptr,
        seconds_ptr,
    ]
    library.ai_factory_price_rough_heston_american_put_gpu_batch.restype = ctypes.c_int


def cpp_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float,
    use_gpu: bool,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    started = perf_counter()
    kernel_seconds = 0.0
    library = _library()
    outputs: list[dict[str, float] | None] = [None] * len(rows)
    for num_steps, indexed_rows in _rows_grouped_by_step_count(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed_rows]
        row_array_type = CRoughHestonRow * len(group_rows)
        output_array_type = CMonteCarloOutput * len(group_rows)
        row_array = row_array_type(
            *[
                _rough_heston_row(
                    CRoughHestonRow,
                    row,
                    model_by_id,
                    product_by_id,
                )
                for row in group_rows
            ]
        )
        output_array = output_array_type()
        if use_gpu:
            elapsed = ctypes.c_double(0.0)
            status = library.ai_factory_price_rough_heston_american_put_gpu_batch(
                row_array,
                len(group_rows),
                num_paths,
                num_steps,
                output_array,
                ctypes.byref(elapsed),
            )
            kernel_seconds += float(elapsed.value)
        else:
            status = library.ai_factory_price_rough_heston_american_put_cpu_batch(
                row_array,
                len(group_rows),
                num_paths,
                num_steps,
                output_array,
            )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        for (index, _), output in zip(indexed_rows, output_array, strict=True):
            outputs[index] = {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return [output for output in outputs if output is not None], timing


def python_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float,
    device: str,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    started = perf_counter()
    groups = _rows_grouped_by_step_count(rows, product_by_id, target_dt)
    outputs: list[dict[str, float] | None] = [None] * len(rows)
    simulation_seconds = 0.0
    lsm_seconds = 0.0
    for num_steps, indexed_rows in groups:
        group_rows = [row for _, row in indexed_rows]
        group_outputs, group_timing = python_price_batch(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=device,
            batch_rows=DEFAULT_PYTORCH_BATCH_ROWS,
        )
        for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
            outputs[index] = output
        simulation_seconds += float(group_timing.get("simulation_seconds", 0.0))
        lsm_seconds += float(group_timing.get("lsm_seconds", 0.0))
    return [output for output in outputs if output is not None], {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "lsm_seconds": lsm_seconds,
    }


from tools.registry.result.common.production_pipeline import (  # noqa: E402
    ProductionPipeline,
    ProductionPipelineConfig,
)

PRODUCTION_MODEL_ID = "rough_heston_01"
PRODUCTION_PRODUCT_ID = "american_puts_01"
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
        payoff_name="American Put",
        first_seed=DEFAULT_FIRST_SEED,
        default_num_paths=DEFAULT_NUM_PATHS,
        default_target_dt=DEFAULT_TARGET_DT,
        cpp_price=cpp_outputs_for_time_grid,
        python_price=python_outputs_for_time_grid,
        source_files_for_engine=source_files_for_engine,
        references_for_engine=references_for_engine,
        time_grid_documentation=time_grid_documentation,
        price_outputs_documentation=price_only_outputs_documentation,
        summary_details={"scheme": "eight-factor Markovian approximation"},
        yaml_details={
            "exercise_policy": {
                "method": "Longstaff-Schwartz",
                "regression_target": (
                    "realized future cashflow discounted to current exercise date"
                ),
                "basis": {
                    "state": ["spot", "markovian_factors_Y1_to_Y8"],
                    "functions": [
                        "1",
                        "L1(S_t / K)",
                        "L2(S_t / K)",
                        "Y_t^i / theta for i=1,...,8",
                    ],
                    "regularization": "relative ridge 1e-10 on the normal matrix",
                },
            }
        },
    )
)


def generate_production_cpp_gpu_result(**kwargs: Any) -> Path:
    return _PRODUCTION_PIPELINE.generate_production_cpp_gpu_result(**kwargs)


def generate_validation_reprice_result(**kwargs: Any) -> Path:
    return _PRODUCTION_PIPELINE.generate_validation_reprice_result(**kwargs)
