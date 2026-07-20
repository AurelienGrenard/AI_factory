"""Shared implementation for Heston asian arithmetic priced result recipes."""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PYTHON = PROJECT_ROOT / "src_python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from ai_factory.pytorch.heston.asian_arithmetic_calls import (
    price_batch as pytorch_asian_price_batch,
    price_delta_crn_batch as pytorch_asian_delta_crn_batch,
)
from tools.registry.result.common.metadata import (
    DEFAULT_RELATIVE_BUMP,
    DEFAULT_TARGET_DT,
    delta_crn_outputs_documentation,
    delta_method_documentation,
    price_only_outputs_documentation,
    time_grid_documentation,
)
from tools.registry.result.common.execution import (
    ordered_outputs_from_groups as _ordered_outputs_from_groups,
    rows_grouped_by_step_count as _rows_grouped_by_step_count,
)
from tools.registry.result.heston.common import (
    CHestonRow,
    HESTON_SCHEME,
    cpp_row as _cpp_row,
    load_cpp_library as _load_cpp_library,
    raise_cpp_error as _raise_cpp_error,
)

DEFAULT_NUM_PATHS = 16_384
DEFAULT_FIRST_SEED = 931_700_001
DEFAULT_PYTORCH_BATCH_ROWS = 16


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
            "topic": "Heston QE-M simulation scheme",
            "reference": {
                "authors": "Andersen, L. B. G.",
                "year": 2007,
                "title": "Efficient Simulation of the Heston Stochastic Volatility Model",
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
    common = ["tools/registry/result/heston/asian_arithmetic_calls.py"]
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/heston/common.py",
            "src_python/ai_factory/pytorch/heston/pathwise.py",
            "src_python/ai_factory/pytorch/heston/asian_arithmetic_calls.py",
            *common,
        ]
    if engine in {"cpp_cpu_philox", "cpp_cpu_philox_delta_crn"}:
        return [
            "src_cpp/ai_factory/cpu/common/philox.cpp",
            "src_cpp/ai_factory/cpu/common/payoffs/asian.hpp",
            "src_cpp/ai_factory/cpu/heston/common.cpp",
            "src_cpp/ai_factory/cpu/heston/asian_arithmetic_calls.cpp",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    if engine in {"cpp_gpu_philox", "cpp_gpu_philox_delta_crn"}:
        return [
            "src_cpp/ai_factory/cuda/common/philox.cuh",
            "src_cpp/ai_factory/cuda/common/reductions.cuh",
            "src_cpp/ai_factory/cuda/common/runtime.cuh",
            "src_cpp/ai_factory/cuda/common/types.cuh",
            "src_cpp/ai_factory/cuda/heston/api.cuh",
            "src_cpp/ai_factory/cuda/heston/dynamics.cuh",
            "src_cpp/ai_factory/cuda/heston/asian_arithmetic_calls.cu",
            "src_cpp/ai_factory/cuda/heston/paths.cu",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    raise ValueError(f"Unsupported engine: {engine}")


def _summarize(discounted: torch.Tensor) -> dict[str, float]:
    price = discounted.mean()
    stderr = discounted.std(unbiased=True) / math.sqrt(discounted.numel())
    return {"price": float(price.cpu()), "standard_error": float(stderr.cpu())}


def _summarize_price_delta(
    discounted: torch.Tensor,
    delta_paths: torch.Tensor,
) -> dict[str, float]:
    price = discounted.mean()
    price_stderr = discounted.std(unbiased=True) / math.sqrt(discounted.numel())
    delta = delta_paths.mean()
    delta_stderr = delta_paths.std(unbiased=True) / math.sqrt(delta_paths.numel())
    return {
        "price": float(price.cpu()),
        "standard_error": float(price_stderr.cpu()),
        "delta": float(delta.cpu()),
        "delta_standard_error": float(delta_stderr.cpu()),
    }


def python_outputs(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    heston_scheme: str = HESTON_SCHEME,
    batch_rows: int = DEFAULT_PYTORCH_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if heston_scheme != HESTON_SCHEME:
        raise ValueError(
            f"PyTorch Heston Asian supports {HESTON_SCHEME!r}; got {heston_scheme!r}."
        )
    return pytorch_asian_price_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        batch_rows=batch_rows,
    )


def python_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    device: str,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    groups = _rows_grouped_by_step_count(rows, product_by_id, target_dt)
    started = perf_counter()

    def run_group(
        num_steps: int,
        group_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        return python_outputs(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=device,
            heston_scheme=heston_scheme,
        )

    outputs, timing = _ordered_outputs_from_groups(rows, groups, run_group)
    timing["wall_seconds"] = perf_counter() - started
    return outputs, timing


def python_delta_crn_outputs(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
    heston_scheme: str = HESTON_SCHEME,
    batch_rows: int = DEFAULT_PYTORCH_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if heston_scheme != HESTON_SCHEME:
        raise ValueError(
            f"PyTorch Heston Asian supports {HESTON_SCHEME!r}; got {heston_scheme!r}."
        )
    return pytorch_asian_delta_crn_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        relative_bump=relative_bump,
        batch_rows=batch_rows,
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
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    groups = _rows_grouped_by_step_count(rows, product_by_id, target_dt)
    started = perf_counter()

    def run_group(
        num_steps: int,
        group_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        return python_delta_crn_outputs(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=device,
            relative_bump=relative_bump,
            heston_scheme=heston_scheme,
        )

    outputs, timing = _ordered_outputs_from_groups(rows, groups, run_group)
    timing["wall_seconds"] = perf_counter() - started
    return outputs, timing


def _configure_cpp_library(library: ctypes.CDLL) -> None:
    row_ptr = ctypes.POINTER(CHestonRow)
    mc_ptr = ctypes.POINTER(CMonteCarloOutput)
    price_delta_ptr = ctypes.POINTER(CPriceDeltaOutput)
    seconds_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cuda_warmup.argtypes = []
    library.ai_factory_cuda_warmup.restype = ctypes.c_int
    library.ai_factory_price_heston_asian_arithmetic_cpu.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        mc_ptr,
    ]
    library.ai_factory_price_heston_asian_arithmetic_cpu.restype = ctypes.c_int
    if hasattr(library, "ai_factory_price_heston_asian_arithmetic_cpu_batch"):
        library.ai_factory_price_heston_asian_arithmetic_cpu_batch.argtypes = [
            ctypes.POINTER(CHestonRow),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(CMonteCarloOutput),
        ]
        library.ai_factory_price_heston_asian_arithmetic_cpu_batch.restype = ctypes.c_int
    if hasattr(library, "ai_factory_price_heston_asian_arithmetic_gpu_batch"):
        library.ai_factory_price_heston_asian_arithmetic_gpu_batch.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            mc_ptr,
            seconds_ptr,
        ]
        library.ai_factory_price_heston_asian_arithmetic_gpu_batch.restype = ctypes.c_int
    if hasattr(library, "ai_factory_price_heston_asian_arithmetic_delta_crn_cpu"):
        library.ai_factory_price_heston_asian_arithmetic_delta_crn_cpu.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            price_delta_ptr,
        ]
        library.ai_factory_price_heston_asian_arithmetic_delta_crn_cpu.restype = ctypes.c_int
    if hasattr(library, "ai_factory_price_heston_asian_arithmetic_delta_crn_cpu_batch"):
        library.ai_factory_price_heston_asian_arithmetic_delta_crn_cpu_batch.argtypes = [
            ctypes.POINTER(CHestonRow),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(CPriceDeltaOutput),
        ]
        library.ai_factory_price_heston_asian_arithmetic_delta_crn_cpu_batch.restype = ctypes.c_int
    if hasattr(library, "ai_factory_price_heston_asian_arithmetic_delta_crn_gpu_batch"):
        library.ai_factory_price_heston_asian_arithmetic_delta_crn_gpu_batch.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            price_delta_ptr,
            seconds_ptr,
        ]
        library.ai_factory_price_heston_asian_arithmetic_delta_crn_gpu_batch.restype = ctypes.c_int


def cpp_outputs(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    library = _load_cpp_library()
    _configure_cpp_library(library)
    outputs: list[dict[str, float]] = []
    started = perf_counter()
    kernel_seconds = 0.0
    if not use_gpu and hasattr(library, "ai_factory_price_heston_asian_arithmetic_cpu_batch"):
        row_array_type = CHestonRow * len(rows)
        output_array_type = CMonteCarloOutput * len(rows)
        row_array = row_array_type(
            *[
                _cpp_row(
                    row,
                    model_by_id[row["model_id"]],
                    product_by_id[row["product_id"]],
                    heston_scheme,
                )
                for row in rows
            ],
        )
        output_array = output_array_type()
        status = library.ai_factory_price_heston_asian_arithmetic_cpu_batch(
            row_array,
            len(rows),
            num_paths,
            num_steps,
            output_array,
        )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        outputs = [
            {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
            for output in output_array
        ]
        return outputs, {"wall_seconds": perf_counter() - started}
    if use_gpu and hasattr(library, "ai_factory_price_heston_asian_arithmetic_gpu_batch"):
        row_array_type = CHestonRow * len(rows)
        output_array_type = CMonteCarloOutput * len(rows)
        row_array = row_array_type(
            *[
                _cpp_row(
                    row,
                    model_by_id[row["model_id"]],
                    product_by_id[row["product_id"]],
                    heston_scheme,
                )
                for row in rows
            ],
        )
        output_array = output_array_type()
        elapsed = ctypes.c_double(0.0)
        status = library.ai_factory_price_heston_asian_arithmetic_gpu_batch(
            row_array,
            len(rows),
            num_paths,
            num_steps,
            output_array,
            ctypes.byref(elapsed),
        )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        outputs = [
            {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
            for output in output_array
        ]
        return outputs, {
            "wall_seconds": perf_counter() - started,
            "kernel_seconds": float(elapsed.value),
        }

    for row in rows:
        model = model_by_id[row["model_id"]]
        product = product_by_id[row["product_id"]]
        cpp_row = _cpp_row(row, model, product, heston_scheme)
        output = CMonteCarloOutput()
        if use_gpu:
            raise RuntimeError("C++ library does not expose Asian GPU single-row pricing.")
        else:
            status = library.ai_factory_price_heston_asian_arithmetic_cpu(
                ctypes.byref(cpp_row),
                num_paths,
                num_steps,
                ctypes.byref(output),
            )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        outputs.append(
            {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
        )
    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return outputs, timing


def cpp_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    use_gpu: bool,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    groups = _rows_grouped_by_step_count(rows, product_by_id, target_dt)
    if use_gpu:
        library = _load_cpp_library()
        _configure_cpp_library(library)
        if library.ai_factory_cuda_warmup() != 0:
            _raise_cpp_error(library, "C++ CUDA warm-up failed")
    started = perf_counter()

    def run_group(
        num_steps: int,
        group_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        return cpp_outputs(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            use_gpu=use_gpu,
            heston_scheme=heston_scheme,
        )

    outputs, timing = _ordered_outputs_from_groups(rows, groups, run_group)
    timing["wall_seconds"] = perf_counter() - started
    return outputs, timing


def cpp_delta_crn_outputs(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    library = _load_cpp_library()
    _configure_cpp_library(library)
    outputs: list[dict[str, float]] = []
    started = perf_counter()
    kernel_seconds = 0.0
    if use_gpu:
        if not hasattr(library, "ai_factory_price_heston_asian_arithmetic_delta_crn_gpu_batch"):
            raise RuntimeError("C++ library does not expose delta CRN GPU batch pricing.")
        row_array_type = CHestonRow * len(rows)
        output_array_type = CPriceDeltaOutput * len(rows)
        row_array = row_array_type(
            *[
                _cpp_row(
                    row,
                    model_by_id[row["model_id"]],
                    product_by_id[row["product_id"]],
                    heston_scheme,
                )
                for row in rows
            ],
        )
        output_array = output_array_type()
        elapsed = ctypes.c_double(0.0)
        status = library.ai_factory_price_heston_asian_arithmetic_delta_crn_gpu_batch(
            row_array,
            len(rows),
            num_paths,
            num_steps,
            relative_bump,
            output_array,
            ctypes.byref(elapsed),
        )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        outputs = [
            {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
                "delta": float(output.delta),
                "delta_standard_error": float(output.delta_standard_error),
            }
            for output in output_array
        ]
        return outputs, {
            "wall_seconds": perf_counter() - started,
            "kernel_seconds": float(elapsed.value),
        }

    if hasattr(library, "ai_factory_price_heston_asian_arithmetic_delta_crn_cpu_batch"):
        row_array_type = CHestonRow * len(rows)
        output_array_type = CPriceDeltaOutput * len(rows)
        row_array = row_array_type(
            *[
                _cpp_row(
                    row,
                    model_by_id[row["model_id"]],
                    product_by_id[row["product_id"]],
                    heston_scheme,
                )
                for row in rows
            ],
        )
        output_array = output_array_type()
        status = library.ai_factory_price_heston_asian_arithmetic_delta_crn_cpu_batch(
            row_array,
            len(rows),
            num_paths,
            num_steps,
            relative_bump,
            output_array,
        )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        outputs = [
            {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
                "delta": float(output.delta),
                "delta_standard_error": float(output.delta_standard_error),
            }
            for output in output_array
        ]
        return outputs, {"wall_seconds": perf_counter() - started}

    if not hasattr(library, "ai_factory_price_heston_asian_arithmetic_delta_crn_cpu"):
        raise RuntimeError("C++ library does not expose delta CRN CPU pricing.")
    for row in rows:
        model = model_by_id[row["model_id"]]
        product = product_by_id[row["product_id"]]
        cpp_row = _cpp_row(row, model, product, heston_scheme)
        output = CPriceDeltaOutput()
        status = library.ai_factory_price_heston_asian_arithmetic_delta_crn_cpu(
            ctypes.byref(cpp_row),
            num_paths,
            num_steps,
            relative_bump,
            ctypes.byref(output),
        )
        if status != 0:
            error = library.ai_factory_cuda_last_error()
            raise RuntimeError(error.decode() if error else "Unknown C++ error")
        outputs.append(
            {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
                "delta": float(output.delta),
                "delta_standard_error": float(output.delta_standard_error),
            }
        )
    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return outputs, timing


def cpp_delta_crn_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float | int,
    use_gpu: bool,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    groups = _rows_grouped_by_step_count(rows, product_by_id, target_dt)
    if use_gpu:
        library = _load_cpp_library()
        _configure_cpp_library(library)
        if library.ai_factory_cuda_warmup() != 0:
            _raise_cpp_error(library, "C++ CUDA warm-up failed")
    started = perf_counter()

    def run_group(
        num_steps: int,
        group_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        return cpp_delta_crn_outputs(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            use_gpu=use_gpu,
            relative_bump=relative_bump,
            heston_scheme=heston_scheme,
        )

    outputs, timing = _ordered_outputs_from_groups(rows, groups, run_group)
    timing["wall_seconds"] = perf_counter() - started
    return outputs, timing


from tools.registry.result.common.production_pipeline import (  # noqa: E402
    ProductionPipeline,
    ProductionPipelineConfig,
)

PRODUCTION_MODEL_ID = "heston_01"
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
        model_name="Heston",
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
        summary_details={"heston_scheme": HESTON_SCHEME},
        pricing_kwargs={"heston_scheme": HESTON_SCHEME},
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
