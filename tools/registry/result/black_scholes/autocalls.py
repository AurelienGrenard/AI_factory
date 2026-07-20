"""Black-Scholes autocall production and validation pipeline."""

from functools import partial
from pathlib import Path
from typing import Any

from tools.registry.result.common.autocalls import (
    DEFAULT_FIRST_SEED,
    DEFAULT_NUM_PATHS,
    DEFAULT_TARGET_DT,
    cpp_outputs,
    economic_diagnostics as _economic_diagnostics,
    python_outputs,
)
from tools.registry.result.common.metadata import (
    price_only_outputs_documentation,
    time_grid_documentation,
)
from tools.registry.result.common.production_pipeline import (
    ProductionPipeline,
    ProductionPipelineConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRODUCTION_MODEL_ID = "black_scholes_01"
PRODUCTION_PRODUCT_ID = "autocalls_01"
PRODUCTION_RESULT_VERSION = "01"
PRODUCTION_ROW_COUNT = 1_000
AUDIT_ROW_COUNT = 100


def references_for_engine(engine: str) -> list[dict[str, Any]]:
    if not engine.startswith("cpp_"):
        return []
    return [{
        "topic": "C++ Philox counter-based random numbers",
        "reference": {
            "authors": "Salmon, J. K.; Moraes, M. A.; Dror, R. O.; Shaw, D. E.",
            "year": 2011,
            "title": "Parallel Random Numbers: As Easy as 1, 2, 3",
        },
    }]


def source_files_for_engine(engine: str) -> list[str]:
    common = ["tools/registry/result/black_scholes/autocalls.py"]
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/common/autocalls.py",
            "src_python/ai_factory/pytorch/black_scholes/pathwise.py",
            "src_python/ai_factory/pytorch/black_scholes/autocalls.py",
            *common,
        ]
    if engine.startswith("cpp_cpu_"):
        return [
            "src_cpp/ai_factory/cpu/common/payoffs/autocall.hpp",
            "src_cpp/ai_factory/cpu/black_scholes/common.cpp",
            "src_cpp/ai_factory/cpu/black_scholes/autocalls.cpp",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            *common,
        ]
    return [
        "src_cpp/ai_factory/cuda/common/philox.cuh",
        "src_cpp/ai_factory/cuda/common/autocall.cuh",
        "src_cpp/ai_factory/cuda/black_scholes/dynamics.cuh",
        "src_cpp/ai_factory/cuda/black_scholes/autocalls.cu",
        "src_cpp/ai_factory/c_api/c_api.cpp",
        *common,
    ]


_PIPELINE = ProductionPipeline(ProductionPipelineConfig(
    project_root=PROJECT_ROOT,
    model_id=PRODUCTION_MODEL_ID,
    product_id=PRODUCTION_PRODUCT_ID,
    result_version=PRODUCTION_RESULT_VERSION,
    model_name="Black Scholes",
    payoff_name="Memory Autocall",
    first_seed=DEFAULT_FIRST_SEED,
    default_num_paths=DEFAULT_NUM_PATHS,
    default_target_dt=DEFAULT_TARGET_DT,
    cpp_price=partial(cpp_outputs, "black_scholes"),
    python_price=partial(python_outputs, "black_scholes"),
    source_files_for_engine=source_files_for_engine,
    references_for_engine=references_for_engine,
    time_grid_documentation=time_grid_documentation,
    price_outputs_documentation=price_only_outputs_documentation,
))

generate_production_cpp_gpu_result = _PIPELINE.generate_production_cpp_gpu_result
generate_validation_reprice_result = _PIPELINE.generate_validation_reprice_result
economic_diagnostics = partial(_economic_diagnostics, "black_scholes")
