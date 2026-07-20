"""Rough Bergomi hybrid-scheme autocall production and validation pipeline."""

from functools import partial
from pathlib import Path
from typing import Any

from tools.registry.result.common.autocalls import (
    DEFAULT_FIRST_SEED, DEFAULT_NUM_PATHS, DEFAULT_TARGET_DT,
    cpp_outputs, economic_diagnostics as _economic_diagnostics, python_outputs,
)
from tools.registry.result.common.metadata import price_only_outputs_documentation, time_grid_documentation
from tools.registry.result.common.production_pipeline import ProductionPipeline, ProductionPipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRODUCTION_MODEL_ID = "rough_bergomi_02"
PRODUCTION_PRODUCT_ID = "autocalls_01"
PRODUCTION_RESULT_VERSION = "01"
PRODUCTION_ROW_COUNT = 1_000
AUDIT_ROW_COUNT = 100


def references_for_engine(engine: str) -> list[dict[str, Any]]:
    references = [{
        "topic": "Rough Bergomi hybrid simulation scheme",
        "reference": {
            "authors": "Bennedsen, M.; Lunde, A.; Pakkanen, M. S.",
            "year": 2017,
            "title": "Hybrid Scheme for Brownian Semistationary Processes",
        },
    }]
    if engine.startswith("cpp_"):
        references.append({
            "topic": "C++ Philox counter-based random numbers",
            "reference": {
                "authors": "Salmon, J. K.; Moraes, M. A.; Dror, R. O.; Shaw, D. E.",
                "year": 2011,
                "title": "Parallel Random Numbers: As Easy as 1, 2, 3",
            },
        })
    return references


def source_files_for_engine(engine: str) -> list[str]:
    common = ["tools/registry/result/rough_bergomi/autocalls.py"]
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/common/autocalls.py",
            "src_python/ai_factory/pytorch/rough_bergomi/common.py",
            "src_python/ai_factory/pytorch/rough_bergomi/pathwise.py",
            "src_python/ai_factory/pytorch/rough_bergomi/autocalls.py", *common,
        ]
    if engine.startswith("cpp_cpu_"):
        return [
            "src_cpp/ai_factory/cpu/common/payoffs/autocall.hpp",
            "src_cpp/ai_factory/cpu/rough_bergomi/common.cpp",
            "src_cpp/ai_factory/cpu/rough_bergomi/autocalls.cpp",
            "src_cpp/ai_factory/c_api/c_api.cpp", *common,
        ]
    return [
        "src_cpp/ai_factory/cuda/common/philox.cuh",
        "src_cpp/ai_factory/cuda/common/autocall.cuh",
        "src_cpp/ai_factory/cuda/rough_bergomi/dynamics.cuh",
        "src_cpp/ai_factory/cuda/rough_bergomi/autocalls.cu",
        "src_cpp/ai_factory/c_api/c_api.cpp", *common,
    ]


_PIPELINE = ProductionPipeline(ProductionPipelineConfig(
    project_root=PROJECT_ROOT, model_id=PRODUCTION_MODEL_ID,
    product_id=PRODUCTION_PRODUCT_ID, result_version=PRODUCTION_RESULT_VERSION,
    model_name="Rough Bergomi", payoff_name="Memory Autocall",
    first_seed=DEFAULT_FIRST_SEED, default_num_paths=DEFAULT_NUM_PATHS,
    default_target_dt=DEFAULT_TARGET_DT,
    cpp_price=partial(cpp_outputs, "rough_bergomi"),
    python_price=partial(python_outputs, "rough_bergomi"),
    source_files_for_engine=source_files_for_engine,
    references_for_engine=references_for_engine,
    time_grid_documentation=time_grid_documentation,
    price_outputs_documentation=price_only_outputs_documentation,
    summary_details={"scheme": "hybrid"},
))

generate_production_cpp_gpu_result = _PIPELINE.generate_production_cpp_gpu_result
generate_validation_reprice_result = _PIPELINE.generate_validation_reprice_result
economic_diagnostics = partial(_economic_diagnostics, "rough_bergomi")
