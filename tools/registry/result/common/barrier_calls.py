"""Production and validation orchestration shared by barrier call datasets."""

from __future__ import annotations

import ctypes
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from tools.common.native_library import load_cpp_library
from tools.common.time_grid import step_count_for_maturity
from tools.registry.result.common.metadata import (
    price_only_outputs_documentation,
    time_grid_documentation,
)
from tools.registry.result.common.production_pipeline import (
    ProductionPipeline,
    ProductionPipelineConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PYTHON = PROJECT_ROOT / "src_python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))
DEFAULT_NUM_PATHS = 16_384
DEFAULT_TARGET_DT = "1/52"
DEFAULT_FIRST_SEED = 935_200_001


class CBlackScholesRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("volatility", ctypes.c_double),
        ("strike", ctypes.c_double), ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


class CHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double), ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double), ("rho", ctypes.c_double),
        ("strike", ctypes.c_double), ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64), ("scheme", ctypes.c_int),
    ]


class CRoughBergomiRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("forward_variance", ctypes.c_double),
        ("eta", ctypes.c_double), ("alpha", ctypes.c_double),
        ("rho", ctypes.c_double), ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double), ("seed", ctypes.c_uint64),
    ]


class CRoughHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double), ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double), ("hurst", ctypes.c_double),
        ("rho", ctypes.c_double), ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double), ("seed", ctypes.c_uint64),
    ]


class CBarrierTerms(ctypes.Structure):
    _fields_ = [("barrier", ctypes.c_double)]


class CBlackScholesBarrierRow(ctypes.Structure):
    _fields_ = [("model", CBlackScholesRow), ("product", CBarrierTerms)]


class CHestonBarrierRow(ctypes.Structure):
    _fields_ = [("model", CHestonRow), ("product", CBarrierTerms)]


class CRoughBergomiBarrierRow(ctypes.Structure):
    _fields_ = [("model", CRoughBergomiRow), ("product", CBarrierTerms)]


class CRoughHestonBarrierRow(ctypes.Structure):
    _fields_ = [("model", CRoughHestonRow), ("product", CBarrierTerms)]


class CMonteCarloOutput(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("standard_error", ctypes.c_double)]


ROW_TYPES = {
    "black_scholes": CBlackScholesBarrierRow,
    "heston": CHestonBarrierRow,
    "rough_bergomi": CRoughBergomiBarrierRow,
    "rough_heston": CRoughHestonBarrierRow,
}

MODEL_NAMES = {
    "black_scholes": "Black Scholes",
    "heston": "Heston",
    "rough_bergomi": "Rough Bergomi",
    "rough_heston": "Rough Heston",
}

PAYOFF_NAMES = {
    "down_and_out_calls": "Down And Out Call",
    "down_and_in_calls": "Down And In Call",
    "up_and_out_calls": "Up And Out Call",
    "up_and_in_calls": "Up And In Call",
}


def _native_row(model_family: str, row: dict[str, Any], models: dict[str, Any], products: dict[str, Any]):
    model = models[row["model_id"]]
    product = products[row["product_id"]]
    strike = float(product["strike"])
    maturity = float(product["maturity"])
    seed = int(row["seed"])
    terms = CBarrierTerms(float(product["barrier"]))
    if model_family == "black_scholes":
        native = CBlackScholesRow(
            float(model["spot"]), float(model["risk_free_rate"]),
            float(model.get("dividend_yield", 0.0)), float(model["volatility"]),
            strike, maturity, seed,
        )
        return CBlackScholesBarrierRow(native, terms)
    if model_family == "heston":
        native = CHestonRow(
            float(model["spot"]), float(model["risk_free_rate"]),
            float(model.get("dividend_yield", 0.0)), float(model["initial_variance"]),
            float(model["kappa"]), float(model["theta"]),
            float(model["volatility_of_variance"]), float(model["rho"]),
            strike, maturity, seed, 2,
        )
        return CHestonBarrierRow(native, terms)
    if model_family == "rough_heston":
        native = CRoughHestonRow(
            float(model["spot"]), float(model["risk_free_rate"]),
            float(model.get("dividend_yield", 0.0)), float(model["initial_variance"]),
            float(model["kappa"]), float(model["theta"]),
            float(model["volatility_of_variance"]), float(model["hurst"]),
            float(model["rho"]), strike, maturity, seed,
        )
        return CRoughHestonBarrierRow(native, terms)
    native = CRoughBergomiRow(
        float(model["spot"]), float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)), float(model["forward_variance"]),
        float(model["eta"]), float(model["hurst"]) - 0.5,
        float(model.get("rho", model.get("correlation"))),
        strike, maturity, seed,
    )
    return CRoughBergomiBarrierRow(native, terms)


def _groups(rows: list[dict[str, Any]], products: dict[str, Any], target_dt: str | float):
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        steps = step_count_for_maturity(float(products[row["product_id"]]["maturity"]), target_dt)
        grouped[steps].append((index, row))
    return sorted(grouped.items())


def cpp_outputs(
    model_family: str,
    product_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float,
    use_gpu: bool,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    row_type = ROW_TYPES[model_family]
    library = load_cpp_library()
    function = getattr(
        library,
        f"ai_factory_price_{model_family}_{product_family}_{'gpu' if use_gpu else 'cpu'}_batch",
    )
    args = [ctypes.POINTER(row_type), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(CMonteCarloOutput)]
    if use_gpu:
        args.append(ctypes.POINTER(ctypes.c_double))
    function.argtypes = args
    function.restype = ctypes.c_int
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    ordered: list[dict[str, float] | None] = [None] * len(rows)
    kernel_seconds = 0.0
    started = perf_counter()
    for num_steps, indexed in _groups(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed]
        native_rows = (row_type * len(group_rows))(*[
            _native_row(model_family, row, model_by_id, product_by_id) for row in group_rows
        ])
        native_outputs = (CMonteCarloOutput * len(group_rows))()
        if use_gpu:
            group_kernel = ctypes.c_double()
            status = function(native_rows, len(group_rows), num_paths, num_steps,
                              native_outputs, ctypes.byref(group_kernel))
            kernel_seconds += float(group_kernel.value)
        else:
            status = function(native_rows, len(group_rows), num_paths, num_steps, native_outputs)
        if status != 0:
            raw = library.ai_factory_cuda_last_error()
            raise RuntimeError(raw.decode() if raw else "Native barrier pricing failed.")
        for (index, _), output in zip(indexed, native_outputs, strict=True):
            ordered[index] = {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return [output for output in ordered if output is not None], timing


def python_outputs(
    model_family: str,
    product_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float,
    device: str,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    module = importlib.import_module(
        f"ai_factory.pytorch.{model_family}.{product_family}"
    )
    ordered: list[dict[str, float] | None] = [None] * len(rows)
    totals = {"wall_seconds": 0.0, "simulation_seconds": 0.0, "payoff_seconds": 0.0}
    started = perf_counter()
    for num_steps, indexed in _groups(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed]
        outputs, timing = module.price_batch(
            group_rows, model_by_id, product_by_id,
            num_paths=num_paths, num_steps=num_steps, device=device,
        )
        for (index, _), output in zip(indexed, outputs, strict=True):
            ordered[index] = output
        for key in ("simulation_seconds", "payoff_seconds"):
            totals[key] += float(timing.get(key, 0.0))
    totals["wall_seconds"] = perf_counter() - started
    return [output for output in ordered if output is not None], totals


def references_for_engine(model_family: str, engine: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if model_family == "heston":
        references.append({"topic": "Heston QE-M simulation scheme", "reference": {
            "authors": "Andersen, L. B. G.", "year": 2007,
            "title": "Efficient Simulation of the Heston Stochastic Volatility Model"}})
    elif model_family == "rough_bergomi":
        references.append({"topic": "Rough Bergomi hybrid scheme", "reference": {
            "authors": "Bennedsen, M.; Lunde, A.; Pakkanen, M. S.", "year": 2017,
            "title": "Hybrid Scheme for Brownian Semistationary Processes"}})
    elif model_family == "rough_heston":
        references.append({"topic": "Markovian approximation of rough Heston", "reference": {
            "authors": "Abi Jaber, E.; El Euch, O.", "year": 2019,
            "title": "Multifactor Approximation of Rough Volatility Models"}})
    if engine.startswith("cpp_"):
        references.append({"topic": "C++ Philox counter-based random numbers", "reference": {
            "authors": "Salmon, J. K.; Moraes, M. A.; Dror, R. O.; Shaw, D. E.",
            "year": 2011, "title": "Parallel Random Numbers: As Easy as 1, 2, 3"}})
    return references


def source_files_for_engine(model_family: str, product_family: str, engine: str) -> list[str]:
    facade = f"tools/registry/result/{model_family}/{product_family}.py"
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/common/barrier_calls.py",
            f"src_python/ai_factory/pytorch/{model_family}/pathwise.py",
            f"src_python/ai_factory/pytorch/{model_family}/{product_family}.py",
            facade,
        ]
    if engine.startswith("cpp_cpu_"):
        return [
            "src_cpp/ai_factory/cpu/common/payoffs/barrier.hpp",
            f"src_cpp/ai_factory/cpu/{model_family}/common.cpp",
            f"src_cpp/ai_factory/cpu/{model_family}/{product_family}.cpp",
            "src_cpp/ai_factory/c_api/c_api.cpp",
            facade,
        ]
    return [
        "src_cpp/ai_factory/cuda/common/philox.cuh",
        "src_cpp/ai_factory/cuda/common/barrier_pricing.cuh",
        f"src_cpp/ai_factory/cuda/{model_family}/{product_family}.cu",
        "src_cpp/ai_factory/c_api/c_api.cpp",
        facade,
    ]


def build_pipeline(model_family: str, product_family: str, model_id: str) -> ProductionPipeline:
    return ProductionPipeline(ProductionPipelineConfig(
        project_root=PROJECT_ROOT,
        model_id=model_id,
        product_id=f"{product_family}_01",
        result_version="01",
        model_name=MODEL_NAMES[model_family],
        payoff_name=PAYOFF_NAMES[product_family],
        first_seed=DEFAULT_FIRST_SEED,
        default_num_paths=DEFAULT_NUM_PATHS,
        default_target_dt=DEFAULT_TARGET_DT,
        cpp_price=lambda *args, **kwargs: cpp_outputs(model_family, product_family, *args, **kwargs),
        python_price=lambda *args, **kwargs: python_outputs(model_family, product_family, *args, **kwargs),
        source_files_for_engine=lambda engine: source_files_for_engine(model_family, product_family, engine),
        references_for_engine=lambda engine: references_for_engine(model_family, engine),
        time_grid_documentation=time_grid_documentation,
        price_outputs_documentation=price_only_outputs_documentation,
        yaml_details={"monitoring": {
            "type": "discrete",
            "dates": "Every time-grid date after t0, including maturity",
            "rebate": 0.0,
        }},
    ))


def price_from_paths(
    paths: list[list[float]],
    *,
    model: dict[str, Any],
    product: dict[str, Any],
    up: bool,
    knock_in: bool,
) -> dict[str, float]:
    """Reprice one barrier call from reconstructed monitored spot paths."""

    import math

    barrier = float(product["barrier"])
    strike = float(product["strike"])
    discount = math.exp(
        -float(model["risk_free_rate"]) * float(product["maturity"])
    )
    payoffs = []
    for path in paths:
        monitored = path[1:]
        hit = max(monitored) >= barrier if up else min(monitored) <= barrier
        active = hit if knock_in else not hit
        payoffs.append(discount * max(path[-1] - strike, 0.0) if active else 0.0)
    count = len(payoffs)
    price = sum(payoffs) / count
    variance = sum((payoff - price) ** 2 for payoff in payoffs) / (count - 1)
    return {
        "price": price,
        "standard_error": math.sqrt(max(variance, 0.0) / count),
    }
