"""Shared Black-Scholes production and validation result helpers."""

from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PYTHON = PROJECT_ROOT / "src_python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from ai_factory.pytorch.black_scholes import (
    american_puts,
    asian_arithmetic_calls,
    lookback_fixed_calls,
    volatility_swaps,
)
from tools.common.native_library import load_cpp_library
from tools.common.time_grid import step_count_for_maturity
from tools.registry.common.paths import registry_database_path, registry_relative_path
from tools.registry.common.timing import benchmark_pricing_call
from tools.registry.common.schema import (
    aligned_result_construction,
    canonical_timing,
    database_reference,
    primary_source_files,
)

DEFAULT_NUM_PATHS = 16_384
DEFAULT_TARGET_DT = "1/52"
DEFAULT_FIRST_SEED = 931_700_001
DEFAULT_RELATIVE_BUMP = 5.0e-4
AUDIT_ROW_COUNT = 100
PRODUCTION_ROW_COUNT = 1_000
AUDIT_SUFFIX = "first_100"


class CBlackScholesRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("volatility", ctypes.c_double),
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


@dataclass(frozen=True)
class ProductConfig:
    product_family: str
    payoff_name: str
    c_function: str
    c_delta_function: str | None
    pytorch_module: Any


PRODUCTS = {
    "lookback_fixed_calls": ProductConfig(
        "lookback_fixed_calls",
        "Lookback Fixed Call",
        "lookback_fixed",
        "lookback_fixed_delta_crn",
        lookback_fixed_calls,
    ),
    "asian_arithmetic_calls": ProductConfig(
        "asian_arithmetic_calls",
        "Asian Arithmetic Call",
        "asian_arithmetic",
        "asian_arithmetic_delta_crn",
        asian_arithmetic_calls,
    ),
    "volatility_swaps": ProductConfig(
        "volatility_swaps",
        "Volatility Swap",
        "volatility_swap",
        None,
        volatility_swaps,
    ),
    "american_puts": ProductConfig(
        "american_puts",
        "American Put",
        "american_put",
        None,
        american_puts,
    ),
}


def time_grid_documentation(target_dt: str | float | int) -> dict[str, str]:
    return {
        "rule": "nearest integer step count to target dt",
        "target_dt": str(target_dt),
        "step_count": "round(maturity / target_dt)",
        "effective_dt": "maturity / step_count",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _registry_path(tier: str, kind: str, section: str, database_id: str, suffix: str) -> Path:
    return registry_database_path(PROJECT_ROOT, tier, kind, section, database_id, suffix)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _configure_function(
    library: ctypes.CDLL,
    name: str,
    *,
    delta: bool,
    gpu: bool,
) -> Any:
    func = getattr(library, name)
    output_type = CPriceDeltaOutput if delta else CMonteCarloOutput
    args: list[Any] = [
        ctypes.POINTER(CBlackScholesRow),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    if delta:
        args.append(ctypes.c_double)
    args.append(ctypes.POINTER(output_type))
    if gpu:
        args.append(ctypes.POINTER(ctypes.c_double))
    func.argtypes = args
    func.restype = ctypes.c_int
    return func


def _configure_error(library: ctypes.CDLL) -> None:
    if hasattr(library, "ai_factory_cuda_last_error"):
        library.ai_factory_cuda_last_error.argtypes = []
        library.ai_factory_cuda_last_error.restype = ctypes.c_char_p


def _last_error(library: ctypes.CDLL) -> str:
    if not hasattr(library, "ai_factory_cuda_last_error"):
        return "unknown C++ error"
    raw = library.ai_factory_cuda_last_error()
    return raw.decode("utf-8") if raw else "unknown C++ error"


def aligned_rows(row_count: int, *, first_seed: int = DEFAULT_FIRST_SEED) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{index + 1:06d}",
            "model_id": f"{index + 1:06d}",
            "product_id": f"{index + 1:06d}",
            "seed": first_seed + index,
        }
        for index in range(row_count)
    ]


def _load_inputs(
    *,
    tier: str,
    model_id: str,
    product_id: str,
    row_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model_data = _read_json(_registry_path(tier, "models", "data", model_id, "json"))
    product_data = _read_json(_registry_path(tier, "products", "data", product_id, "json"))
    model_by_id = {row["id"]: row["parameters"] for row in model_data["models"]}
    product_by_id = {row["id"]: row["parameters"] for row in product_data["products"]}
    return aligned_rows(row_count), model_by_id, product_by_id


def _c_row(row: dict[str, Any], model_by_id: dict[str, Any], product_by_id: dict[str, Any]) -> CBlackScholesRow:
    model = model_by_id[row["model_id"]]
    product = product_by_id[row["product_id"]]
    return CBlackScholesRow(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)),
        float(model["volatility"]),
        float(product.get("strike", product.get("volatility_strike"))),
        float(product["maturity"]),
        int(row["seed"]),
    )


def _groups_by_step(
    rows: list[dict[str, Any]],
    product_by_id: dict[str, Any],
    target_dt: str | float,
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        maturity = float(product_by_id[row["product_id"]]["maturity"])
        groups.setdefault(step_count_for_maturity(maturity, target_dt), []).append(row)
    return groups


def cpp_outputs(
    *,
    product_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    target_dt: str | float,
    use_gpu: bool,
    delta: bool = False,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    config = PRODUCTS[product_family]
    if delta and config.c_delta_function is None:
        raise ValueError(f"{product_family} has no delta CRN configuration.")
    library = load_cpp_library()
    _configure_error(library)
    engine_name = config.c_delta_function if delta else config.c_function
    assert engine_name is not None
    symbol = f"ai_factory_price_black_scholes_{engine_name}_{'gpu' if use_gpu else 'cpu'}_batch"
    func = _configure_function(library, symbol, delta=delta, gpu=use_gpu)
    ordered: dict[str, dict[str, float]] = {}
    kernel_seconds = 0.0
    started = perf_counter()
    for num_steps, group_rows in _groups_by_step(rows, product_by_id, target_dt).items():
        array_type = CBlackScholesRow * len(group_rows)
        row_array = array_type(*[_c_row(row, model_by_id, product_by_id) for row in group_rows])
        output_type = CPriceDeltaOutput if delta else CMonteCarloOutput
        output_array_type = output_type * len(group_rows)
        output_array = output_array_type()
        if use_gpu:
            group_kernel = ctypes.c_double(0.0)
            if delta:
                status = func(
                    row_array,
                    len(group_rows),
                    num_paths,
                    num_steps,
                    relative_bump,
                    output_array,
                    ctypes.byref(group_kernel),
                )
            else:
                status = func(
                    row_array,
                    len(group_rows),
                    num_paths,
                    num_steps,
                    output_array,
                    ctypes.byref(group_kernel),
                )
            kernel_seconds += float(group_kernel.value)
        elif delta:
            status = func(
                row_array,
                len(group_rows),
                num_paths,
                num_steps,
                relative_bump,
                output_array,
            )
        else:
            status = func(row_array, len(group_rows), num_paths, num_steps, output_array)
        if status != 0:
            raise RuntimeError(_last_error(library))
        for row, output in zip(group_rows, output_array, strict=True):
            result = {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
            if delta:
                result["delta"] = float(output.delta)
                result["delta_standard_error"] = float(output.delta_standard_error)
            ordered[row["id"]] = result
    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return [ordered[row["id"]] for row in rows], timing


def pytorch_outputs(
    *,
    product_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    target_dt: str | float,
    device: str,
    delta: bool = False,
    relative_bump: float = DEFAULT_RELATIVE_BUMP,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    config = PRODUCTS[product_family]
    if product_family == "american_puts" and not delta:
        step_counts = [
            step_count_for_maturity(
                float(product_by_id[row["product_id"]]["maturity"]), target_dt
            )
            for row in rows
        ]
        return config.pytorch_module.price_variable_grid_batch(
            rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            step_counts=step_counts,
            device=device,
        )
    ordered: dict[str, dict[str, float]] = {}
    started = perf_counter()
    simulation = 0.0
    payoff = 0.0
    for num_steps, group_rows in _groups_by_step(rows, product_by_id, target_dt).items():
        if delta:
            outputs, timing = config.pytorch_module.price_delta_crn_batch(
                group_rows,
                model_by_id,
                product_by_id,
                num_paths=num_paths,
                num_steps=num_steps,
                device=device,
                relative_bump=relative_bump,
            )
        else:
            outputs, timing = config.pytorch_module.price_batch(
                group_rows,
                model_by_id,
                product_by_id,
                num_paths=num_paths,
                num_steps=num_steps,
                device=device,
            )
        simulation += float(timing.get("simulation_seconds", 0.0))
        payoff += float(timing.get("payoff_seconds", timing.get("lsm_seconds", 0.0)))
        for row, output in zip(group_rows, outputs, strict=True):
            ordered[row["id"]] = output
    return [ordered[row["id"]] for row in rows], {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation,
        "payoff_seconds": payoff,
    }


def _outputs_doc(delta: bool) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "price": {"estimator": "Monte Carlo discounted payoff mean"},
        "standard_error": {"estimator": "Monte Carlo standard error of discounted payoff"},
    }
    if delta:
        doc["delta"] = {"estimator": "Central finite difference with common random numbers"}
        doc["delta_standard_error"] = {
            "estimator": "Monte Carlo standard error of pathwise finite-difference delta"
        }
    return doc


def _source_files(product_family: str, engine: str) -> list[str]:
    if engine.startswith("python_"):
        return [
            "src_python/ai_factory/pytorch/black_scholes/pathwise.py",
            f"src_python/ai_factory/pytorch/black_scholes/{product_family}.py",
            "tools/registry/result/black_scholes/common.py",
        ]
    if engine.startswith("cpp_cpu"):
        return [
            "src_cpp/ai_factory/cpu/common/philox.cpp",
            "src_cpp/ai_factory/cpu/black_scholes/common.cpp",
            f"src_cpp/ai_factory/cpu/black_scholes/{product_family}.cpp",
            "tools/registry/result/black_scholes/common.py",
        ]
    return [
        "src_cpp/ai_factory/cuda/common/philox.cuh",
        "src_cpp/ai_factory/cuda/black_scholes/dynamics.cuh",
        f"src_cpp/ai_factory/cuda/black_scholes/{product_family}.cu",
        "src_cpp/ai_factory/c_api/c_api.cpp",
        "tools/registry/result/black_scholes/common.py",
    ]


def write_result(
    *,
    tier: str,
    database_id: str,
    generator_script: str,
    model_id: str,
    product_id: str,
    product_family: str,
    engine: str,
    device: str,
    rows: list[dict[str, Any]],
    outputs: list[dict[str, float]],
    timing: dict[str, float],
    num_paths: int,
    target_dt: str | float,
    delta: bool = False,
    source_production_result: str | None = None,
) -> Path:
    timing = canonical_timing(timing)
    json_path = _registry_path(tier, "results", "data", database_id, "json")
    yaml_path = _registry_path(tier, "results", "specifications", database_id, "yaml")
    priced_rows = [
        {
            "id": row["id"],
            "model_id": row["model_id"],
            "product_id": row["product_id"],
            "seed": row["seed"],
            "outputs": output,
        }
        for row, output in zip(rows, outputs, strict=True)
    ]
    construction = aligned_result_construction(DEFAULT_FIRST_SEED)
    model_database = database_reference(PROJECT_ROOT, tier, "models", model_id)
    product_database = database_reference(PROJECT_ROOT, tier, "products", product_id)
    _write_json(
        json_path,
        {
            "format": "ai_factory.results.v1",
            "database_id": database_id,
            "status": "priced",
            "specification": _relative(yaml_path),
            "generation_script": generator_script,
            "row_count": len(rows),
            "model_database": model_database,
            "product_database": product_database,
            "result_construction": construction,
            "engine": engine,
            "timing": timing,
            "results": priced_rows,
        },
    )
    config = PRODUCTS[product_family]
    source_files = primary_source_files(model_id, product_id, engine)
    if source_files[0] not in _source_files(product_family, engine):
        raise ValueError(f"Missing primary implementation source {source_files[0]}.")
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "num_paths": num_paths,
        "model": "Black Scholes",
        "payoff": config.payoff_name,
        "engine": engine,
        "device": device,
        "source_files": source_files,
    }
    if source_production_result is not None:
        summary["source_production_result"] = source_production_result
    _write_yaml(
        yaml_path,
        {
            "title": f"{model_id} x {product_id} {engine}",
            "format": "ai_factory.results.v1",
            "database_id": database_id,
            "status": "priced",
            "json_path": _relative(json_path),
            "generation_script": generator_script,
            "summary": summary,
            "time_grid": time_grid_documentation(target_dt),
            "outputs": _outputs_doc(delta),
            "model_database": model_database,
            "product_database": product_database,
            "result_construction": construction,
            "timing": timing,
        },
    )
    return json_path


def generate_result(
    *,
    tier: str,
    model_id: str,
    product_id: str,
    product_family: str,
    result_version: str,
    engine: str,
    row_count: int,
    num_paths: int = DEFAULT_NUM_PATHS,
    target_dt: str | float = DEFAULT_TARGET_DT,
    source_production_result: str | None = None,
    benchmark_row_multiplier: int = 1,
) -> Path:
    rows, model_by_id, product_by_id = _load_inputs(
        tier=tier,
        model_id=model_id,
        product_id=product_id,
        row_count=row_count,
    )
    delta = engine.endswith("delta_crn")
    if engine.startswith("python_"):
        device = "gpu" if "_gpu_" in engine else "cpu"
        pricing_kwargs = {
            "product_family": product_family,
            "rows": rows,
            "model_by_id": model_by_id,
            "product_by_id": product_by_id,
            "num_paths": num_paths,
            "target_dt": target_dt,
            "device": "cuda" if device == "gpu" else "cpu",
            "delta": delta,
        }
        pricing_call = lambda: pytorch_outputs(**pricing_kwargs)
    elif engine.startswith("cpp_"):
        device = "gpu" if "_gpu_" in engine else "cpu"
        pricing_kwargs = {
            "product_family": product_family,
            "rows": rows,
            "model_by_id": model_by_id,
            "product_by_id": product_by_id,
            "num_paths": num_paths,
            "target_dt": target_dt,
            "use_gpu": device == "gpu",
            "delta": delta,
        }
        pricing_call = lambda: cpp_outputs(**pricing_kwargs)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    if tier == "validation":
        outputs, timing = benchmark_pricing_call(
            pricing_call,
            repetitions=1,
            warmup_runs=1,
        )
        timing["benchmark_row_count"] = len(rows)
        timing["benchmark_workload"] = "validation slice"
    else:
        outputs, timing = pricing_call()
    if tier == "validation" and benchmark_row_multiplier > 1:
        benchmark_kwargs = {
            **pricing_kwargs,
            "rows": rows * benchmark_row_multiplier,
        }
        benchmark_call = (
            (lambda: pytorch_outputs(**benchmark_kwargs))
            if engine.startswith("python_")
            else (lambda: cpp_outputs(**benchmark_kwargs))
        )
        _, benchmark_timing = benchmark_pricing_call(
            benchmark_call,
            repetitions=1,
            warmup_runs=1,
        )
        timing["benchmark_seconds"] = float(benchmark_timing["benchmark_seconds"])
        timing["benchmark_row_count"] = len(benchmark_kwargs["rows"])
        timing["benchmark_row_multiplier"] = benchmark_row_multiplier
        timing["benchmark_repetitions"] = 1
        timing["benchmark_workload"] = "duplicated validation slice"
        if "kernel_seconds" in benchmark_timing:
            timing["benchmark_kernel_seconds"] = float(
                benchmark_timing.get(
                    "benchmark_kernel_seconds",
                    benchmark_timing["kernel_seconds"],
                )
            )
    database_id = f"{model_id}__{product_id}__{engine}_{result_version}"
    generator_script = registry_relative_path(
        PROJECT_ROOT,
        tier,
        "results",
        "generators",
        database_id,
        "py",
    )
    return write_result(
        tier=tier,
        database_id=database_id,
        generator_script=generator_script,
        model_id=model_id,
        product_id=product_id,
        product_family=product_family,
        engine=engine,
        device=device,
        rows=rows,
        outputs=outputs,
        timing=timing,
        num_paths=num_paths,
        target_dt=target_dt,
        delta=delta,
        source_production_result=source_production_result,
    )
