"""Registry orchestration shared by terminal equity call products."""

from __future__ import annotations

import ctypes
import importlib
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

from tools.common.native_library import load_cpp_library
from tools.common.time_grid import step_count_for_maturity
from tools.registry.common.paths import registry_database_path, registry_relative_path
from tools.registry.common.schema import (
    aligned_result_construction,
    canonical_timing,
    database_reference,
    exact_transition_time_grid,
    primary_source_files,
)
from tools.registry.common.timing import benchmark_pricing_call

DEFAULT_NUM_PATHS = 16_384
DEFAULT_TARGET_DT = "1/52"
DEFAULT_FIRST_SEED = 931_700_001
PRODUCTION_ROW_COUNT = 1_000
AUDIT_ROW_COUNT = 100


class CBlackScholesRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("volatility", ctypes.c_double),
        ("strike", ctypes.c_double), ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


class CRoughBergomiRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("forward_variance", ctypes.c_double),
        ("eta", ctypes.c_double), ("alpha", ctypes.c_double),
        ("rho", ctypes.c_double), ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double), ("seed", ctypes.c_uint64),
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


class CRoughHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double), ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double), ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double), ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double), ("hurst", ctypes.c_double),
        ("rho", ctypes.c_double), ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double), ("seed", ctypes.c_uint64),
    ]


class CMonteCarloOutput(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("standard_error", ctypes.c_double)]


@dataclass(frozen=True)
class ModelConfig:
    title: str
    row_type: type[ctypes.Structure]
    pytorch_package: str
    simulated_grid: bool


MODELS = {
    "black_scholes": ModelConfig("Black Scholes", CBlackScholesRow, "black_scholes", False),
    "heston": ModelConfig("Heston", CHestonRow, "heston", True),
    "rough_bergomi": ModelConfig("Rough Bergomi", CRoughBergomiRow, "rough_bergomi", True),
    "rough_heston": ModelConfig("Rough Heston", CRoughHestonRow, "rough_heston", True),
}
PRODUCTS = {
    "european_calls": ("european_call", "European Call"),
    "digital_calls": ("digital_call", "Cash-or-Nothing Digital Call"),
}


def _path(tier: str, kind: str, section: str, database_id: str, suffix: str) -> Path:
    return registry_database_path(PROJECT_ROOT, tier, kind, section, database_id, suffix)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(row_count: int) -> list[dict[str, Any]]:
    return [
        {"id": f"{i + 1:06d}", "model_id": f"{i + 1:06d}",
         "product_id": f"{i + 1:06d}", "seed": DEFAULT_FIRST_SEED + i}
        for i in range(row_count)
    ]


def _inputs(tier: str, model_id: str, product_id: str, row_count: int):
    models = _read(_path(tier, "models", "data", model_id, "json"))["models"]
    products = _read(_path(tier, "products", "data", product_id, "json"))["products"]
    return (
        _rows(row_count),
        {row["id"]: row["parameters"] for row in models},
        {row["id"]: row["parameters"] for row in products},
    )


def _native_row(model_family: str, row, model, product):
    common = (float(model["spot"]), float(model["risk_free_rate"]),
              float(model.get("dividend_yield", 0.0)))
    tail = (float(product["strike"]), float(product["maturity"]), int(row["seed"]))
    if model_family == "black_scholes":
        return CBlackScholesRow(*common, float(model["volatility"]), *tail)
    if model_family == "heston":
        return CHestonRow(
            *common, float(model["initial_variance"]), float(model["kappa"]),
            float(model["theta"]), float(model["volatility_of_variance"]),
            float(model["rho"]), *tail, 2
        )
    if model_family == "rough_heston":
        return CRoughHestonRow(
            *common, float(model["initial_variance"]), float(model["kappa"]),
            float(model["theta"]), float(model["volatility_of_variance"]),
            float(model["hurst"]), float(model["rho"]), *tail
        )
    return CRoughBergomiRow(
        *common, float(model["forward_variance"]), float(model["eta"]),
        float(model["hurst"]) - 0.5, float(model["correlation"]), *tail
    )


def _groups(config: ModelConfig, rows, products, target_dt):
    if not config.simulated_grid:
        return {1: rows}
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        steps = step_count_for_maturity(float(products[row["product_id"]]["maturity"]), target_dt)
        result.setdefault(steps, []).append(row)
    return result


def _native_outputs(model_family, product_family, rows, models, products, num_paths, target_dt, gpu):
    config = MODELS[model_family]
    c_product = PRODUCTS[product_family][0]
    library = load_cpp_library()
    symbol = f"ai_factory_price_{model_family}_{c_product}_{'gpu' if gpu else 'cpu'}_batch"
    function = getattr(library, symbol)
    args = [ctypes.POINTER(config.row_type), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(CMonteCarloOutput)]
    if gpu:
        args.append(ctypes.POINTER(ctypes.c_double))
    function.argtypes = args
    function.restype = ctypes.c_int
    if hasattr(library, "ai_factory_cuda_last_error"):
        library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    ordered = {}
    kernel = 0.0
    started = perf_counter()
    for steps, group in _groups(config, rows, products, target_dt).items():
        c_rows = (config.row_type * len(group))(*[
            _native_row(model_family, row, models[row["model_id"]], products[row["product_id"]])
            for row in group
        ])
        c_outputs = (CMonteCarloOutput * len(group))()
        if gpu:
            elapsed = ctypes.c_double()
            status = function(c_rows, len(group), num_paths, steps, c_outputs, ctypes.byref(elapsed))
            kernel += elapsed.value
        else:
            status = function(c_rows, len(group), num_paths, steps, c_outputs)
        if status:
            raw = library.ai_factory_cuda_last_error()
            raise RuntimeError(raw.decode() if raw else f"Native call {symbol} failed")
        for row, output in zip(group, c_outputs, strict=True):
            ordered[row["id"]] = {"price": output.price, "standard_error": output.standard_error}
    timing = {"wall_seconds": perf_counter() - started}
    if gpu:
        timing["kernel_seconds"] = kernel
    return [ordered[row["id"]] for row in rows], timing


def _pytorch_outputs(model_family, product_family, rows, models, products, num_paths, target_dt, device):
    config = MODELS[model_family]
    module = importlib.import_module(f"ai_factory.pytorch.{config.pytorch_package}.{product_family}")
    ordered = {}
    started = perf_counter()
    for steps, group in _groups(config, rows, products, target_dt).items():
        outputs, _ = module.price_batch(
            group, models, products, num_paths=num_paths, num_steps=steps, device=device
        )
        for row, output in zip(group, outputs, strict=True):
            ordered[row["id"]] = output
    return [ordered[row["id"]] for row in rows], {"wall_seconds": perf_counter() - started}


def _time_grid(config: ModelConfig, target_dt):
    if not config.simulated_grid:
        return exact_transition_time_grid("single terminal date T")
    return {
        "rule": "nearest integer step count to target dt", "target_dt": str(target_dt),
        "step_count": "round(maturity / target_dt)",
        "effective_dt": "maturity / step_count",
    }


def _write_result(*, tier, database_id, generator_script, model_id, product_id,
                  model_family, product_family, engine, rows, outputs, timing,
                  num_paths, target_dt, source_production_result):
    json_path = _path(tier, "results", "data", database_id, "json")
    yaml_path = _path(tier, "results", "specifications", database_id, "yaml")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    model_ref = database_reference(PROJECT_ROOT, tier, "models", model_id)
    product_ref = database_reference(PROJECT_ROOT, tier, "products", product_id)
    construction = aligned_result_construction(DEFAULT_FIRST_SEED)
    timing = canonical_timing(timing)
    payload = {
        "format": "ai_factory.results.v1", "database_id": database_id, "status": "priced",
        "specification": yaml_path.relative_to(PROJECT_ROOT).as_posix(),
        "generation_script": generator_script, "row_count": len(rows),
        "model_database": model_ref, "product_database": product_ref,
        "result_construction": construction, "engine": engine, "timing": timing,
        "results": [
            {**row, "outputs": output} for row, output in zip(rows, outputs, strict=True)
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = MODELS[model_family]
    summary = {
        "row_count": len(rows), "num_paths": num_paths, "model": config.title,
        "payoff": PRODUCTS[product_family][1], "engine": engine,
        "device": "gpu" if "_gpu_" in engine else "cpu",
        "source_files": primary_source_files(model_id, product_id, engine),
    }
    if source_production_result:
        summary["source_production_result"] = source_production_result
    spec = {
        "title": f"{model_id} x {product_id} {engine}", "format": "ai_factory.results.v1",
        "database_id": database_id, "status": "priced",
        "json_path": json_path.relative_to(PROJECT_ROOT).as_posix(),
        "generation_script": generator_script, "summary": summary,
        "time_grid": _time_grid(config, target_dt),
        "outputs": {
            "price": {"estimator": "Monte Carlo discounted payoff mean"},
            "standard_error": {"estimator": "Monte Carlo standard error of discounted payoff"},
        },
        "model_database": model_ref, "product_database": product_ref,
        "result_construction": construction, "timing": timing,
    }
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return json_path


def generate_result(*, tier: str, model_id: str, product_id: str,
                    model_family: str, product_family: str, result_version: str,
                    engine: str, row_count: int, num_paths: int = DEFAULT_NUM_PATHS,
                    target_dt: str | float = DEFAULT_TARGET_DT,
                    source_production_result: str | None = None) -> Path:
    rows, models, products = _inputs(tier, model_id, product_id, row_count)
    if engine.startswith("cpp_"):
        call = lambda: _native_outputs(
            model_family, product_family, rows, models, products, num_paths, target_dt,
            "_gpu_" in engine
        )
    elif engine.startswith("python_"):
        call = lambda: _pytorch_outputs(
            model_family, product_family, rows, models, products, num_paths, target_dt,
            "cuda" if "_gpu_" in engine else "cpu"
        )
    else:
        raise ValueError(f"Unsupported engine {engine}")
    outputs, timing = (
        benchmark_pricing_call(call, repetitions=1, warmup_runs=1)
        if tier == "validation" else call()
    )
    if tier == "validation":
        timing["benchmark_row_count"] = len(rows)
        timing["benchmark_workload"] = "validation slice"
    database_id = f"{model_id}__{product_id}__{engine}_{result_version}"
    generator_script = registry_relative_path(
        PROJECT_ROOT, tier, "results", "generators", database_id, "py"
    )
    return _write_result(
        tier=tier, database_id=database_id, generator_script=generator_script,
        model_id=model_id, product_id=product_id, model_family=model_family,
        product_family=product_family, engine=engine, rows=rows, outputs=outputs,
        timing=timing, num_paths=num_paths, target_dt=target_dt,
        source_production_result=source_production_result,
    )
