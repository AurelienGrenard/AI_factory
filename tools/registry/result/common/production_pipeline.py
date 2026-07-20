"""Shared production pricing and validation-repricing orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from tools.registry.common.paths import registry_database_path, registry_relative_path
from tools.registry.common.slicing import DEFAULT_VALIDATION_ROW_COUNT
from tools.registry.common.timing import benchmark_pricing_call
from tools.registry.common.schema import (
    aligned_result_construction,
    canonical_timing,
    database_reference,
    primary_source_files,
)

PricingFunction = Callable[..., tuple[list[dict[str, float]], dict[str, float]]]


@dataclass(frozen=True)
class ProductionPipelineConfig:
    project_root: Path
    model_id: str
    product_id: str
    result_version: str
    model_name: str
    payoff_name: str
    first_seed: int
    default_num_paths: int
    default_target_dt: str | float
    cpp_price: PricingFunction
    python_price: PricingFunction
    source_files_for_engine: Callable[[str], list[str]]
    references_for_engine: Callable[[str], list[dict[str, Any]]]
    time_grid_documentation: Callable[[str | float], dict[str, Any]]
    price_outputs_documentation: Callable[[], dict[str, Any]]
    cpp_delta: PricingFunction | None = None
    python_delta: PricingFunction | None = None
    delta_outputs_documentation: Callable[[], dict[str, Any]] | None = None
    delta_method_documentation: Callable[[float], dict[str, Any]] | None = None
    default_relative_bump: float = 5.0e-4
    production_row_count: int = 1_000
    validation_row_count: int = DEFAULT_VALIDATION_ROW_COUNT
    validation_suffix: str = "first_100"
    summary_details: dict[str, Any] = field(default_factory=dict)
    yaml_details: dict[str, Any] = field(default_factory=dict)
    pricing_kwargs: dict[str, Any] = field(default_factory=dict)


class ProductionPipeline:
    """Generate production CUDA results and independent validation repricings."""

    def __init__(self, config: ProductionPipelineConfig) -> None:
        self.config = config

    @property
    def audit_model_id(self) -> str:
        return f"{self.config.model_id}__{self.config.validation_suffix}"

    @property
    def audit_product_id(self) -> str:
        return f"{self.config.product_id}__{self.config.validation_suffix}"

    def production_result_id(self, *, delta_crn: bool = False) -> str:
        engine = "cpp_gpu_philox_delta_crn" if delta_crn else "cpp_gpu_philox"
        return (
            f"{self.config.model_id}__{self.config.product_id}"
            f"__{engine}_{self.config.result_version}"
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.config.project_root).as_posix()

    def _read_database(
        self, tier: str, kind: str, database_id: str
    ) -> dict[str, Any]:
        path = registry_database_path(
            self.config.project_root, tier, kind, "data", database_id, "json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parameters_by_id(data: dict[str, Any], row_key: str) -> dict[str, Any]:
        return {row["id"]: row["parameters"] for row in data[row_key]}

    def _aligned_rows(self, row_count: int) -> list[dict[str, Any]]:
        return [
            {
                "id": f"{index:06d}",
                "model_id": f"{index:06d}",
                "product_id": f"{index:06d}",
                "seed": self.config.first_seed + index - 1,
            }
            for index in range(1, row_count + 1)
        ]

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _write_result(
        self,
        *,
        tier: str,
        database_id: str,
        rows: list[dict[str, Any]],
        outputs: list[dict[str, float]],
        timing: dict[str, float],
        model_id: str,
        product_id: str,
        engine: str,
        device: str,
        num_paths: int,
        target_dt: str | float,
        relative_bump: float | None = None,
        source_production_result: str | None = None,
    ) -> Path:
        timing = canonical_timing(timing)
        target_json = registry_database_path(
            self.config.project_root, tier, "results", "data", database_id, "json"
        )
        target_yaml = registry_database_path(
            self.config.project_root,
            tier,
            "results",
            "specifications",
            database_id,
            "yaml",
        )
        generator = registry_relative_path(
            self.config.project_root,
            tier,
            "results",
            "generators",
            database_id,
            "py",
        )
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
        construction = aligned_result_construction(self.config.first_seed)
        model_database = database_reference(
            self.config.project_root, tier, "models", model_id
        )
        product_database = database_reference(
            self.config.project_root, tier, "products", product_id
        )
        self._write_json(
            target_json,
            {
                "format": "ai_factory.results.v1",
                "database_id": database_id,
                "status": "priced",
                "specification": self._relative(target_yaml),
                "generation_script": generator,
                "row_count": len(priced_rows),
                "model_database": model_database,
                "product_database": product_database,
                "result_construction": construction,
                "engine": engine,
                "results": priced_rows,
            },
        )

        source_files = primary_source_files(model_id, product_id, engine)
        declared_sources = self.config.source_files_for_engine(engine)
        if source_files[0] not in declared_sources:
            raise ValueError(
                f"Primary implementation {source_files[0]} is missing from "
                f"the declared source dependency list for {database_id}."
            )
        summary = {
            "row_count": len(priced_rows),
            "num_paths": num_paths,
            "model": self.config.model_name,
            "payoff": self.config.payoff_name,
            "engine": engine,
            "device": device,
            **self.config.summary_details,
            "source_files": source_files,
            "references": self.config.references_for_engine(engine),
        }
        delta_crn = "delta_crn" in engine
        outputs_documentation = self.config.price_outputs_documentation()
        if delta_crn:
            if self.config.delta_outputs_documentation is None:
                raise ValueError(f"Delta CRN is not configured for {self.config.payoff_name}.")
            outputs_documentation = self.config.delta_outputs_documentation()

        specification: dict[str, Any] = {
            "title": f"{model_id} x {product_id} {engine}",
            "format": "ai_factory.results.v1",
            "database_id": database_id,
            "status": "priced",
            "json_path": self._relative(target_json),
            "generation_script": generator,
            "summary": summary,
            "time_grid": self.config.time_grid_documentation(target_dt),
            "outputs": outputs_documentation,
            **self.config.yaml_details,
        }
        if delta_crn:
            if relative_bump is None or self.config.delta_method_documentation is None:
                raise ValueError("Delta CRN metadata requires a configured relative bump.")
            specification["delta_method"] = self.config.delta_method_documentation(
                relative_bump
            )
        specification.update(
            {
                "model_database": model_database,
                "product_database": product_database,
                "result_construction": construction,
                "timing": timing,
            }
        )
        if source_production_result is not None:
            specification["source_production_result"] = source_production_result
        self._write_yaml(target_yaml, specification)
        return target_json

    def _inputs(
        self, *, tier: str, model_id: str, product_id: str, row_count: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        models = self._read_database(tier, "models", model_id)
        products = self._read_database(tier, "products", product_id)
        if len(models["models"]) < row_count or len(products["products"]) < row_count:
            raise ValueError("Model and product databases must contain every requested row.")
        return (
            self._aligned_rows(row_count),
            self._parameters_by_id(models, "models"),
            self._parameters_by_id(products, "products"),
        )

    def generate_production_cpp_gpu_result(
        self,
        *,
        row_count: int | None = None,
        num_paths: int | None = None,
        target_dt: str | float | None = None,
        delta_crn: bool = False,
        relative_bump: float | None = None,
    ) -> Path:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the production C++ GPU result.")
        row_count = row_count or self.config.production_row_count
        num_paths = num_paths or self.config.default_num_paths
        target_dt = target_dt or self.config.default_target_dt
        relative_bump = relative_bump or self.config.default_relative_bump
        rows, model_by_id, product_by_id = self._inputs(
            tier="production",
            model_id=self.config.model_id,
            product_id=self.config.product_id,
            row_count=row_count,
        )
        engine = "cpp_gpu_philox_delta_crn" if delta_crn else "cpp_gpu_philox"
        function = self.config.cpp_delta if delta_crn else self.config.cpp_price
        if function is None:
            raise ValueError(f"{engine} is not configured for {self.config.payoff_name}.")
        kwargs: dict[str, Any] = {
            "num_paths": num_paths,
            "target_dt": target_dt,
            "use_gpu": True,
            **self.config.pricing_kwargs,
        }
        if delta_crn:
            kwargs["relative_bump"] = relative_bump
        outputs, timing = function(rows, model_by_id, product_by_id, **kwargs)
        return self._write_result(
            tier="production",
            database_id=self.production_result_id(delta_crn=delta_crn),
            rows=rows,
            outputs=outputs,
            timing=timing,
            model_id=self.config.model_id,
            product_id=self.config.product_id,
            engine=engine,
            device="cuda",
            num_paths=num_paths,
            target_dt=target_dt,
            relative_bump=relative_bump if delta_crn else None,
        )

    def generate_validation_reprice_result(
        self,
        *,
        engine: str,
        device: str,
        row_count: int | None = None,
        num_paths: int | None = None,
        target_dt: str | float | None = None,
        relative_bump: float | None = None,
        benchmark_row_multiplier: int = 1,
    ) -> Path:
        row_count = row_count or self.config.validation_row_count
        num_paths = num_paths or self.config.default_num_paths
        target_dt = target_dt or self.config.default_target_dt
        relative_bump = relative_bump or self.config.default_relative_bump
        rows, model_by_id, product_by_id = self._inputs(
            tier="validation",
            model_id=self.audit_model_id,
            product_id=self.audit_product_id,
            row_count=row_count,
        )
        delta_crn = "delta_crn" in engine
        if engine.startswith("python_"):
            function = self.config.python_delta if delta_crn else self.config.python_price
            kwargs: dict[str, Any] = {
                "num_paths": num_paths,
                "target_dt": target_dt,
                "device": device,
                **self.config.pricing_kwargs,
            }
        elif engine.startswith("cpp_"):
            function = self.config.cpp_delta if delta_crn else self.config.cpp_price
            kwargs = {
                "num_paths": num_paths,
                "target_dt": target_dt,
                "use_gpu": device == "cuda",
                **self.config.pricing_kwargs,
            }
        else:
            raise ValueError(f"Unsupported engine: {engine}")
        if function is None:
            raise ValueError(f"{engine} is not configured for {self.config.payoff_name}.")
        if delta_crn:
            kwargs["relative_bump"] = relative_bump

        repetitions = 1
        outputs, timing = benchmark_pricing_call(
            lambda: function(rows, model_by_id, product_by_id, **kwargs),
            repetitions=repetitions,
            warmup_runs=1,
        )
        timing["benchmark_row_count"] = len(rows)
        timing["benchmark_workload"] = "validation slice"
        if benchmark_row_multiplier > 1:
            benchmark_rows = rows * benchmark_row_multiplier
            _, benchmark_timing = benchmark_pricing_call(
                lambda: function(
                    benchmark_rows,
                    model_by_id,
                    product_by_id,
                    **kwargs,
                ),
                repetitions=repetitions,
                warmup_runs=1,
            )
            timing["benchmark_seconds"] = float(
                benchmark_timing["benchmark_seconds"]
            )
            timing["benchmark_row_count"] = len(benchmark_rows)
            timing["benchmark_row_multiplier"] = benchmark_row_multiplier
            timing["benchmark_repetitions"] = repetitions
            timing["benchmark_workload"] = "duplicated validation slice"
            if "kernel_seconds" in benchmark_timing:
                timing["benchmark_kernel_seconds"] = float(
                    benchmark_timing.get(
                        "benchmark_kernel_seconds",
                        benchmark_timing["kernel_seconds"],
                    )
                )
        database_id = (
            f"{self.audit_model_id}__{self.audit_product_id}"
            f"__{engine}_{self.config.result_version}"
        )
        source_result = None
        if engine.startswith("cpp_gpu_"):
            source_result = registry_relative_path(
                self.config.project_root,
                "production",
                "results",
                "data",
                self.production_result_id(delta_crn=delta_crn),
                "json",
            )
        return self._write_result(
            tier="validation",
            database_id=database_id,
            rows=rows,
            outputs=outputs,
            timing=timing,
            model_id=self.audit_model_id,
            product_id=self.audit_product_id,
            engine=engine,
            device=device,
            num_paths=num_paths,
            target_dt=target_dt,
            relative_bump=relative_bump if delta_crn else None,
            source_production_result=source_result,
        )
